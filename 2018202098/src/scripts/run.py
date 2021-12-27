import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


from transformers import BertTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from data import QAData
from bart import ClosedQABart


def run(args, logger):

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    train_data = QAData(logger, args, args.train_file, True)
    dev_data = QAData(logger, args, args.predict_file, False)

    train_data.load_dataset(tokenizer)
    # train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    # dev_data.load_dataloader()

    if args.do_train:
        if args.checkpoint is not None:

            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith("module."):
                        return key[7:]
                    return key

                return {_convert(key): value for key, value in state_dict.items()}

            model = ClosedQABart.from_pretrained(
                args.pretrained_model_path,
                state_dict=convert_to_single_gpu(torch.load(args.checkpoint)),
            )
        else:
            model = ClosedQABart.from_pretrained(args.pretrained_model_path)

        if args.local_rank == -1:
            model = torch.nn.DataParallel(model)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
            ws = 1
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
            torch.distributed.init_process_group(backend="nccl")
            n_gpu = 1
        model.to(device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        train_data.load_dataloader()
        dev_data.load_dataloader()
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=100000
        )
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        checkpoint = os.path.join(args.output_dir, "best-model.pt")

        def convert_to_single_gpu(state_dict):
            def _convert(key):
                if key.startswith("module."):
                    return key[7:]
                return key

            return {_convert(key): value for key, value in state_dict.items()}

        model = ClosedQABart.from_pretrained(
            args.pretrained_model_path,
            state_dict=convert_to_single_gpu(torch.load(checkpoint)),
        )
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model, dev_data, save_predictions=True)
        logger.info(
            "%s on %s data: %.2f"
            % (dev_data.metric, dev_data.data_type, np.mean(ems) * 100)
        )


def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training = False

    writer = SummaryWriter(args.output_dir)
    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in train_data.dataloader:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            loss = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                decoder_input_ids=batch[2],
                decoder_attention_mask=batch[3],
                is_training=True,
            )
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()
            if args.local_rank == 0:
                writer.add_scalar("Train/Loss", train_losses[-1], global_step)
            if global_step % args.train_loss_report_period == 0:
                logger.info(
                    "Step %d at rank %d Train loss %.2f",
                    global_step,
                    args.local_rank,
                    float(train_losses[-1]),
                )

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enought gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0 and args.local_rank <= 0:
                model.eval()
                curr_em = inference(model, dev_data)
                writer.add_scalar("Dev/EM", curr_em * 100, global_step)
                logger.info(
                    "Step %d Train loss %.2f %s %.2f%% on epoch=%d"
                    % (
                        global_step,
                        np.mean(train_losses),
                        dev_data.metric,
                        curr_em * 100,
                        epoch,
                    )
                )
                train_losses = []
                if best_accuracy < curr_em:
                    model_state_dict = {
                        k: v.cpu() for (k, v) in model.state_dict().items()
                    }
                    torch.save(
                        model_state_dict, os.path.join(args.output_dir, "best-model.pt")
                    )
                    logger.info(
                        "Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d"
                        % (
                            dev_data.metric,
                            best_accuracy * 100.0,
                            curr_em * 100.0,
                            epoch,
                            global_step,
                        )
                    )
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break


def inference(model, dev_data, save_predictions=True):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    for i, batch in enumerate(dev_data.dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(
            input_ids=batch[0],
            attention_mask=batch[1],
            num_beams=dev_data.args.num_beams,
            max_length=dev_data.args.max_output_length,
            early_stopping=True,
        )
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions))
