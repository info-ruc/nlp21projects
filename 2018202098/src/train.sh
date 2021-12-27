train_bs=40
test_bs=100
python -m torch.distributed.launch --nproc_per_node=3 cli.py --do_train --output_dir out/nq-bart-closed-qa \
        --train_file data/train.json \
        --predict_file data/dev.json \
        --train_batch_size ${train_bs} \
        --predict_batch_size ${test_bs} \
        --append_another_bos \
        --eval_period 500