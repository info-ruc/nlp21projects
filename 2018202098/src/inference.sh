test_bs=100
python cli.py --do_predict --output_dir out/nq-bart-closed-qa \
        --predict_file src/dataset/dev.json \
        --predict_batch_size ${test_bs} \
        --append_another_bos --prefix dev_
python cli.py --do_predict --output_dir out/nq-bart-closed-qa \
        --predict_file src/dataset/test.json \
        --predict_batch_size ${test_bs} \
        --append_another_bos --prefix test_