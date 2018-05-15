#/bin/bash
python2 train/train.py --cpu --model frustum_pointnets_v1 --log_dir train/log_v1 --num_point 1024 --max_epoch 201 --batch_size 1 --decay_step 800000 --decay_rate 0.5
