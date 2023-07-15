#!/bin/bash

# Paper: https://assets.amazon.science/d5/2b/589dac2b4da58f5af4573d9a3959/knowledge-distillation-via-module-replacing-for-automatic-speech-recognition-with-recurrent-neural-network-transducer.pdf
# spends 38 hours to train LibriSpeech for 100 epochs with batch size 256 using 8 Nvidia V100 16GB GPUs

# Paper: https://ieeexplore-ieee-org.proxy.library.vanderbilt.edu/stamp/stamp.jsp?tp=&arnumber=9053889
# spends 122 hours to train LibriSpeech (seems train-clean-100 only) for 400 epochs with batch size 256 using 8 Tesla V100 GPUs

i=0
w=8
test=1
compute_time=( 0.5 1.0 1.5 2.0 2.5 3.0 )
batch_size=( 1024 )
node="10.140.81.191"
tune_worker=0

total_test=$((${#compute_time[@]} * ${#batch_size[@]}))
for t in ${compute_time[*]}
do
    for b in ${batch_size[*]}
    do
        data_dir=data/run$i/$t/$b
        mkdir -p $data_dir
        vmtouch -e /$node/
        echo "Exp[$test/$total_test]: worker=$w, batch_size=$b, compute_time=$t"
        echo "$w,$b,1,100,10,$t,$tune_worker" > /app/dlcache_exp.txt
        python train.py +configs=librispeech
        mv /tmp/*.npy $data_dir/
        # mv /share/train_cache_usage.npy $data_dir/
        python3 report.py $i $t $b
        ((test=test+1))
    done
done
