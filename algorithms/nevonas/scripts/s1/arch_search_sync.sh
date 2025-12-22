#!/bin/bash
# bash ./s1/arch_search_test1.sh cifar10 gpu outputs data_path

echo script name: $0
echo $# arguments

config_path="./s2/configs/CMAES-NAS.config"
config_root="./s2/configs"

dataset=$1
gpu=$2
#output_dir=$3
#data_path=$4
epochs=30
train_epochs=1
train_discrete=0
init_channel=16
layers=5
knn=5
novelty_threshold=0.0
mutate_rate=0.1
pop_size=40

batch_size=96
valid_batch_size=1024

if [ $train_discrete -gt 0 ]
then
  python3 ./s1/arch_search.py --gpu ${gpu} --init_channel ${init_channel} --layers ${layers} --dataset ${dataset} \
                                 --epochs ${epochs} --train_epochs ${train_epochs} --knn ${knn} \
                                 --mutate_rate ${mutate_rate} --pop_size ${pop_size} --train_discrete
else
  python3 ./s1/arch_search.py --gpu ${gpu} --init_channel ${init_channel} --layers ${layers} --dataset ${dataset} \
                                 --epochs ${epochs} --train_epochs ${train_epochs} --knn ${knn} \
                                 --mutate_rate ${mutate_rate} --pop_size ${pop_size}
fi
