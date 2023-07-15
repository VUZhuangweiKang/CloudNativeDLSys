#!/bin/bash

if [ $1 == "deepspeech" ]
then
    docker rmi zhuangweikang/deepspeech-dev:latest
    docker build -t zhuangweikang/deepspeech-dev:latest -f LibriSpeech/Dockerfile .
    docker push zhuangweikang/deepspeech-dev:latest
elif [ $1 == "imagenet" ]
then
    docker rmi zhuangweikang/imagedatasets-dev:latest
    docker build -t zhuangweikang/imagedatasets-dev:latest -f ImageNet/Dockerfile .
    docker push zhuangweikang/imagedatasets-dev:latest
elif [ $1 == "ucf" ]
then
    docker rmi zhuangweikang/ucf-dev:latest
    docker build -t zhuangweikang/ucf-dev:latest -f UCF101/Dockerfile .
    docker push zhuangweikang/ucf-dev:latest
fi