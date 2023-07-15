#!/bin/bash

docker rmi -f cndlsys-dev:$1
docker rmi -f zhuangweikang/cndlsys-dev:$1
docker build -t cndlsys-dev:$1 -f $1/Dockerfile .
docker tag cndlsys-dev:$1 zhuangweikang/cndlsys-dev:$1
docker push zhuangweikang/cndlsys-dev:$1