# FROM python:3.8-alpine
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

RUN apt update && apt install -y python3-pip nano tar unzip pigz
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y openjdk-11-jre

RUN mkdir -p /mnt/nfs/ssd
RUN mkdir -p /mnt/nfs/hdd
RUN mkdir app

WORKDIR /app
COPY manager ./manager
COPY commons ./commons
COPY databus ./databus
COPY database ./database
COPY cloudbucket ./cloudbucket
COPY ioperf ./ioperf
COPY configurations ./configurations
COPY logger ./logger
COPY k8s ./k8s
WORKDIR /app/manager
RUN pip install -r requirements.txt
RUN chmod +x *
CMD ["bash"]