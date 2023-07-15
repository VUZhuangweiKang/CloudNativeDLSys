FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3-pip fio iproute2 bc nano vmtouch && rm -rf /var/lib/apt/lists/*

WORKDIR /app/
COPY ioperf/ /app/
RUN chmod +x *
RUN pip3 install pyzmq numpy
