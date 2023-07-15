FROM ubuntu:20.04
RUN apt update && apt install -y python3-pip nano nfs-common vmtouch
RUN mkdir app
RUN mkdir -p /mnt/nfs/ssd
RUN mkdir -p /mnt/nfs/hdd
WORKDIR /app
COPY client ./
COPY databus ./databus
COPY database ./database
COPY configurations ./configurations
COPY logger ./logger
RUN pip3 install -r requirements.txt
RUN chmod +x *
CMD [ "bash" ]
# CMD [ "python3", "client.py" ]