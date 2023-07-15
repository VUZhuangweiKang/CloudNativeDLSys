# FROM python:3.8-alpine
FROM ubuntu:20.04
RUN apt update && apt install -y python3-pip nano tar unzip pigz && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /mnt/nfs/ssd
RUN mkdir -p /mnt/nfs/hdd
RUN mkdir app
WORKDIR /app
COPY manager-worker ./
COPY databus ./databus
COPY cloudbucket ./cloudbucket
COPY configurations ./configurations
COPY commons ./commons
COPY logger ./logger
RUN pip install -r requirements.txt
RUN chmod +x *
CMD [ "python3", "Worker.py" ]
# CMD ["bash"]