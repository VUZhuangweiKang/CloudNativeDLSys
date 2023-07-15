FROM ubuntu:20.04
RUN apt update && apt install -y python3-pip nano
RUN mkdir app
WORKDIR /app
COPY frontend ./
COPY cloudbucket ./cloudbucket
COPY database ./database
COPY databus ./databus
COPY commons ./commons
COPY logger ./logger
COPY configurations ./configurations
RUN pip3 install -r requirements.txt
RUN chmod +x *
# CMD [ "python3", "FrontEnd.py" ]
CMD [ "bash" ]