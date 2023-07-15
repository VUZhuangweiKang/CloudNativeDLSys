FROM pytorch/pytorch:latest
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get update && apt-get install -y nano git libsndfile1 sox liblzma-dev libgl1-mesa-glx libglib2.0-0 vmtouch && \
    rm -rf /var/lib/apt/lists/* 

RUN mkdir app
WORKDIR /app
COPY UCF101/ .
COPY lib ./lib
RUN pip install -r requirements.txt
RUN chmod +x *
RUN ./entrypoint.sh
CMD ["bash"]
