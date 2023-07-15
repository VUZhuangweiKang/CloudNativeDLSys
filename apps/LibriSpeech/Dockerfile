FROM pytorch/pytorch:latest

# Update and install dependencies in one RUN command
RUN apt-get update && apt-get install -y nano git libsndfile1 sox liblzma-dev vmtouch && \
    rm -rf /var/lib/apt/lists/*  # Clear apt cache

WORKDIR /app

# Only copy required files
COPY LibriSpeech/ .
COPY lib ./lib

# Install Python dependencies in one RUN command and clean cache
RUN pip3 install --no-cache-dir -r requirements.txt -e . && \
    pip3 uninstall -y torch torchaudio && \
    pip3 install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Run entrypoint.sh
RUN chmod +x *
RUN ./entrypoint.sh

CMD [ "bash" ]
