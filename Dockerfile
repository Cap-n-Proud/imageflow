#docker build -t scripts .
#https://github.com/borgmatic-collective/docker-borgmatic/tree/master/base
#https://gist.github.com/andyshinn/3ae01fa13cb64c9d36e7

#docker run -it -d -v  /home/paolo/NAS/Apps_Config:/mnt/Apps_Config -v /home/paolo/NAS/VDev-BCK:/mnt/VDev-BCK borgmatic
#docker exec -it cc2dac2eab5f /bin/sh

# Add hostpath as per source directory (e.g.: /mnt/Documents) and repositories (e.g.: mnt/VDev-BCK/Borgmatic/Documents). See congis files.

# Each instruction in this file generates a new layer that gets pushed to your local image cache
# Lines preceeded by # are regarded as comments and ignored

FROM ubuntu:latest

#Identify the maintainer of an image
#LABEL maintainer="myname@somecompany.com"

ENV HOSTNAME paddleocr-docker

# Update the image to the latest packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
	nano \
	wget \
	sudo \
	screen \
	python3 \
	python3-pip \
  sshfs \
  openssl \
  curl \
  bash \
  bash-completion \
	bash-doc \
	&& rm -rf /var/cache/apt/*

WORKDIR /app
#requirements.txt must contain all the required libraries in order to run every script
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD /bin/bash /app/start.sh
