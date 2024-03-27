# use pytorch base image
FROM python:latest

# set the working directory in the Docker image
WORKDIR /magic

# copy the current directory contents into the container at /unet
COPY . /magic

# recreate enviornment
RUN pip install --no-cache-dir -r requirements.txt
