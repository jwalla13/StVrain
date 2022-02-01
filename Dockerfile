FROM python:3.7

WORKDIR /stvrain-docker

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./src ./src

COPY ./tmp ./tmp

CMD ["python", "./src/split_video.py", "-rmi", "-c", "-sv", "-rdf"]

