FROM python:3.8-slim-buster

MAINTAINER nandan pandey <pandeynandancse@gmail.com>




WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt                                                                            

EXPOSE 5000

#ENTRYPOINT  ["python3"]

CMD ["flask", "run", "--host", "0.0.0.0" ]
