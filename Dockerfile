FROM python:3.6

WORKDIR /home/yhoatsu/

RUN apt-get -y update
RUN apt-get install -y --fix-missing cmake

COPY ./ ./
RUN pip install --no-cache-dir -r requirements.txt


