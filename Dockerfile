FROM python:3.6

WORKDIR /home/yhoatsu/IdeaProjects/ML_News_Face_Recognition

RUN apt-get -y update
RUN apt-get install -y --fix-missing cmake

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ./images/training/Pedro_Sanchez/5b3e87167470e.jpeg ./
COPY ./untitled.mp4 ./
COPY ./face_recognition.py ./
COPY ./pickles/* ./pickles/
COPY ./models/* ./models/

CMD [ "python", "face_recognition.py", "5_face_landmarks", "cnn", "./untitled.mp4", "1", "no", "0.55"]

