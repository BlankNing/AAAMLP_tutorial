FROM python:3.8-slim-buster
WORKDIR /tutorial
COPY . .
RUN pip3 install -r requirements.txt
CMD [ "python", "main.py" ]