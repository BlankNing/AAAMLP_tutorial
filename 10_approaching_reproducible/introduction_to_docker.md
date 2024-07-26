# 1 Build Docker container and run codes through docker command
You wrote 
code on your computer and that might not work on someone else’s computer 
because of many different reasons. So, it would be nice if when you distribute the 
code, you could replicate your computer and others can too when they install your 
software or run your code. To do this, the most popular way these days is to use 
Docker Containers. 

Docker containers can be considered as small virtual machines. You can create a 
container for your code, and then everyone will be able to use it and access it.

First and foremost, you need a file with requirements for your python project. 
Requirements are contained in a file called requirements.txt.

``` pip freeze > requirements.txt ```

Now, we will create a docker file called Dockerfile. No extension. There are several 
elements to Dockerfile. 

```dockerfile
# Dockerfile
# First of all, we include where we are getting the image
# from. Image can be thought of as an operating system.
# You can do "FROM ubuntu:18.04"
# this will start from a clean ubuntu 18.04 image.
# All images are downloaded from dockerhub
# Here are we grabbing image from nvidia's repo
# they created a docker image using ubuntu 18.04
# and installed cuda 10.1 and cudnn7 in it. Thus, we don't have to 
# install it. Makes our life easy.
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# this is the same apt-get command that you are used to
# except the fact that, we have -y argument. Its because
# when we build this container, we cannot press Y when asked for
RUN apt-get update && apt-get install -y \
 git \
 curl \
 ca-certificates \
 python3 \
285
 python3-pip \
 sudo \
 && rm -rf /var/lib/apt/lists/*
# We add a new user called "abhishek"
# this can be anything. Anything you want it
# to be. Usually, we don't use our own name,
# you can use "user" or "ubuntu"
RUN useradd -m abhishek
# make our user own its own home directory
RUN chown -R abhishek:abhishek /home/abhishek/
# copy all files from this direrctory to a 
# directory called app inside the home of abhishek
# and abhishek owns it.
COPY --chown=abhishek *.* /home/abhishek/app/
# change to user abhishek
USER abhishek
RUN mkdir /home/abhishek/data/
# Now we install all the requirements
# after moving to the app directory
# PLEASE NOTE that ubuntu 18.04 image
# has python 3.6.9 and not python 3.7.6
# you can also install conda python here and use that
# however, to simplify it, I will be using python 3.6.9
# inside the docker container!!!!
RUN cd /home/abhishek/app/ && pip3 install -r requirements.txt
# install mkl. its needed for transformers
RUN pip3 install mkl
# when we log into the docker container,
# we will go inside this directory automatically
WORKDIR /home/abhishek/app
```

Once we have created the docker file, we need to build it. 

```docker build -f Dockerfile -t bert:train .```

This command builds a container from the provided Dockerfile. The name of the 
docker container is bert:train.

Now, you can log into 
the container using the following command.

```$ docker run -ti bert:train /bin/bash```

Note that any change in the python scripts, means that the docker container needs 
to be rebuilt!

So, with some very simple changes, you have now “dockerized” your training code. 
You can now take this code and train on (almost) any system you want.

# 2 Serving it to the end-user

The next part is “serving” this model that we have trained to the end-user. Suppose, 
you want to extract sentiment from a stream of incoming tweets. To do this kind of 
task, you must create an API that can be used to input the sentence and in turns 
returns an output with sentiment probabilities. The most common way of building 
an API using Python is with Flask, which is a micro web service framework.

```python 

# api.py
import config
import flask
import time
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from model import BERTBaseUncased

app = Flask(__name__)
MODEL = None
DEVICE = "cuda"


def sentence_prediction(sentence):
    """
    A prediction function that takes an input sentence
    and returns the probability for it being associated
    to a positive sentiment
    """
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN

    review = str(sentence).strip()

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len,
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids += ([0] * padding_length)
    mask += ([0] * padding_length)
    token_type_ids += ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(DEVICE)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()

    return outputs[0][0]


@app.route("/predict", methods=["GET"])
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()

    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction

    response = {
        "response": {
            "positive": str(positive_prediction),
            "negative": str(negative_prediction),
            "sentence": str(sentence),
            "time_taken": str(time.time() - start_time),
        }
    }

    return jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(
        torch.load(config.MODEL_PATH, map_location=torch.device(DEVICE))
    )
    MODEL.to(DEVICE)
    MODEL.eval()

    app.run(host="0.0.0.0")

```

And you start the API by running the command “python api.py”. The API will start 
on localhost on port 5000

comparing between the response from curl and directly access to the page through browser:

```
❯ curl 
$'http://192.168.86.48:5000/predict?sentence=this%20is%20the%20best%20boo
k%20ever'
{"response":{"negative":"0.0032927393913269043","positive":"0.99670726","
sentence":"this is the best book 
ever","time_taken":"0.029126882553100586"}}
```

```
{
 response: {
 negative: "0.8646619468927383",
 positive: "0.13533805",
 sentence: "this book is too complicated for me",
 time_taken: "0.03852701187133789"
 }
}
```
Now, we have created a simple API that we can use to serve a small number of 
users. Why small? Because this API will serve only one request at a time. Let’s use 
CPU and make it work for many parallel requests using gunicorn which is a python 
WSGI HTTP server for UNIX. Gunicorn can create multiple processes for the API, 
and thus, we can serve many customers at once. You can install gunicorn by using 
“pip install gunicorn”.

To convert the code compatible with gunicorn, we need to remove init main and 
move everything out of it to the global scope. Also, we are now using CPU instead 
of the GPU. See the modified code in api_gunicorn.py

And we run this API using the following command.

```$ gunicorn api_gunicorn:app --bind 0.0.0.0:5000 --workers 4```

This means we are running our flask app with 4 workers on a provided IP address 
and port. Since there are 4 workers, we are now serving 4 simultaneous requests. Please note that now our endpoint uses CPU and thus, it does not need a GPU 
machine and can run on any standard server/VM. Still, we have one problem, we 
have done everything in our local machine, so we must dockerize it. Take a look at 
the following uncommented Dockerfile which can be used to deploy this API. 

```dockerfile
# CPU Dockerfile
FROM ubuntu:18.04
RUN apt-get update && apt-get install -y \
 git \
 curl \
 ca-certificates \
 python3 \
 python3-pip \
 sudo \
 && rm -rf /var/lib/apt/lists/*
RUN useradd -m abhishek
RUN chown -R abhishek:abhishek /home/abhishek/
COPY --chown=abhishek *.* /home/abhishek/app/
USER abhishek
RUN mkdir /home/abhishek/data/
RUN cd /home/abhishek/app/ && pip3 install -r requirements.txt
RUN pip3 install mkl
WORKDIR /home/abhishek/app
```

Let's build a new docker container:

```$ docker build -f Dockerfile -t bert:api .```

When the docker container is built, we can now run the API directly by using the 
following command.

```$ docker run -p 5000:5000 -v  /home/abhishek/workspace/approaching_almost/input/:/home/abhishek/data/ -ti bert:api /home/abhishek/.local/bin/gunicorn api:app --bind 0.0.0.0:5000 --workers 4```

Please note that we expose port 5000 from the container to 5000 outside the 
container. This can also be done in a nice way if you use docker-compose. Docker compose is a tool that can allow you to run different services from different or the 
same containers at the same time. You can install docker-compose using “pip install 
docker-compose” and then run “docker-compose up” after building the container. 
To use docker-compose, you need a docker-compose.yml file.

```dockerfile

# docker-compose.yml
# specify a version of the compose
version: '3.7'
# you can add multiple services
services:
 # specify service name. we call our service: api 
 api:
 # specify image name
 image: bert:api
 # the command that you would like to run inside the container
 command: /home/abhishek/.local/bin/gunicorn api:app --bind 
0.0.0.0:5000 --workers 4
 # mount the volume
 volumes:
 -
/home/abhishek/workspace/approaching_almost/input/:/home/abhishek/data/
 # this ensures that our ports from container will be 
 # exposed as it is
 network_mode: host
 
 ```

 Now you can rerun the API just by using the command mentioned above, and it 
will work the same way as before. Congratulations! Now you have managed to 
dockerized the prediction API too, and it is ready for deployment anywhere you 
want.

 In this chapter, we learned docker, building APIs using flask, serving API 
using gunicorn and docker and docker-compose. There is a lot more to docker 
than we have seen here, but this should give you a start. Rest can be learned as you 
progress. We have also skipped on many tools like **kubernetes, bean-stalk, 
sagemaker, heroku and many others** that people use these days for deploying 
models in production. 
Remember that once 
you have dockerized your application, deploying using any of these 
technologies/platforms is a piece of cake. Always remember to make your code and 
model usable and well-documented for others so that anyone can use what you have 
developed without asking you several times. 