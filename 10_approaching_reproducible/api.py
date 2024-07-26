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
