from flask import Flask
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TextClassificationPipeline
from flask import request
import tensorflow
import torch

tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

app = Flask(__name__)
@app.route('/check')
def hello_world():
    result = pipeline(request.args.get("text"))[0]
    return result


if __name__ == '__main__':
    app.run()
