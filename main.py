from flask import Flask, jsonify, request
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os


modelPath="checkpoint-33800"

tokenizer = Wav2Vec2Tokenizer.from_pretrained(modelPath)
print("**************** TOKENIZER LOADED ****************")
model = Wav2Vec2ForCTC.from_pretrained(modelPath)
print("**************** MODEL LOADED ****************")


def recognizeAudio(audioFile):
    audio, rate = librosa.load(audioFile ,sr=16000)
    input_values = tokenizer(audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(prediction)[0]
    return transcription

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_audio():
    # Catch the audio file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Audio File doesn't exist"

    file = request.files.get('file')

    if not file:
        return

    # Return on a JSON format
    return jsonify(prediction=recognizeAudio(file))


@app.route('/', methods=['GET'])
def index():
    return 'Speech Recognition System'


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
