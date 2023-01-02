from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, make_response, abort
import os
import sys
import re, os
from flask import Flask, jsonify, render_template, request
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyvi import ViTokenizer

def main():
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    app = Flask(__name__)
    app.config.from_object(__name__)  # load config from this file , flaskr.py

    # Load default config and override config from an environment variable
    app.config.from_envvar('FLASKR_SETTINGS', silent=True)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    seed = 42
    np.random.seed(seed)
    from keras import backend as K


    def preprocess (raw_input, tokenizer):
        input_text_pre = list(tf.keras.preprocessing.text.text_to_word_sequence(raw_input))
        input_text_pre = " ".join(input_text_pre)
        input_text_pre_accent = ViTokenizer.tokenize(input_text_pre)
        tokenized_data_text = tokenizer.texts_to_sequences([input_text_pre_accent])
        vec_data = pad_sequences(tokenized_data_text, padding = 'post', maxlen = 50)
        return vec_data

    def inference (input, model):
        output = model(input).numpy()[0]
        result = output.argmax()
        conf = float(output.max())
        label_dict = {'negative': -1, 'neutral': 0, 'positive': 1 }
        label = list(label_dict.keys())
        return label[int(result)]

    def prediction (raw_input, tokenizer, model):
        input_model = preprocess(raw_input, tokenizer)
        result, conf = inference(input_model, model)
        return result
    def result (new_text):
        model = load_model('./model/model_lstm_cnn.h5')

        with open ("tokenizer.pkl", "rb") as input_file:
            token = pickle.load(input_file)
        sentiments = prediction(new_text, token, model)
        return sentiments


    # @app.route('/')
    # def home():
    #     return render_template('home.html')

    @app.route('/about')
    def about():
        return 'About Us'

    @app.route('/', methods=['POST', 'GET'])
    def demo():
        if request.method == 'POST':
            if 'sentence' not in request.form:
                flash('No sentence post')
                redirect(request.url)
            elif request.form['sentence'] == '':
                flash('No sentence')
                redirect(request.url)
            else:
                sent = request.form['sentence']
                sentiments= result(sent)
                return render_template('demo_result.html', sentence=sent, sentiments=sentiments)
        return render_template('demo.html')

    @app.errorhandler(404)
    def not_found(error):
        return make_response(jsonify({'error': 'Not found'}), 404)

    app.run(debug=False)


if __name__ == '__main__':
  main.run(host='0.0.0.0', port=5001)
