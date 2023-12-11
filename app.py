from flask import Flask, request, jsonify, render_template

from flask_bootstrap import Bootstrap
from flask_appconfig import AppConfig
from flask_wtf import FlaskForm, RecaptchaField
from wtforms import StringField, ValidationError
from wtforms.fields import HiddenField, RadioField, BooleanField, SubmitField
from wtforms.validators import DataRequired

from wtforms import StringField, SubmitField

import gevent.monkey
from gevent.pywsgi import WSGIServer
gevent.monkey.patch_all()

import re

import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

app = Flask(__name__)

dataset = load_dataset("wiki_qa", split="train[:100]")  # Load the first 100 rows


model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def ask_question(data):
    translator = Translator()
    question = data
    lang = translator.detect(question).lang
    
    if lang == 'en':
        # If the text is in English, return it as is
        question
    else:
        # Translate the text to English
        translation = translator.translate(question, dest='en')
        question= translation.text


    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['answer'])
    similarity_scores = cosine_similarity(tfidf_vectorizer.transform([question]), tfidf_matrix)
    sorted_indices = similarity_scores.argsort()[0][::-1]
    top_n = 5 
    ba = [dataset['answer'][index] for index in sorted_indices[:top_n]]
    cosine_ans=''
    for  a in enumerate(ba):
        cosine_ans=cosine_ans+a[1]
    

    # Initialize variables to keep track of the best answer and its confidence
    best_answer = None
    best_confidence = 0.0

    
    context = cosine_ans
    text = context
    translation = translator.translate(text, dest=lang)
    text= translation.text

    # print("\nFull Description:")
    # print(text)
    # print()
        # Tokenize the question and context
    encoding = tokenizer.encode_plus(text=question,text_pair=context)

    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens
    start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]),return_dict=False)

    start_index = torch.argmax(start_scores)

    end_index = torch.argmax(end_scores)

    answer = ' '.join(tokens[start_index:end_index+1])
        # Calculate the confidence score (sum of start and end scores)
    corrected_answer = ''

    for word in answer.split():

    #If it's a subword token
       if word[0:2] == '##':
          corrected_answer += word[2:]
       else:
          corrected_answer += ' ' + word
    
    translation = translator.translate(corrected_answer, dest=lang)
    corrected_answer= translation.text
    # print("Summarize Answer:")
    # print(corrected_answer)
    return corrected_answer, text

class ExampleForm(FlaskForm):
    question = StringField('', description='', validators=[DataRequired()])
    submit_button = SubmitField('Go')

def create_app(configfile=None):
    app = Flask(__name__)
    AppConfig(app, configfile)
    Bootstrap(app)

    app.config['SECRET_KEY']= 'ffedg0890489574'

    @app.route('/', methods=('GET', 'POST'))
    def index():
        if request.method == 'POST':
            question = request.form['question']
            # print(question)
            answer, text = ask_question(question)
            # print ("answer: "),answer
            return render_template('answer.html', answer=answer, question=question, text=text)

        form = ExampleForm()
        return render_template('index.html', form=form)
    return app

# create main callable
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
