import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

from sqlalchemy import create_engine
import pickle
import nltk
import re

import sqlite3

app = Flask(__name__)

def tokenize(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    text = text.lower().replace("'"," ").replace('"'," ")
    ## remove urls
    url_regexes = ['http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
                   'http (bit|ow).ly+[ ]+[a-z0-9]*']

    for url_regex in url_regexes:    
        text = re.sub(url_regex, " urlplaceholder ", text)
        
    ## remove numbers
    number_regexes = '\d+(\,|\.?)*\d*'
    text = re.sub(number_regexes, " numberplaceholder ", text)

    ## remove all characters but # and @
    character_regexes = '[^a-z0-9#@]' 
    text = re.sub(character_regexes, " ", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    
    tokens = [w for w in tokens if w not in stop_words]
    
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    feature_2word = [' '.join(wds) for wds in zip(clean_tokens, clean_tokens[1:])]
    
    return clean_tokens + feature_2word 


# load data

conn = sqlite3.connect('../data/DisasterResponse.db')
    
df = pd.read_sql('Select * From DisasterResponse;', conn, index_col='index')

# load model
model = pickle.load(open("../models/classifier.pkl", 'rb'))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()