import sys
import nltk
import re
import numpy as np
import pandas as pd

from datetime import datetime

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from xgboost import XGBClassifier
    
import sqlite3

from sklearn.metrics import f1_score
import pickle

def load_data(database_filepath):
    
    conn = sqlite3.connect(database_filepath)
    
    df = pd.read_sql('Select * From DisasterResponse;', conn, index_col='index')
    
    conn.close()
    
    labels = ['related',
             'request',
             'offer',
             'aid_related',
             'medical_help',
             'medical_products',
             'search_and_rescue',
             'security',
             'military',
             'child_alone',
             'water',
             'food',
             'shelter',
             'clothing',
             'money',
             'missing_people',
             'refugees',
             'death',
             'other_aid',
             'infrastructure_related',
             'transport',
             'buildings',
             'electricity',
             'tools',
             'hospitals',
             'shops',
             'aid_centers',
             'other_infrastructure',
             'weather_related',
             'floods',
             'storm',
             'fire',
             'earthquake',
             'cold',
             'other_weather',
             'direct_report']
    
    X = df[['ml_input']]
    Y = df[[d for d in df.columns if d in labels]]
    
    return X.ml_input.values, Y.values, [d for d in Y.columns]


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


def build_model():

    pipeline_XG = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('multi_clf', MultiOutputClassifier(XGBClassifier(eval_metric='logloss', scale_pos_weight = 20)))
                ])

    return pipeline_XG


def evaluate_model(model, X_test, Y_test, category_names):
    accuracy = []
    Y_pred = model.predict(X_test)

    for k,v in enumerate([d for d in category_names]):
        accuracy.append((v, 
                        f1_score(Y_test[:,k], Y_pred[:,k], zero_division=1), 
                        accuracy_score(Y_test[:,k], Y_pred[:,k]) ))


    return pd.DataFrame(accuracy, columns=['category','f1_score','accuracy'])
    
    
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()