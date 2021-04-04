# Import libraries
import sys
import numpy as np
import pandas as pd
import copy

import time
from datetime import datetime

import sqlite3
from sqlalchemy import create_engine

# NLP relevant libraries
import re #library to remove punctuation with a regular expression (regex)
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet']) #necessary for word tokenization, to remove stopwords, and for lemmatization
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer # for lemmatization
from nltk.tokenize import word_tokenize

# Scikit-learn libraries
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    The function loads the pre-processed message and category datasets
    from a SQLite database into a dataframe and creates a predictor (X) and a target (Y) datasets
    and a list of the target names (category names).

    IN:
        database_filepath - name of the SQL database where the relevant table can be found
    OUT:
        X - array of message texts
        Y - array of binary labels for each category per message record
        category_names - list of name of the target values
    '''
    
    # Create engine where the data comes from
    engine = create_engine('sqlite:///'+ database_filepath)
    
    # Load data from the SQLite database into a dataframe
    df = pd.read_sql_table('DisasterTable', engine)
    
    # Create predictor and target datasets
    X = df.message.copy()
    Y = df.iloc[:, 4:].copy()

    # Category names for the final model evaluation report
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    '''
    The function tokenizes the text after normalizing it, including transforming it to lower case,
    removing punctuation and stopwords and then it applies lemmatization on the tokens.
    
    IN: text - text of a message related to a disaster
    
    OUT: clean_tokens - tokenized text
    '''
    
    # Normalize to lower case
    text = text.lower()
    
    # Normalize by removing punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenize words
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words('english')]
    
    # Lemmatize words
    clean_tokens = []
    
    for w in words:
        clean_t = WordNetLemmatizer().lemmatize(w, pos='n').strip()
        clean_t = WordNetLemmatizer().lemmatize(clean_t, pos='v')
        clean_tokens.append(clean_t)
    
    return clean_tokens


def build_model():
    '''
    The function builds a machine learning pipeline
    that conducts natural language processing and classification of text messages.

    OUT:
        model - best estimator of the hyper-tuned model
    '''
    

def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
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