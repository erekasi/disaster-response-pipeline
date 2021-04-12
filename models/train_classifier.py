# Import libraries
import sys
import numpy as np
import pandas as pd
import copy
import pickle

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


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
        category_names - list of name of the target variables
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
    # Build a machine learning pipeline
    pipeline_rf = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
    ])

    # Set hyper-parameters for tuning
    parameters = {
    'clf__estimator__max_depth': [10, 50]
    }

    # Set up hyper-tuning model
    cv = GridSearchCV(pipeline_rf, param_grid=parameters, cv=4)
    
    # Hyper-tune the model with grid search
    start1 = time.time()
    print('----- Start time of training the model: ', datetime.fromtimestamp(start1), ' -----')
    cv.fit(X_train, Y_train)
    end1 = time.time()
    print('----- Training the random forest multi-output model took: {} minute(s) and {} second(s).'.format((end1-start1)//60, int((end1-start1)%60)))
    
    # Retreive the best estimator model
    model = cv.best_estimator_

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    The function evaluates a multi-output binary classification model.
    
    IN:
        model - multi-output classification model
        X_test - test set of predictors (text message)
        Y_test - test set of the target variables
        category_names - list of name of the target variables
    
    OUT:
        precision, recall, F1-score and accuracy - evaluation metrics for each target variable
        avg_accuracy - average of accuracy per target variable, printed
        avg_recall - average of recall (or sensitivity) per target variable, printed
    '''

    # Make prediction on the test set
    Y_pred = model.predict(X_test)

    # Print precision, recall, F1-score per label per output
    # and accuracy score per output
    print('\n')
    print('======= Report on the Hyper-tuned multi-output random forest classifier model =======')
    print('\n')

    for i, col in enumerate(category_names):
        print('({0}) category: {1}'.format(i+1, col))
        print('\n')
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy score = %.4f' %accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i]))
        print('----------------------------')
        print('\n')
    
    # Calculate and print average accuracy score
    all_accuracy = []
    for i in range(len(category_names)):
        all_accuracy.append(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i]))
    avg_accuracy = sum(all_accuracy) / len(all_accuracy)
    print('---- Average accuracy of the hyper-tuned model is {} ---- '.format(round(avg_accuracy,4)))
    
    # Calculate average recall
    all_recall = []
    for i in range(len(category_names)):
        all_recall.append(recall_score(Y_test.iloc[:, i].values, Y_pred[:,i], average='macro'))
    print('---- Average recall with macro average for the hyper-tuned model is %.4f ----' %np.mean(all_recall))

def save_model(model, model_filepath):
    '''
    The function saves the model using pickle.
    
    IN:
       model - hyper-tuned multi-output binary classification model
       model_filepath - filepath to where the model has to be saved
    OUT:
        The model gets saved under the name inserted as model_filepath.
    '''

    # Open a file named model_filepath with wb as write bytes mode
    # and store the reference to that file in variable f.
    # Save the object (the model) in that file.
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

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