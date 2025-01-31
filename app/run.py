#%%
# Import libraries
import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

#%%
app = Flask(__name__)

#%%
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#%%
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterTable', engine)

#%%
# load model
model = joblib.load("../models/classifier.pkl")

#%%
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_prop = round(100*genre_counts/genre_counts.sum(), 2)
    genre_names = list(genre_counts.index)
    category_names = df.iloc[:, 4:].columns
    category_sums = (df.iloc[:, 4:] !=0).sum().values
    
    # create visuals
    graphs = [
        {
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "hole": 0.4,
                "name": "Genre composition (%)",
                "pull": 0,
                "domain": {
                  "x": genre_prop,
                  "y": genre_names
                },
                "marker": {
                  "colors": [
                    "#581845",
                    "#900c3f",
                    "#c70039"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre_names,
                "values": genre_prop
              }
            ],
            "layout": {
              "title": "Distribution of messages by genre"
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_sums
                )
            ],

            'layout': {
                'title': 'Prevalence of message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

#%%
# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

#%%
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
# %%
