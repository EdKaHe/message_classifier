import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import re
from sqlalchemy import create_engine


app = Flask(__name__)

# get the length of each message
class MessageLength(BaseEstimator, TransformerMixin):
    """
    class that extracts the message length from a text message
    methods:
        msg_len: returns the number of words in a text message
        fit: fits the estimator to the given messages
        transform: gets the length of each text message
    """
    
    def msg_len(self, msg):
        return len(tokenize_word(msg))

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).apply(self.msg_len)
        return pd.DataFrame(X)

def tokenize_word(text):
    """
    function that returns a list of lemmatized and stemmed
    words of a text message
    arguments:
        text: string that contains a text message
    return:
        token: list of lemmatized and stemmed words 
    """
    
    # transform string to lowercase
    text = text.lower()
    # substitute any characters but letters and numbers
    text = re.sub(r"[^a-z0-9!?]", " ", text)
    # tokenize words
    token = word_tokenize(text)
    # reduce words to their stem
    token =  [PorterStemmer().stem(tok) for tok in token]
    # lemmatize the words
    token = [WordNetLemmatizer().lemmatize(tok) for tok in token]
    
    return token

# extract the parts of speech in each message
def tokenize_pos(msg):
    """
    loads dataframes from specified filepaths
    arguments:
        msg: string that contains a text message
    return:
        pos_tags: part of speech tags of the given message 
    """
    
    tokens = tokenize_word(msg)
    try:
        pos_tags = list(zip(*nltk.pos_tag(tokens)))[1]
    except:
        pos_tags = 'INVALID'
    
    pos_tags = list(pos_tags)
        
    return pos_tags

# load data
engine = create_engine('sqlite:///..//data//DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # data for second graph
    category_names = df.columns[4:]
    category_counts = df[category_names].sum().tolist()
    # sort the data
    category_names = [x for _, x in sorted(zip(category_counts, category_names))]
    category_counts = sorted(category_counts)
    print(category_counts)
    
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
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
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
    classification_results = dict(zip(df.columns[4:], classification_labels))

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