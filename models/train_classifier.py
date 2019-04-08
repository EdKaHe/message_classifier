import sys
from sqlalchemy import create_engine
import pandas as pd
import pickle
import re

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

nltk.download(['wordnet', 'punkt', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath.replace('/', '//'))
    df = pd.read_sql_table('messages', con=engine)
    # define the category names
    category_names = ['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']
    # seperate predictor and response variable
    X = df['message']
    y = df[category_names].values
                           
    return X, y, category_names
                           
# extract the parts of speech in each message
def tokenize_pos(msg):
    tokens = tokenize_word(msg)
    try:
        pos_tags = list(zip(*nltk.pos_tag(tokens)))[1]
    except:
        pos_tags = 'INVALID'
    
    pos_tags = list(pos_tags)
        
    return pos_tags
                         
# get the length of each message
class MessageLength(BaseEstimator, TransformerMixin):
    def msg_len(self, msg):
        return len(tokenize_word(msg))

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X).apply(self.msg_len)
        return pd.DataFrame(X)

def tokenize_word(text):
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

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
            ])),
            ('pos_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize_pos)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('msg_len', MessageLength()),
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100))),
    ])
                           
    return pipeline
                         
def evaluate_model(model, X_test, y_test, category_names):    
    # predict the test data
    y_pred_test = model.predict(X_test)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

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