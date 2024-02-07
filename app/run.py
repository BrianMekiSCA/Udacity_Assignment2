import json
import plotly
import pandas as pd
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine
import matplotlib
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')

app = Flask(__name__)

def tokenize(text):
    '''
    define how to tokenize given text
    normalize the text by,
        conveting to lower case by using text.lower(),
        tokenize text by using word_tokenize(),
        removing all punctuation string.punctuation(),
        removing stop words using the English dictionary by using stopwords.words('english'),
        stem to reduce inflected words using PorterStemmer(),
        determine semmantic relationships between words using WordNetLemmatizer(),
    return clean tokenized text
     
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def identity_tokenizer(tokens):
    '''
    create an identity tokenizer
    '''
    return tokens

# load data
data_db_url = 'sqlite:///data/DisasterResponse.db' 
# ml_model_url = '/models/classifier.pkl'  
ml_model_url = "../models/your_model_name.pkl"

# Create an engine to connect to the database
engine = create_engine(data_db_url)
df = pd.read_sql_table('DisasterResponseTable', engine)

# load model
model = joblib.load(ml_model_url)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)

    genre_counts = df['medical_help'].value_counts()
    genre_names = list(genre_counts.index)

    # create visuals
    
    # Extracting data for the second graph (example histogram of message lengths)
    message_lengths = df['message'].apply(len)
    
    # TODO: Below is an example - modify to create your own visuals
    # create visuals
    
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
                Histogram(
                    x=message_lengths,
                    nbinsx=20
                )
            ],
            'layout': {
                'title': 'Distribution of Message Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Length"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]  # Generate unique IDs for each graph
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)  # Convert graphs to JSON format

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)  # Render the template with graph data


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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()