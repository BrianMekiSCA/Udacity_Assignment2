# Import Libraries
import sys
import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import wordnet
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import training data 
def load_data(database_filepath):
    '''
    import training data from SQL database and store in object called df
    use function tokenize() to tokenize messages column
    merge tokenized data with original data frame df and return result
     
    '''
    # Create an SQLAlchemy engine to connect to the database
    engine = create_engine('sqlite:///'+database_filepath)

    # Query the database and load data into a DataFrame
    df = pd.read_sql_query('SELECT * FROM DisasterResponseTable', engine)
    
    # Clean data by replacing NaNs with 0
    df = df.dropna()

    # Update messages data frame 
    df['tokenized_text'] = df['message'].apply(tokenize)
            
    # Define predicted and predictor variables
    X = df['tokenized_text'].apply(lambda x: ' '.join(x))
    Y = df.iloc[:, 4:-1]
    category_colnames = Y.columns
    
    # Return results
    return X, Y, category_colnames

# Define tokenizing function
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
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Removing Punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokenized_list = [stemmer.stem(token) for token in tokens]
    
    # Cleaning
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokenized_list:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    # Return results
    return clean_tokens

# Define a function to join the tokenized words back into strings
def identity_tokenizer(tokens):
    '''
    define a tokenizer that joins the tokenized words back into strings
    '''
    return tokens
    
# Build model pipeline
def build_model():

    '''
    create a machine learning pipeline by,
        first determine importance of different words in the tokenized data using the tfi-df method,
        then fit a multi-output model initialized on a random forest classifier 
        implement grid serch to find the best model parameters
        
    '''
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer = identity_tokenizer, lowercase=True, stop_words='english')),  # Vectorize tokenized text data using TF-IDF
    ('clf', RandomForestClassifier())  # Multi-output Random Forest Classifier
    ])
    
    # Define the parameter grid to search
    parameters = {
        'tfidf__max_features': [1000, 5000, 10000],
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20]
    }
    
    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
    
    return grid_search

# Test and evaluate model 
def evaluate_model(model, X_test, y_test, category_names):
    
    '''
    test trained model using X_test as input
    convert predicted output to a dataframe whose column names are category_names
    evaluate how well the predictions are by comparing them to the given y_test data
    carry out the evaluation process by determining the f1 score, the accuracy, recalland precision measures result for each column in the predicted output
    
    '''
    
    # Predict 
    predictions = model.predict(X_test)

    # Format predicted output
    predictions_df = pd.DataFrame(predictions)
    predictions_df.columns = category_names
    predictions_df.head();
    
    # Accuracy
    # Create an empty list to store accuracy values
    accuracies = []

    # Iterate over each output separately
    for i in range(y_test.shape[1]): 
        accuracy_i = accuracy_score(y_test.iloc[:, i], predictions_df.iloc[:, i])
        accuracies.append(accuracy_i)

    # Calculate mean accuracy across all outputs
    mean_accuracy = sum(accuracies) / len(accuracies)
    # print("Mean Accuracy:", mean_accuracy)

    # Recall
    # Create an empty list to store recall values
    recalls = []

    # Iterate over each output separately
    for i in range(y_test.shape[1]):
        recalls_i = recall_score(y_test.iloc[:, i], predictions_df.iloc[:, i], average='weighted', zero_division=0)
        recalls.append(recalls_i)
        
    # Calculate mean recalls across all outputs
    mean_recall = sum(recalls) / len(recalls)
    # print("Mean Recall:", mean_recall)

    # F1 Score
    # Create an empty list to store f1_score values
    f1_scores = []

    # Iterate over each output separately
    for i in range(y_test.shape[1]):
        f1_score_i = f1_score(y_test.iloc[:, i], predictions_df.iloc[:, i], average='weighted', zero_division=0)
        f1_scores.append(f1_score_i)
        
    # Calculate mean recalls across all outputs
    mean_f1_score = sum(f1_scores) / len(f1_scores)
    # print("Mean F1 Score:", mean_f1_score)

    # Precision
    # Create an empty list to store accuracy values
    precisions = []

    # Iterate over each output separately
    for i in range(y_test.shape[1]):
        class_report_i = precision_score(y_test.iloc[:, i], predictions_df.iloc[:, i], average='weighted', zero_division=0)
        precisions.append(accuracy_i)

    # Calculate mean accuracy across all outputs
    mean_precision = sum(precisions) / len(precisions)
    # print("Mean Precision:", mean_precision)
    
    # create a data frame of measures
    model_eval_output = pd.DataFrame({'Category': category_names, 'F1_Score': f1_scores, 'Precision': precisions, 'Recall': recalls, 'Accuracy': accuracies})
      
    # return results
    #return "The Model Perfomes as Follows: Mean Accuracy "+ str(round(mean_accuracy*100, 2))+"%; Mean Recall "+str(round(mean_recall*100, 2))+"%; Mean F1 Score "+str(round(mean_f1_score*100, 2))+"% & Mean Precision "+str(round(mean_precision*100, 2))+"%"
    return print(model_eval_output)

# Save (best) model
def save_model(model, model_filepath):

    '''
    save the trained model to the location specified in model_filepath
    '''
    # Save the trained model to a file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

# Execute steps above
def main():
    
    '''
    import data specified in database_filepath
    split data into trainining (X_train, y_train) and testing (X_test, y_test) data
    train a machine learning model on X_train and y_train which are the explanatory and reponse variables
    fit the model defined in build_model()
    evaluate trained model by passing the X_test and y_test to the function evaluate_model()
    save trained model by using save_model()
    
    '''
    
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
            
        print('Preparing test and training data...')
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # formating and aligning data
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.DataFrame(y_test)
        
        # Define temporary training dataframe
        df_train = pd.concat([X_train, y_train], axis=1)
        df_train = df_train.drop_duplicates()
        df_train = df_train.dropna()
        
        # Define temporary testing dataframe
        df_test = pd.concat([X_test, y_test], axis=1)
        df_test = df_test.drop_duplicates()
        df_test = df_test.dropna()
        
        # re-defining training samples
        X_train = df_train['tokenized_text']
        y_train = df_train.drop(columns=['tokenized_text'])
        
        # re-defining testing samples
        X_test = df_test['tokenized_text']
        y_test = df_test.drop(columns=['tokenized_text'])
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
