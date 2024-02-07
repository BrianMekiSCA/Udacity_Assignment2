
# Project motivation
This project tries to build pipelines for data extraction, transformation and loading, and fitting this data to appropriate machine learning solutions.
Ulitmately, a user interactive tool is developed that makes use of the trained models to help answer user-given questions
Moreover, data visualizations are also provided to help further richly inform user  

# Project Description
This is a multi-faceted project that comprises the following:
- Creation of a data Extration, Transformation and Loading (ETL) ETL pipeline
- Creation of a Machine Learning (ML) pipeline
- Creation and deployment of a user interactive (UI) application based on the above

## ETL Pipeline
This pipelines achives the following:
- import messages and their respective categories in CSV form, i.e., messages.csv & categories.csv
- clean data by removing duplicated and / missing entries,
- formart data by coverting the predicted variable categories to hot-one-encoding,
- merging both the explanatory input message and response variable categories into a single source and finally,
- export and save the output data to a SQL lite database for subsequent use, this if the file with the extension *.db*

The following libraries are eployed in this task:
* import pysftp 
* import pandas as pd 
* from datetime import datetime as dt 
* import os
* from sqlalchemy import create_engine 

The final script pipeline is executed in the command prompt as follows:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

## ML Pipeline
Here,
- the data results of ETL pipeline are ingested.
- the messages variable is are tokenized as it will serve as the explanatory input to the selected ML algorithm,
- the algorithm of choise is the best multi-class multi-label model based on the f1 score, accuracy and precision measures
- the trained model is saved to a pickle file to be used in essentially "out-of-sample" predictions.

The libraries employed here are:
* import sys
* import pandas as pd 
* import pickle
* from sklearn.model_selection import train_test_split, GridSearchCV
* from sklearn.feature_extraction.text import TfidfVectorizer
* from sklearn.ensemble import RandomForestClassifier
* from sklearn.multioutput import MultiOutputClassifier
* from sklearn.metrics import classification_report
* from sklearn.pipeline import Pipeline
* import string
* from nltk.tokenize import word_tokenize
* from nltk.corpus import stopwords
* from nltk.stem import PorterStemmer
* from nltk.stem import WordNetLemmatizer
* from nltk.stem import wordnet
* from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

The ML pipeline is executed in the command prompt as follows:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

## UI Application
- The application aims to:
- 1. make use of the afore-mentioned algorithm to predict likely classification of messages provided by the user,
- 2. train a new classifier based on new data provided by the user,
- 3. provide simple graphical visualizations of the underlying data

The libraries employed here are:
* import json
* import plotly
* import pandas as pd
* from nltk.tokenize import word_tokenize
* from flask import Flask, render_template, request
* from plotly.graph_objs import Bar, Histogram
* import joblib
* from sqlalchemy import create_engine
* import matplotlib
* import matplotlib.pyplot as plt
* from nltk.stem import WordNetLemmatizer
* import nltk
* nltk.download('punkt')

