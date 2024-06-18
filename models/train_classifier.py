# import necessary packages
import re
import sys
import nltk
import joblib
import sklearn
    

import pandas as pd
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from sklearn.metrics import f1_score, accuracy_score

from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    This function receives the path of the database, 
    reads the data and returns it in form of a DataFrame.
    Input:
    - database_filepath (str): path of database.
    Output:
    - features (DataFrame): predictors of machine learning model.
    - labels (DataFrame): labels for predictors.
    - categories (list): tags for output labels.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('ETL', engine)
    labels = df.pop("genre")
    features = df["message"]
    
    labels = pd.get_dummies(labels)
    
    return features, labels, labels.columns.tolist()



def tokenize(text):
    """
    Lemmatize input text and convert into tokens.
    Input
    - text (str): input emergency text.
    Output 
    - clean_tokens (list) list of tokens for input text.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # create a list of urls found in input text
    detected_urls = re.findall(url_regex, text)
    # iterate over list of urls and replace them
    # with placeholder in input text
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate over list of tokens and apply tokenizer
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    This function creates a pipeline preprocessing input 
    data that goes into a random forest classifier. Then 
    gets parameters that will be applied in a grid search
    to find the best prameters in order to create the best 
    model possible.
    Input:
    - None
    Ouput:
    - cv (sklearn.pipeline): complete machine learning pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))

    ])
    

    parameters = {
        #"clf__estimator__n_estimators":[50, 100, 200], 
        "clf__estimator__n_estimators":[50, 100], 

        #"clf__estimator__min_samples_split":[2, 3, 4],
        "clf__estimator__min_samples_split":[2, 3],

        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Reveives the model, test predictors, test labels and categories
    and returns f1 score of the model.
    Inputs:
    - model (sklean.model): trained model.
    - X_test (np.array): test features.
    - Y_test (np.array): test labels.
    - category_names (list): list of labels categories.
    """
    test_preds = model.predict(X_test)
    test_f1_score = f1_score(test_preds, Y_test.values, average=None)
    
    return test_f1_score


def save_model(model, model_filepath):
    """
    Saves model in a path.
    Input:
    - model (sklearn.model): trained model.
    - model_filepath (string): model path.
    Output:
    None
    """
    joblib.dump(model, model_filepath)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building pipeline...')
        pipeline = build_model()
        
        print('Training pipeline...')
        pipeline.fit(X_train, Y_train)
        
        print("Extract best model...")
        model = pipeline.best_estimator_

        
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