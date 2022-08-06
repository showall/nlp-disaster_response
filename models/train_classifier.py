import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle

def load_data(database_filepath):
    """
        load sqlite database from database_filepath
        and make dataframe from `Message` table
        
        input:
        database_filepath: directory of database
        
        returns:
        X: message text list
        Y: corresponding category list
        columns: categorical labels of Y
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("InsertTableName", con = engine)
    X = df["message"]
    Y = df.iloc[:,4:].astype(int)
    for column in Y.columns:
        Y[column] =Y[column].apply(lambda x : 1 if x > 1 else x)
        if Y[column].sum(axis=0)==0:
            Y =Y.drop(column, axis=1) 
    category_names = Y.columns
    Y = np.array(Y)
    X = np.array(X)
    X = X.reshape(len(X),1)
    X = X.flatten()
    return X, Y, category_names
    

def tokenize(text):
    """
        tokenize the text and returns list of token
        - tokenize
        - lemmatize
        - normalize
        - stop words filtering
        - punctuation filtering
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
        Build a count_vectorizer,multi-output logistic regression model as a pipeline
        returns:
        model : a pipeline model
    """
    model = Pipeline(steps= [("tokenizer",CountVectorizer(tokenizer=tokenize, ngram_range=(1, 1))),
    ("estimator",MultiOutputClassifier(LogisticRegression(C=0.5, penalty="l1")))])
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
        evaludate the model by X_test, Y_Test dataset
        print the classification report(f1-score, recall, precision) of each column(category)
    """
    score = []
    cr = []
    for i in range (len(category_names)):
        mask = np.array([Y_test[:,i] == 1])
        Y_test_sub = pd.DataFrame(Y_test)
        mask =mask.flatten()
        index = Y_test_sub.iloc[mask,i].index
        y_test_sub = np.array(Y_test_sub)[index]
        X_test_sub = X_test[index]
        y_pred_sub = model.predict(X_test_sub)
        score.append(accuracy_score(y_test_sub[:,i],y_pred_sub[:,i]))
        cr.append(classification_report(y_test_sub,y_pred_sub))    
    return cr


def save_model(model, model_filepath):
    """
        Save the model as a pickle file to model_filepath
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        print('Building model...')
        model = build_model()
        print(Y_train.sum())
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