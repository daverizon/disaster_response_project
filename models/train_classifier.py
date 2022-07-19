import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    INPUT: 
    - database_filepath: The file path of the database that will be connected to
    
    OUTPUT:
    X: Pandas dataframe of the features for the model
    Y: Pandas dataframe of the labels for the model
    category_names: List of column names for the model labels
    
    FUNCTIONALITY: This method creates an SQLAlchemy engine and uses it to connect to the database specified in "database_filepath".  It will read all the data into a pandas dataframe, separate out the features and save it as dataframe "X", separate out the labels and save it as dataframe Y.
    NOTE: Additional cleaning is needed for the model, "child_alone" column is removed since it's never populated with anything other than "0", and any occurrence of a value 2 in the categorical labels are reclassified as 1 since we're working with classification and possible values are only 0 or 1.  Label columns are also summarized in a list and return with this method.
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # 'message_categories' is the table where I saved the ETL work in the previous workspace
    df = pd.read_sql('SELECT * FROM message_categories', engine)
    
    # Using message we want to predict which category the data should fall into
    # 'child_alone' column is removed since it's always 0 for all entries
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    
    # List of the categories names
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    # Remove punctuation and normalize the text
    new_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize the text based on word
    tokens = word_tokenize(new_text)
    
    # Remove stop words
    tokens_cleaned = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatize first based on nouns
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in tokens_cleaned]
    # Lemmatize results based on verbs
    final_tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return final_tokens


def build_model():
    '''
    INPUT: 
    - Nothing

    OUTPUT: 
    - A MultiOutput Classifier model that is hyperparameter tuned
    
    FUNCTIONALTY: This method creates a Pipeline for the model which includes
    # - CountVectorizer Transformer
    # - TfidfTransformer Transformer
    # - MultiOutputClassifier
    It also defined hyperparameters which it will tune using GridSearchSV to give an optimal model which will be returned from this method
    '''

    # CountVectorizer Transformer (Bag of Words): Count and vectorize the tokenized words according to "tokenize" method used
    # TfidfTransformer Transformer (TF-IDF, Term Frequency-Inverse Document Frequency): Transforms the count matrix to a normalized representation
    # MultiOutputClassifier: Multi-target classification
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])
    
    parameters = {
        'vect__max_df': (0.75, 1.0),
        'clf__estimator__penalty': ('l1', 'l2')
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    - model: A hyperparameter-tuned model using Pipeline and GridSearchSV
    - X_test: A pandas dataframe of test features to feed into the model to predict on
    - Y_test: A pandas dataframe of the test labels to evaluate against the predicted results based on X_test
    - category_names: A List of category names
    OUTPUT:
    - Return True after completion
    
    FUNCTIONALITY: This method used the hyperparameter-tuned model to predict the categories of X_test data, then prints out the precision, recall, f1-score, and support for the predicted data for each of the categories
    '''
    
    Y_pred = model.predict(X_test)
    print('Printing the classification report for each category')
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    
    return True


def save_model(model, model_filepath):
    '''
    INPUT:
    - model: The hyperparameter tuned model using Pipeline and GridSearchSV
    - model_filepath: Path where the model pickle file will be saved to
    
    OUTPUT:
    - Returns True after method is complete
    
    FUNCTIONALITY: This method takes the hyperparameter-tuned model and creates a pickly file that is saved in the "model_filepath" specified
    '''
    
    # Save model as pickle 
    pickle.dump(model, open(model_filepath, 'wb'))
    
    return True


def main():
    # Main method where all other methods are called from.  Arguements are passed in at runtime which are
    # - database_filepath
    # - model_filepath
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
