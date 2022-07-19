import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - filepath where the the first datasource is located
    categories_filepath - filepath where the second datasource is located
    
    OUTPUT:
    df - a pandas dataframe with datasource 1 and datasrouce 2 are merged together on column "id"
    
    FUNCTIONALITY: This method will merge the two datasources together based on thir same 'id' column value
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    '''
    INPUT:
    df - a pandas dataframe that needs to be cleaned
    
    
    OUTPUT:
    df - a pandas dataframe that is cleaned to the specifications listed in the respective course notebook.  Note that additional cleaning is needed, but that will be done in the train_classifier.py file since it was not in the steps for cleaning of the process_data.py respective notebook
    
    FUNCTIONALITY: The method puts each of the categories into it's own dataframe column, put it's respective values in the appropriate dataframe column as a numeric value, and drops the rows with duplicate data
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = list(map(lambda idx: idx.split('-')[0] , list(row)))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #There is a category where values of 2 are in the column.  This doesn't make sense since it should be either 0's or 1's.  We will replace all 2's with 1's
    categories = categories.replace(2, 1)
     
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    # Since rows with the same id can have different data, I'll assume "duplicates" in the instructions mean duplicate rows of data where all columns are the same values in order for them to be labeled as "duplicate"
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filepath):
    '''
    INPUT: 
    df - a pandas dataframe containing all the data that needs to be stored in a DB
    database_filename - filename of the database where the new table will be created for df's data to be pushed in to
    NOTE: Initially "database_filepath" was specified instead of "database_filename" for this method, however in the main method it's labeled as "database_filepath" so I changed it to that
    
    OUTPUT:
    - Returns true after data is pushed to the "message_categories" table of the database_filepath specified
    
    FUNCTIONALITY: This method creates a SQLAlchemy engine and connects to the database specified by "database_filepath".  It will then push all the data from dataframe "df" into a table caleld "message_categories" without the dataframe indices, then return True
    '''
    
    # Create SQLAlchemy engine
    engine = create_engine('sqlite:///' + database_filepath)
    
    # Save the dataframe into SQL
    df.to_sql('message_categories', engine, index=False, if_exists='replace')
    
    return True


def main():
    # Main method where all other methods are called from.  Arguements are passed in at runtime which are
    # - messages_filepath
    # - categories_filepath
    # - database_filepath
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
