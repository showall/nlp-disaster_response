import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        load message and category files and returns as a merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    categories = df.categories.str.split(pat=";", expand=True)  
    row = categories.iloc[0]
    category_colnames = row.str.split(pat="-",expand=True).iloc[:,0]
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split(pat="-",expand=True)[1]
    df.drop(["categories"], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    return df
    
    
def clean_data(df):
    """
        clean the data frame.
        - rename the category columns.
        - convert category value to numeric
        - drop duplicates
        returns cleaned dataframe.
    """    
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
        Save dataframe to `Message` table in sqlite file at database_filename
        if `Message` table is exists, replace it.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("YourTableName", engine, index=False)  


def main():
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