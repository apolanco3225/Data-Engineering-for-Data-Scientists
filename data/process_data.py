# import packages
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function receives the path for messages
    and categories dataset, then process it in order
    to merge them.
    inputs:
    - messages_filepath (str): path to messages dataset
    - categories_filepath (str): path to categories
    Outputs:
    - data (DataFrame): merged dataset
    """
    # load data in DataFrames
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # select names in categories column
    columns_names = [col[:-2] for col in categories.categories[0].split(";")]
    # create a dataframe with categories
    categories_df = categories["categories"].str.split(";", expand=True)
    categories_df.columns = columns_names
    categories_df = categories_df.applymap(lambda x: x[-1:])
    categories_df["id"] = categories["id"]

    
    # merge of two categories and messages dataframes
    data = categories_df.merge(messages, left_on="id", right_on="id")
    return data
    





def clean_data(df):
    """
    This function cleans data by dropping duplicates
    from the merge dataset.
    Input:
    - df (DataFrame): input dataframe.
    Output:
    - clean_df (DataFrame): clean dataframe without duplicates.
    """
    clean_df = df.query("related != '2'")

    clean_df.drop_duplicates(inplace=True)

    return clean_df


def save_data(df, database_filename):
    """
    This function receives a DataFrame and a file path
    and then saves data into a sql dataset in the given
    path.
    Inputs:
    - df (DataFrame) clean dataframe.
    - database_filename (str) path for storing database.
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('ETL', engine, index=False, if_exists='replace')  



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