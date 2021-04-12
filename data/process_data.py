#%%
# Import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)

#%%
def load_data(messages_filepath, categories_filepath):
    '''
    The function loads the messages and the categories datasets
    into separate pandas dataframes and merges these by the message ID (as 'id').

    IN:
        messages_filepath - path and name of the file that contains messages
        categories_filepath - path and name of the file that contains categories for each message
    OUT:
        df - merged dataframe of messages and categories by message
    '''

    # Load the messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load the categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.merge(messages, categories, on='id', how='inner')
    return df


def clean_data(df):
    '''
    The function cleans the dataframe of messages and categories by
    (1) creating separate dummy variables from the categories feature
    (2) removing duplicate records.

    IN:
        df - pandas dataframe containig records of messages and categories
    OUT:
        df_clean - cleaned dataframe
    '''
    
    # (1.1) Split categories into separate category columns
    ### Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=";", expand=True)

    ### Rename the new columns based on the values of the first row
    row = categories.iloc[0]
    category_colnames = list(i[:-2] for i in row)
    categories.columns = category_colnames

    # (1.2) Convert category values to just numbers 0 or 1
    for column in categories:
    
        ### Set each value to be the last character of the string
        categories[column] = list(i[-1] for i in categories[column])
    
        ### Convert column from string to numeric
        categories[column] = categories[column].astype('int32')

    # (1.3) Replace categories column in df with new category columns
    ### Drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    #### Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # (2) Remove duplicate records
    ### Create a value for each row noting whether it has multiple occurences (except for the first occurence)
    ### True if the row is a duplication of another row (except for the first occurence), False if the row is unique
    df['duplicate_boolean'] = df.duplicated(keep='first')

    ### Drop duplicates
    df = df.drop(df[df.duplicate_boolean == True].index)
    
    ### Drop the column created to find duplicates
    df_clean = df.drop(['duplicate_boolean'], axis=1)

    return df_clean


def save_data(df, database_filename):
    '''
    The function saves the dataframe as a SQLite dataset.

    IN:
       df - dataframe to be saved
       database_filename - name of the database where the function saves the dataset 
    OUT:
        dataset saved in a SQLite database
    ''' 
    
    # Create engine for the SQLite database
    engine = create_engine('sqlite:///'+ database_filename)
    
    # Save dataset in the SQLite database
    df.to_sql('DisasterTable', engine, index=False)

    # Print output message
    print('{} dataframe has been saved in {}'.format(df, database_filename))


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
# %%
