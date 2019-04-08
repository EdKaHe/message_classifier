import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages, categories

def clean_data(messages, categories):
    # merge both datasets
    df = pd.merge(messages, categories, on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # use the first row to extract all category columns
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0]).unique().tolist()
    
    # rename the columns
    categories.columns = category_colnames
    
    # extract the 1 and 0 from each entry
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the categories column from the dataframe
    df.drop(labels=['categories'], axis=1, inplace=True)
    
    # concatenate the dataframes
    df = pd.concat((df, categories), axis=1)
    
    # remove duplicates
    df = df[~df.duplicated()]
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename.replace('/', '//'))
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:
        print(sys.argv)
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
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