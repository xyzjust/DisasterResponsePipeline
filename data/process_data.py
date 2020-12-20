import sys
import nltk
import re
import numpy as np
import pandas as pd

from datetime import datetime
    
import sqlite3



def load_data(messages_filepath, categories_filepath):
    
    df_messages = pd.read_csv(messages_filepath).drop_duplicates()
    df_categories = pd.read_csv(categories_filepath).drop_duplicates()
    
    
    all_categories = set([d.split('-')[0] for d in np.array([d for d in df_categories['categories'].str.split(';')]).flatten()])

    df_cateogries_expand = df_categories['categories'].str.split(";", expand = True)

    col_dict = {}
    for col_name, col_value in df_cateogries_expand.iteritems():
        col_dict.update({col_name : list(set([d.split('-')[0] for d in col_value]))[0] }) 

    for i in df_cateogries_expand.columns:
        df_cateogries_expand[i] = pd.to_numeric(df_cateogries_expand[i].str.split('-').apply(lambda x: x[1]))

    df_categories = df_categories.merge( df_cateogries_expand.rename( columns = col_dict),
                                         left_index=True,
                                         right_index=True,
                                         how='outer')

    df_categories= df_categories.drop(columns = 'categories').set_index('id')
    labels = [d for d in df_categories.columns]
    df_whole = df_messages.merge(df_categories, left_on = 'id', right_index=True, how='outer')

    return df_whole


def clean_data(df):
    
    df_return = df[['id']].copy(deep=True)
    all_columns = [d for d in df.columns if d != 'id']
    
    for col in all_columns:
        df_temp = df.groupby(['id'])[[col]].max()
        
        df_return = df_return.merge(df_temp, 
                                    left_on='id',
                                    right_index=True,
                                    how='left')
    
    df_return = df_return[df_return['related'] < 2].drop_duplicates()
    df_return['ml_input'] =  df_return['genre'] +':' + df_return['message']
    
    return df_return


def save_data(df, database_filename):
    
    conn = sqlite3.connect(database_filename)

    df.to_sql("DisasterResponse", conn, if_exists="replace")

    conn.commit()
    conn.close()

    pass  


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