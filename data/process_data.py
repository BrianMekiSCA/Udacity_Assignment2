# import libraries
import sys
import pysftp # library provides sftp connectivity tools
import pandas as pd # library provides mathematical suit 
from datetime import datetime as dt #library allows for manipulation of dates and time
import os # library allows for detection and manipulation of file paths / directories
from sqlalchemy import create_engine # library allows for creation of sql engine

# import data from user given pathways
def load_data(messages_filepath, categories_filepath):
    '''
    import messages and categories input data from given pathways messages_filepath and categories_filepath respectively 
    merge messages and categories data into a single source data frame called df
    
    '''
    # Import raw messages and categories
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge raw data
    df = pd.merge(messages, categories, on='id', how='inner')
    df = pd.DataFrame(df)
    
    # return results
    return df

# clean and custom format data
def clean_data(df):
    '''
    import dataframe called df comprising of messages and categories columns 
    isolate categories column
    split values of categories column by ";" factor into 36 seperate categies
    retain the split categories as respective column values 
    from the first row, select the first string from each of the 36 values
    use the first value of each split string as the column title
    select the last entry of each split string as the value for that column such that each column is binary with values 0/1 only
    convert the new categories dataset into a data frame
    replace the categories column in the df dataframe with the new 36-column binary data frame and return the new df object
    remove all rows with values greater 1 to make sure all columns are strictly binary

    '''
    # clean data by removing duplicated rows
    df = pd.DataFrame(df)

    # Clean data by replacing NaNs with 0
    df = df.dropna()
    
    # Split string messages into individual words 
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [value[:-2] for value in row]
    categories.columns = category_colnames
    
    # Convert each column into a 0-1 variable
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = [value[-1] for value in categories[column]]

        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Specify data frame type
    categories = pd.DataFrame(categories)
    
    # Drop original categories colunm from prevailing data frame & replace it with the split-columns categories data frame
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    df = pd.DataFrame(df)
    
    # Remove columns with NaN values
    df = df.dropna(axis=0)
    
    # remove rows with values greater or equal to 2
    df = df[~(df[category_colnames] > 1).any(axis=1)]

    # return results
    return df

# save output to given filepath
def save_data(df, database_filename):
    
    '''
    import the clean dataframe df
    export the df object to a database specified in database_filename
     
    '''
    # Create an SQLAlchemy engine to connect to the database
    engine = create_engine('sqlite:///'+database_filename)

    # Write the DataFrame to a SQL database table named 'DisasterResponse'
    df = pd.DataFrame(df)
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists='replace')

def main():
    '''
    import raw data via the function load_data() and output a single merged data frame 
    use function clean_data() to:
        clean merged dataframe by removing rows with missing data and values greater than 1, 
        customize by raw splitting category strings into individual columns
    export clean dataframe to a database using save_data()  
    
    '''
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