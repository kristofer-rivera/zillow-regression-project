from env import host, user, password, get_db_url
import pandas as pd 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Acquire raw data
def get_zillow(use_cache=True):
    '''
    This function takes in no arguments, uses the imported get_db_url function to establish a connection 
    with the mysql database, and uses a SQL query to retrieve telco data creating a dataframe,
    The function caches that dataframe locally as a csv file called zillow_raw.csv, it uses an if statement to use the cached csv
    instead of a fresh SQL query on future function calls. The function returns a dataframe with the telco data.
    '''
    filename = 'zillow_project.csv'

    if os.path.isfile(filename) and use_cache:
        print('Using cached csv...')
        return pd.read_csv(filename)
    else:
        print('Retrieving data from mySQL server...')
        df = pd.read_sql('''
        SELECT transactiondate, taxvaluedollarcnt, taxamount, roomcnt, bathroomcnt, bedroomcnt, garagecarcnt, numberofstories, lotsizesquarefeet, garagetotalsqft, calculatedfinishedsquarefeet, yearbuilt, fips, regionidcounty, regionidzip, propertycountylandusecode 
        FROM properties_2017 
        LEFT JOIN predictions_2017 USING (parcelid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        WHERE propertylandusedesc IN ('Single Family Residential', 'Inferred Single Family Residential') 
        AND YEAR(transactiondate) = 2017;''' , get_db_url('zillow'))
        print('Caching data as csv file for future use...')
        df.to_csv(filename, index=False)
        return df

#remove outliers

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

# Scale data after splitting with MinMax
def scale_data_mvp(train, validate, test, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.
    '''
    columns_to_scale = ['bedrooms', 'bathrooms', 'square_feet']
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

def visualize_scaler(scaler, df, target_columns, bins=10):
    fig, axs = plt.subplots(len(target_columns), 2, figsize=(16, 9))
    df_scaled = df.copy()
    df_scaled[target_columns] = scaler.fit_transform(df[target_columns])
    for (ax1, ax2), col in zip(axs, target_columns):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    return fig, axs




# Univariate exploration

def plot_distributions(df):
    for col in df.columns:
        sns.histplot(x = col, data=df)
        plt.title(col)
        plt.show()

def get_hist(df, cols):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = cols

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()



def get_box(df, cols):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = cols

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    #Select only the necesary columns 
    df = df[['bedroomcnt', 'bathroomcnt', 'yearbuilt', 'calculatedfinishedsquarefeet', 'fips', 'taxvaluedollarcnt']]
   
    # rename columns for clarity and readability
    df = df.rename(columns={'bedroomcnt': 'bedrooms', 'bathroomcnt':'bathrooms', 'yearbuilt':'year_built', 'calculatedfinishedsquarefeet':'square_feet',
    'taxvaluedollarcnt':'assessed_value', 'fips':'county'})
    
    # Remove outliers from our data
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'square_feet', 'assessed_value'])
    
    # get distributions of numeric data
    ## First define the columns I will look at
    cols = ['bedrooms', 'bathrooms', 'square_feet', 'assessed_value']
    get_hist(df, cols)
    get_box(df, cols)

    # Converting floats to ints where appropriate
    df.bedrooms = df.bedrooms.astype(int)
    df.square_feet = df.square_feet.astype(int)
    df.assessed_value = df.assessed_value.astype(int)
    
    # Converting fips into a string value as it is categorical data
    ## Must first be converted to int to remove the decimal and zero
    df.county = df.county.astype(int)
    df.county = df.county.astype(str)
    df.year_built = df.year_built.astype(str)

    # convert the fips values to corresponding counties
    df['county'] = df.county.map({'6037': 'Los Angeles', '6059':'Orange', '6111':'Ventura'})

     # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    #Imputing for null values in year_built
    imputer = SimpleImputer(strategy='median')  # build imputer
    imputer.fit(train[['year_built']]) # fit to train
    
    # transform the data
    
    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])

    
    return train, validate, test 


def wrangle_zillow_mvp():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(get_zillow())
    
    return train, validate, test