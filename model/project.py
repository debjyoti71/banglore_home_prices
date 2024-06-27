import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load the dataset
df = pd.read_csv("C:/Users/HP/Desktop/csv_files/bengaluru_house_prices.csv")

# Drop specified columns
df1 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

# Drop rows where any of the remaining columns are NaN
df2 = df1.dropna()

# Create a new column 'bhk' by extracting the number of bedrooms from the 'size' column
df2.loc[:, 'bhk'] = df2['size'].apply(lambda x: int(x.split(' ')[0]))

# Define a function to check if a value can be converted to a float
def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

# Filter out rows where 'total_sqft' cannot be converted to a float
df2 = df2[df2['total_sqft'].apply(is_float)]

# Define a function to convert 'total_sqft' values to a numerical format
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

# Apply the conversion function to the 'total_sqft' column
df2.loc[:, 'total_sqft'] = df2['total_sqft'].apply(convert_sqft_to_num)

# Calculate price per square foot and create a new column 'price_per_sqft'
df2.loc[:, 'price_per_sqft'] = df2['price'] * 100000 / df2['total_sqft']

# Remove leading and trailing spaces from the 'location' column
df2.loc[:, 'location'] = df2['location'].apply(lambda x: x.strip())

# Get the count of unique locations and consolidate locations with less than or equal to 10 listings
location_stats = df2['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats <= 10]
df2.loc[:, 'location'] = df2['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Remove entries where total_sqft per bhk is less than 300
df2 = df2[~(df2.total_sqft / df2.bhk < 300)]

# Function to remove outliers based on price per square foot
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df2 = remove_pps_outliers(df2)

# Function to remove bhk outliers
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df2 = remove_bhk_outliers(df2)

# Remove entries where the number of bathrooms is greater than the number of bedrooms + 2
df2 = df2[df2.bath < df2.bhk + 2]

# Drop 'size' and 'price_per_sqft' columns
df2 = df2.drop(['size', 'price_per_sqft'], axis='columns')

# Create dummy variables for 'location' and drop the original 'location' column
dummies = pd.get_dummies(df2.location)
df2 = pd.concat([df2, dummies.drop('other', axis='columns')], axis='columns')
df2 = df2.drop('location', axis='columns')

# Define the features (X) and target variable (y)
X = df2.drop(['price'], axis='columns')
y = df2['price']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train the Linear Regression model
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Evaluate the model
print(lr_clf.score(X_test, y_test))

# Define a function to predict house prices
def predict_price(location, sqft, bath, bhk):    
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

# Test the predict function
print(predict_price('1st Phase JP Nagar', 1000, 2, 2))

# Save the model and columns
import pickle
with open('C:/Users/HP/Desktop/python/banglore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)

import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("C:/Users/HP/Desktop/python/columns.json", "w") as f:
    f.write(json.dumps(columns))