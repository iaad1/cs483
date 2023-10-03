# shbang
# Judah Tanninen
# Description: Linear regression

# Imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor

import random

# Classes
class SelectColumns(BaseEstimator, TransformerMixin): 
    def __init__(self, columns):
        self.columns = columns    
    def fit(self, xs, ys, **params):
        return self    
    def transform(self, xs):
        return xs[self.columns]

# Not really used anymore, was going to be used for data analysis on what the best columns were
def genRandChoices(nums, cats, n):
    num_sample = random.sample(nums, k=n)
    cat_sample = random.sample(cats, k=n)
    return num_sample, cat_sample

# Funciton
def genRelations(columns, df):
    mapping = {}
    for col in columns:
        # Find the new columns related to the old column
        related_columns = [new_col for new_col in df.columns if new_col.startswith(col)]
        # Store the mapping in the dictionary
        mapping[col] = related_columns
    return mapping

# Read the csv into a dataframe
df = pd.read_csv('AmesHousing.csv')


# Numerical columns we will use (picked randomly by me)
num_features = ['Lot Area', 'Lot Frontage', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Garage Area', 'Garage Cars', 'TotRms AbvGrd', 'Fireplaces', 'Gr Liv Area']
# Categorical features we will use. (also picked randomly)
cat_features=['Neighborhood', 'House Style', 'Bldg Type', 'Kitchen Qual', 'Garage Qual', 'Paved Drive', 'Fence', 'Functional', 'Bsmt Qual', 'Exter Qual', 'Garage Type']

# Get the dummies for the categorical stuff
catDF = pd.get_dummies(df[cat_features], dtype=float)

# Create relations between old categorical features and the new dummy columns, so we can easily use the new columns
mapping = genRelations(cat_features, catDF)

# Add the new columns to the dataframe
df = pd.concat([df, catDF], axis=1)
# Fill empty (nan/na) with 0
df = df.fillna(0)


select_columns = []
# Made this for finding the best columns to use for the grid search
# Ended up being useless, because I'm apparently the warren buffet of house prices
for i in range(0, 1): # Would make this more, but this is all not needed
    all = []
    nums, cats = genRandChoices(num_features, cat_features, 11) # 11 columns, if actually using this properly, it wouldn't be 11
    all += nums
    catsall = []
    for val in cats:
        newcats = mapping[val]
        all += newcats
    select_columns.append(all)

# Split the data into features and the target variable
xs = df.drop('SalePrice', axis=1)
ys = df['SalePrice']

# Create the regressor, base is a square root.
regressor = TransformedTargetRegressor(                
    LinearRegression( n_jobs = -1 ),
    func = np.sqrt,
    inverse_func = np.square
)

# Steps for the pipeline.
steps = [
    ( 'column_select', SelectColumns([])),
    ( 'linear_regression', regressor)
]

# Make the pipeline
pipe = Pipeline(steps)

# Create the grid, using the generated select_columns value, and some regressors
grid = {
    'column_select__columns': select_columns,
    'linear_regression': [
        LinearRegression( n_jobs = -1 ), # No transformation 
        TransformedTargetRegressor( # Square root it
            LinearRegression(n_jobs=-1),
            func = np.sqrt,
            inverse_func=np.square),
        TransformedTargetRegressor( # Cube root
            LinearRegression(n_jobs=-1),
            func = np.cbrt,
            inverse_func=lambda y:np.power(y, 3)),
        TransformedTargetRegressor( # Logorithm
            LinearRegression(n_jobs=-1),
            func = np.log,
            inverse_func = np.exp)
    ]
}

# Copying the slides on the above, still not very sure why it is needed, once the best fit is found, we don't really need this?
# Leaving it in, but its essentially

# Make the search
search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1)
# Make the fit
search.fit(xs, ys)

print(search.best_score_)
print(search.best_params_)

# Grid is kind of useless after you've figured out the best columns, best pipeline is the columns I have selected, with the square root regression
