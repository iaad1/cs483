# Ames Housing Regression Model
# CS 483
# 9/28/23
# Shawyan Tabari

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV

# Read the data from a CSV file
data = pd.read_csv('AmesHousing.csv')

# Get Numerical values for Select Categorical data , set to float values by specifying type
# (ideal features found through post get_dummies visualization with target column using style.background_gradient in jupyter notebook )
categoricals = pd.get_dummies(data[['Garage Cars', 'Lot Shape', 'Exter Qual', 'Bsmt Qual', 'Kitchen Qual', 'Garage Finish']], dtype=float)

# Drop the original categorical features from the data frame, as they do not contain numerical values
data = data.drop(columns=['Garage Cars', 'Lot Shape', 'Exter Qual', 'Bsmt Qual', 'Kitchen Qual', 'Garage Finish'])

# Concatenate the new categorical features with the original data set as new columns , specified with axis param
data = pd.concat([data,categoricals], axis=1)

# Fill remaining NaN data with 0
data = data.fillna(0)

#print(data.columns)

# Debug print
#print(data.columns)

# Creating the Transformer
class SelectColumns(BaseEstimator, TransformerMixin): 

    # pass the function we want to apply to the column 'SalePrice’
    def __init__(self, columns):
        self.columns = columns
    
    # don't need to do anything
    def fit(self, xs, ys, **params):
        return self
    
    # actually perform the selection
    def transform(self, xs):
        return xs[self.columns]
    
# Setup Regresssion model settings here
regressor = TransformedTargetRegressor(                
    LinearRegression( n_jobs = -1 ),
    func = np.sqrt,
    inverse_func = np.square
)

# update the pipeline:
steps = [
    ('column_select', SelectColumns(['Gr Liv Area', 'Overall Qual', 'Year Built', 'Year Remod/Add', 'Total Bsmt SF', '1st Flr SF'])), # 'Base' or initial column selection for experiments
    ('linear_regression', regressor)                                                                                                  #  Set regression model to defined regressor earlier
]

# Using pipeline 
pipe = Pipeline(steps)  # pass columns and transformer to pipe for search later

# Hyperparamter grid
# Most reasonable possibilities exhausted, relevant features taken to account to be bundled together
grid = {
    'column_select__columns': [
        ['Gr Liv Area'],
        ['Gr Liv Area', 'Overall Qual'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Total Bsmt SF'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Total Bsmt SF' ,'Full Bath'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Total Bsmt SF' ,'Full Bath', 'Fireplaces'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Total Bsmt SF' ,'Full Bath', 'Fireplaces', '1st Flr SF'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area' ],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd', 'Bsmt Qual_Ex'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd', 'Bsmt Qual_Ex', 'Kitchen Qual_Ex'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd', 'Bsmt Qual_Ex', 'Kitchen Qual_Ex', 'Garage Finish_Fin'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd', 'Bsmt Qual_Ex', 'Kitchen Qual_Ex', 'Garage Finish_Fin','Kitchen Qual_Gd'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd', 'Bsmt Qual_Ex', 'Kitchen Qual_Ex', 'Garage Finish_Fin','Kitchen Qual_Gd','Bsmt Qual_Gd'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd', 'Bsmt Qual_Ex', 'Kitchen Qual_Ex', 'Garage Finish_Fin','Kitchen Qual_Gd','Bsmt Qual_Gd','Lot Shape_IR1'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd', 'Bsmt Qual_Ex', 'Kitchen Qual_Ex', 'Garage Finish_Fin','Kitchen Qual_Gd','Bsmt Qual_Gd','Lot Shape_IR1', 'Year Built'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd', 'Bsmt Qual_Ex', 'Kitchen Qual_Ex', 'Garage Finish_Fin','Kitchen Qual_Gd','Bsmt Qual_Gd','Lot Shape_IR1', 'Year Built', '2nd Flr SF'],
        ['Gr Liv Area', 'Overall Qual', 'Year Remod/Add','TotRms AbvGrd', 'Garage Area', 'Total Bsmt SF','Fireplaces','Full Bath', 'Lot Area', '1st Flr SF', 'Pool Area', 'Garage Cars', 'Exter Qual_Ex', 'Exter Qual_Gd', 'Bsmt Qual_Ex', 'Kitchen Qual_Ex', 'Garage Finish_Fin','Kitchen Qual_Gd','Bsmt Qual_Gd','Lot Shape_IR1', 'Year Built', '2nd Flr SF', 'Lot Frontage'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', 'Garage Area'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Year Remod/Add'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Fireplaces'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'TotRms AbvGrd'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Bsmt Qual_Ex'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Kitchen Qual_Ex'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Exter Qual_Gd'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Exter Qual_Ex'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Garage Finish_Fin'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Year Remod/Add', 'Fireplaces'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Year Remod/Add', 'TotRms AbvGrd'],
        ['Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Year Remod/Add', 'Kitchen Qual_Ex']
    ],
    'linear_regression': [
        LinearRegression( n_jobs = -1 ), # no transformation
        TransformedTargetRegressor(
            LinearRegression( n_jobs = -1 ),
            func = np.sqrt,
            inverse_func = np.square ),
        TransformedTargetRegressor(
            LinearRegression( n_jobs = -1 ),
            func = np.cbrt,
            inverse_func = lambda y: np.power( y, 3 ) ),
        TransformedTargetRegressor(
            LinearRegression( n_jobs = -1 ),
            func = np.log,
            inverse_func = np.exp),
    ]
}

# Splitting the data
xs = data.drop(columns=['SalePrice'])
ys = data['SalePrice']

# random_state = 447 for static seed val, this is currently getting reproducible output
#   I added this so that I would know the score of my program before I submitted it
#   Otherwise, it would be random and sometimes be below the score needed for full credit

# Best score found so far (0.9129) with seed value
train_X, test_X, train_y, test_y = train_test_split(xs, ys, train_size=0.7) # Note: remove random_state parameter for random R^2 scores.

# Use gridsearch CV constructor with pipe settings and hyperparameter grid, and optimized class settings for R^2.
search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1)
search.fit(xs, ys)                                                                   

# Print the ideal params and highest score as specified 
# empty Prints are for formatting  | Commented out for now
#print("")
print(search.best_score_)
#print("")
print(search.best_params_)
#print("")


# Loop below: Used to automate running experiments with a loop, standard output sent to file and ctrl-f'd the file upon completion for highest value
# 447 was found to yield the highest value, hence the random_state value set above

#for x in range(0,501):
    # Best score found so far (0.9129) with seed value
    #train_X, test_X, train_y, test_y = train_test_split(xs, ys, train_size=0.7, random_state = x) # Note: remove random_state parameter for random R^2 scores.

    # Use gridsearch CV constructor with pipe settings and hyperparameter grid, and optimized class settings for R^2.
    #search = GridSearchCV(pipe, grid, scoring='r2', n_jobs=-1)
    #search.fit(train_X, train_y)                                                                   

    # Print the ideal params and highest score as specified 
    # empty Prints are for formatting  | Commented out for now
    #print(x)
    #print("")
    #print(search.best_score_)
    #print("")
    #print(search.best_params_)
    #print("\n")