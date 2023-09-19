# shbang
# Judah Tanninen
# September 18th, 2023
# Description: Naive Bayes implementation, calculates the probably that an animal is of a certain class, given its "features"

# Imports
import pandas as pd
import math

# Global vars
CLASSES_LEN=7
FEATURES=["hair","feathers","eggs","milk","airborne","aquatic","predator","toothed","backbone","breathes","venomous","fins","legs","tail","domestic","catsize"]
# Functions

# Returns a zero based array of the 
def calcClassProbs(df):
    # Get the total number of items in the dataframe
    totalCnt = len(df.index)
    arr = [] # Should be an array of probabilities
    # Loop over the number of classes
    for i in range(CLASSES_LEN):
        classDf = df.query("class_type == " + str(i + 1))
        num = len(classDf.index)
        arr.append(num / totalCnt) # Add the probability
    return arr


# Begin the program
# Load the csv in
data = pd.read_csv('zoo.csv')

# Getting the training data
training = data.sample(frac=0.7)

# Generate the testing sample
test = data.drop(training.index)

classProbs = calcClassProbs(training) # Array should contain a zeroed array of class probabilites


