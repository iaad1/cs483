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
ALPHA=0.01
# Functions

def laPlaceIt(top, bottom):
    numerator = top + ALPHA
    denominator = bottom + (ALPHA * CLASSES_LEN)
    return numerator / denominator

# Returns a zero based array of the probabilites for each class 
# (to get a classes prob, subtract 1 and get the element from this array)
def calcClassProbs(df):
    # Get the total number of items in the dataframe
    totalCnt = len(df.index)
    arr = [] # Should be an array of probabilities
    # Loop over the number of classes
    for i in range(CLASSES_LEN):
        classDf = df.query("class_type == " + str(i + 1)) # Get the rows of this class type (i + 1)
        num = len(classDf.index) # Count the rows.
        arr.append(laPlaceIt(num, totalCnt)) # Add the probability
    return arr

# Should return another zero based array of the class types, this time each one will contain a dict containing the conditional probability
# IE: to get (feathers | mammal), one would do the following: arr[0]['feathers']
def calcConditionalProbs(df):
    arr = [] # Array of dicts
    # Loop over the classes again
    for i in range(CLASSES_LEN):
        featureDict = {}
        className = str(i + 1)
        classDf = df.query("class_type == " + className) # Get the rows of this class type
        classCnt = len(classDf.index) # Count the rows.
        # Loop over the features
        for ft in FEATURES:
            # Generate a query string
            qryStr = "class_type == " + className + " and "
            # Get all unique values
            vals = df[ft].unique()
            for val in vals: # Loop through each unique value, and compute the prob for each
                query = qryStr + ft + " == " + str(val)
                key = ft + str(val) # Key for the dictionary
                rows = df.query(query) # get all rows following the above
                cnt = len(rows)
                featureDict[key] = laPlaceIt(cnt, classCnt)
        arr.append(featureDict) # Add the dict to the array, pretty simple.
    return arr

def calcProbForRow(row, classProb, condProb: dict):
    cprob = 0 # Start at one, should be real small by the time we are done.
    # Loop through each feature, and multiply the ones we need to cprob
    for ft in FEATURES:
        val = row.get(ft)
        key = ft + str(val)
        if key in condProb: # Possible we didn't have training data for this value, if not, skip.
            cprob +=  math.log2(condProb[key]) # Underflow protection
    cprob += math.log2(classProb) # add (multiply) by the probability of being the class
    return math.pow(2, cprob) # Bring two to the power of cprob to get the actual prob

def printCols(df):
    cols = list(df.columns)
    # Add the three additional columns
    cols.extend(["predicted", "probability", "correct?"])
    cols = ",".join(cols)
    print(cols)

# Begin the program
# Load the csv in
data = pd.read_csv('zoo.csv')

# Getting the training data
training = data.sample(frac=0.7)

# Generate the testing sample
test = data.drop(training.index)

# Generate the probablities of getting different classes
classProbs = calcClassProbs(training) # Array should contain a zeroed array of class probabilites

# Generate all the conditional probabilites
condProbs = calcConditionalProbs(training)

# Print the columns
printCols(test)

# Now we got all the probabilites we need. Loop through the testing data to generate
for i, row in test.iterrows():
    # Loop over each class and calculate the probability for each. Store the (unnormalized) probability
    probs = []
    for j in range(CLASSES_LEN):
        classProb = classProbs[j]
        condProb = condProbs[j]
        prob = calcProbForRow(row, classProb, condProb)
        probs.append(prob)
    # Get the largest value in the list
    biggest = max(probs)
    # Get the class number that it is (0 indexed)
    cIndex = probs.index(biggest)

    # Turn the class into a string (1 indexed)
    classGuess = str(cIndex + 1)
    # Get the sum of all the probs (for normalizing)
    psum = sum(probs)
    normalized = biggest / psum
    # Get the rows data
    rowArr = row.to_list()

    # Get the ACTUAL class type
    actualClass = str(row.get('class_type'))
    correct = "CORRECT" if actualClass == classGuess else "wrong"
    # Add the three values to the row
    rowArr.extend([classGuess, str(normalized), correct])
    # Turn to csv string
    rowStr = ",".join(map(str, rowArr))
    print(rowStr)