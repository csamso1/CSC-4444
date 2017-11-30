"""Clayton Samson

To predict Rush Yards use the following stats:

statsUsing = ['Rush Attempts', 'Rush TD', 'Fumbles Lost', 'Rush 2PT', 'Rush Yards Allowed', 'Rush over Total Yards Allowed']
Scores: 
- Mean^2 Error Avg    254.524079283
- Variances Avg   0.819760890617


To predict Rush TD use the following stats:

statsUsing = ['Rush Attempts', 'Rush Yards', 'Fumbles Lost', 'Rush 2PT', 'Rush Yards Allowed', 'Rush over Total Yards Allowed']
Scores:
- Mean^2 Error Avg	0.20038091655
- Variances Avg	0.235909735687"""


"""below is the methods I used, you might need to look at how I created the 'Rush over Total Yards Allowed' series as that seems to improve my predictions significantly"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import operator
import argparse
from collections import OrderedDict
from collections import Counter
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# Linear Regression method that takes the Xfeatures you want to input, Y feature that you want to predict, a filePath to a csv or a DataFrame
# xFeatures should be of type list, yFeature need be a string
# DO NOT PASS POINTS with the data, it'll give you a false sense of security. Talking about FF Points which is labeled simply as 'Points' in my csv
# Also, the data you pass should be for a prediction on a single player or defensive team. It'll take whatever but logically we need predictions for individuals
# But ughhh IDK either try to predict for all positions or try for single player then scale up, whatever floats your boat. IDK which is the better start
def multivariateLinearRegression( xFeatures, yfeature, players_data):
    #removed param: file=None,
    # works best if you pass the full file path instead of just the file name

    # Change this to your data dir for more convenient file passing
    os.chdir('/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI')
    # Gotta make sure you passed a valid fileName, or filepath if you didn't change the line above
    if players_data is None:
        try:
            players_data = pd.read_csv("RBAllYears.csv")
        except:
            print("Invalid file name or file path passed as argument")

    # Creates a set, phenomenal for membership checking, unions, intersections, etc,
    setList = set(list(players_data.columns))   #We first make it a list since a set needs to be passed an iterable data structure
    # print(setList)
    # print(yfeature)
    # print(xFeatures)
    # Test if the feature arguments is a valid column
    if not yfeature in set(setList):
        raise Exception(yfeature + "Feature doesn't exist. Did you mispell it or pass the from csv")
    for x in xFeatures:
        if not x in set(setList):
            raise Exception(str(x) + "Feature doesn't exist. Did you mispell it or pass the from csv")


    # We need to add nonzero column to our X data due to matrix and vector maths, and the need for axis intercepts that aren't equal to 0
    X = players_data[list(xFeatures)].values.reshape(-1, len(xFeatures))
    Y = players_data[yfeature]

    # At some point we are going to have to standardize all the values, but for now it's easier to understand without standardizing
    # X_train, X_test, Y_train, Y_test = train_test_split(preprocessing.scale(X), preprocessing.scale(Y), test_size=.2)
    # Here we are splitting the data into a random 80% for training, with the remaining 20% used for testing, change if you want
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

    # I was doing something but forgot, just let these two lines live on and don't pay them any attention
    coeff_titles = list(xFeatures)
    coeff_titles.insert(0, 'Intercept')

    # Our linear regression model
    reg = linear_model.LinearRegression()
    # I love the sound of a PassiveAgressiveRegressor, try to make it work for shits and gigs pleeeeeease :)
    # reg = linear_model.PassiveAggressiveRegressor()

    model = reg.fit(X_train, Y_train) #Fitting the model with our training data
    yfeature_prediction = model.predict(X_test)
    print("Predicting: ",yfeature)
    print("The Coeffecients:\n ")
    # Let's format the list of float point coeffs so they're easier to read
    pattern = "%.2f"
    floatsstrings = [pattern % i for i in list(reg.coef_)]
    print(floatsstrings)
    # 2 lines below were part of trial to print the title of each coef, uncommenting will enable that super power...maybe
    # floats = list(float(i) for i in floatsstrings)
    # hmmm = dict(zip(coeff_titles, floats))
    # print(hmmm)


    mean2Err = mean_squared_error(Y_test, yfeature_prediction)
    varianceScore = r2_score(Y_test, yfeature_prediction)
    return (mean2Err, varianceScore)

# method name says it all
def combineOffensePositionWithDefense(oPosition, save=False):
    # change working directory to the correct one. If you're unsure what your current dir is then run (os.getcwd())
    os.chdir('/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI')
    players_data = pd.read_csv(oPosition + "AllYears.csv")
    defense_data = pd.read_csv("DSTAllYears.csv")
    pd_data = pd.merge(players_data, defense_data, left_on=['Year', 'Week', 'Opponent'],
                       right_on=['Year', 'Week', 'Team'])
    #pd_data.assign(rush_yards_over_total_yards = )
    #pd_data('Rush Yards Allowed')
    # print(pd_data.head)
    if save:
        pd_data.to_csv(oPosition + "_with_Defense.csv")
    return pd_data

#Creates a new series of the DST ratio of Rushing Yards allowed / Total Yards allowed, which can later be joined with the primary dataframe
def createNewColumn(dataframe):
    column_rush_yards_allowed = dataframe['Rush Yards Allowed']
    column_total_yards_allowed = dataframe['Total Yards']
    ratio = []
    for i in range(len(column_rush_yards_allowed)):
        ratio.append(column_rush_yards_allowed[i] / column_total_yards_allowed[i])
    #new_dataframe = dataframe.assign(Rush_Yards_over_Total_Yards=df['Rush over Total Yards'], ratio)
    new_dataframe = pd.DataFrame({'Rush over Total Yards Allowed' : ratio})
    print(new_dataframe.head())
    return new_dataframe


def main():
    # These are the stats to predict. All of them have a nonzero coefficient in Fantasy Football Points algorithm for an Offensive player
    stats = ['Fumble TD', 'Fumbles Lost', 'Pass 2PT',
             'Pass Attempts', 'Pass Completions', 'Pass Interceptions', 'Pass TD',
             'Pass Yards', 'Receiving 2PT', 'Receiving TD',
             'Receiving Yards', 'Receptions', 'Rush 2PT', 'Rush Attempts', 'Rush TD',
             'Rush Yards']
    # These are the stats I'm actually concerned about accurately predicting right now
    statsUsing = ['Rush Attempts', 'Rush TD', 'Fumbles Lost', 'Rush 2PT', 'Rush Yards Allowed', 'Rush over Total Yards Allowed']
    #scanning in args
    player_name = input("Please enter the prediction file name (PlayerName): ")
    statistic_to_predict = input("What statistic are we predicting? ")
    os.chdir("/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI/Predictions")
    stats_used_to_make_prediction = 'using ' + ' '.join(str(stat) for stat in statsUsing)
    ourPredictions = open(player_name + ' - ' + statistic_to_predict + ' - ' + stats_used_to_make_prediction + '.txt', 'w')
    resultTuples = []
    players_data = combineOffensePositionWithDefense('RB')
    #creating a new colum with Rush Yards / Total Yards Column
    enhanced_player_data = createNewColumn(players_data)
    #merging new column to the orginal dataframe
    players_data = pd.merge(players_data, enhanced_player_data, left_index=True, right_index=True)
    # exporting this dataframe to CSV to make sure it is correct
    # players_data.to_csv("enhanced_player.csv")
    # trimming down to just 1 players stats
    # players_data = getSpecifiedStatisticsForSpecificPlayerGrouped(players_data, player_name)
    # players_data.to_csv("just_individual_player_data.csv")

    #Running the simulation
    resultTuples.append(multivariateLinearRegression(statsUsing, statistic_to_predict, players_data))
    mean2Errs, variances = zip(*resultTuples)
    ourPredictions.write("Predicting: \t" + statistic_to_predict)
    ourPredictions.write('\nMean^2 Error Avg\t' + str(sum(mean2Errs) / float(len(mean2Errs))))
    ourPredictions.write('\nVariances Avg\t' + str(sum(variances) / float(len(variances))))
    ourPredictions.close()
    print("Predicting: \t" + statistic_to_predict)
    print('\nMean^2 Error Avg\t' + str(sum(mean2Errs) / float(len(mean2Errs))))
    print('\nVariances Avg\t' + str(sum(variances) / float(len(variances))))

if __name__ == "__main__": main()