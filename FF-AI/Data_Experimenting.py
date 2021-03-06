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
# #from openpyxl.cell import get_column_letter
# from openpyxl.utils import get_column_letter
#Nicholas Dugal

# ?
## for scaling data before input use
# data = model.Scale(data, data, scale=float(1./256))
# see mnist file line 78

# This is a file for experimenting with Fantasy Foootball data!

# I frequently need to copy paste code for trials in jupyter notebook or python console. This is the reason for many
# methods that seem useless or repetitive. That's on purpose.
# The Method that we are most concerned with

# This returns a data frame of a particular player, set save=True to save it
def getPlayerWithName(pName, position, save=False, sorted=True):
    # change working directory to the correct one. If you're unsure what your current dir is then run (os.getcwd())
    os.chdir('/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI')
    players_data = pd.read_csv(position + "AllYears.csv")
    players_data.reindex(index=['Year', 'Week', 'Name'])
    playerDataFrame = players_data[players_data.Name == pName]
    if save:
        playerDataFrame.to_csv(pName + ".csv")
    if sorted:
        players_data.sort_index(inplace=True)
    return playerDataFrame


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


# delete this, I'm keeping for future other stuff
def heatMap():
    heatmap, xedges, yedges = np.histogram2d(X, Y)  # , bins=(64, 64))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap, extent=extent)
    plt.show()


# Runs linear regression on two variables. I can update if needed. May or may not break.
# X and Y should be two Pandas Series or numpy arrays, maybe also a regular list will work.
def linearRegression(X=None, Y=None):
    #os.chdir('data')
    # rb_all = os.listdir(os.getcwd())
    rb_data = pd.read_csv('RBAllYears.csv')
    print(rb_data.columns)
    if (X == None, Y == None):
        Y = rb_data['Rush Yards']
        X = rb_data['Rush Attempts']
    # XSquared = list(map(lam))

    X = X[:, np.newaxis]
    Y = Y[:, np.newaxis]

    # Creates two Dataframes, one for training with random samples consistenting of 80% the total samples, one for testing with the remaining 20%
    X_train, X_test = train_test_split(X, test_size=.2)
    Y_train, Y_test = train_test_split(Y, test_size=.2)

    # Here we create and fit the linear regression model. The call to fit iteratively updates the coeffecients of our model until it's mean squared difference is within
    # a certain threshold
    reg = linear_model.LinearRegression()
    reg.fit(X_train, Y_train)

    # Here we throw the testing data into the model. This returns an array of predicted y values
    PassAttempts_prediction = reg.predict(X_test)

    print("The Coeffecients:\n ", reg.coef_)
    # Gives us the average square distance between predicted values and the actual values
    # Essentially, how far from correct my model is
    mean2Err = mean_squared_error(Y_test, PassAttempts_prediction)
    # Variance ranges from 0 to 1, 1 is peeeeerfect!
    # If I expect myself to get 4 answers wrong, but actually get 10 wrong:high variance(close to 0), high error
    # If I actually get 5 wrong: low variance(close to 1), medium error
    # If I actually get 0 wrong: higher variance, no error
    varianceScore = r2_score(Y_test, PassAttempts_prediction)
    print("The Mean Squared Error\n", mean2Err)
    # Explained Variance Score: 1 is perfect prediction
    print('Variance Score: %.2f' % varianceScore)
    # Here we plot our testing data to get a look at it. Let's make sure a linear model is the model for us
    plt.scatter(X_test, Y_test, color='black')
    # Here we plot our model's line as defined during training against the testing data
    plt.plot(X_test, PassAttempts_prediction, color='blue', linewidth=3)

    plt.xticks()
    plt.yticks()
    # plt.annotate(('hmm','whatUP'))
    # Here we type code that belongs somewhere else
    logReg = linear_model.LogisticRegression()
    logReg.fit(X_train, Y_train)

    # This shows us what the values for our model is,
    # For N features, you have N+1 coefficients
    coef = reg.coef_[0].tolist()
    # This gives us the min and max values of our graph. Move along..
    xmin, _ = plt.xlim()
    _, ymax = plt.ylim()
    s = str('Our line is defined as \nhaving coefficient values of ' + str(coef) +
            '\nThe Mean Squared Error: ' + str(mean2Err) +
            '\nVariance Score: ' + str(varianceScore))
    plt.text(8, 30, s, rotation=40,
             horizontalalignment='center',
             verticalalignment='center',
             multialignment='center', color='red')

    # plt.scatter(X,Y, s=12,c='red',marker='.')
    # plt.xlim(-5,40)
    # plt.ylim(0,175)                         # xmin, xmax = xlim() use to get current values

    plt.title('Relationship Between Rush Attempts and Rush Yards')

    plt.ylabel('Rush Yards')
    plt.xlabel('Rush Attempts', multialignment='center')

    plt.show()


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

#Ignore This
def groupPlayersByNameYear(players_data):
    os.chdir("/Users/nickdugal/Documents/Fantasy-Football/data/Updated NFL Data Sets/Indexed Data/")
    player_data = pd.read_csv('QB_with_Defense.csv')
    # where players_data is a pandas dataframe
    their_stats = players_data.groupby(['Name', 'Year']).agg({'Pass Yards': [np.size, np.mean, np.sum]})
    # Now to get a specific player using this format
    dBrees_data = their_stats.loc[['Drew Brees', ], :]

#Ignore this
    # So here we can get a specific player then group their data into whatever we want, iterate through those groupings and get whatever stats we want on each column
    # Yeah that's kinda a rant
def getSpecifiedStatisticsForSpecificPlayerGrouped(player_data, name):
    #player_data = pd.read_csv("RBAllYears.csv", index_col=['Year', 'Week'].sort())
    player_data = player_data[player_data.Name == name]
    #individual_player_data.drop("Unnamed: 0", axis=1, inplace=True)
    # grouped = individual_player_data.groupby(['Year'])
    # for item, value in grouped:
    #     if item == 2012:
    #         print(np.mean(value['Rush Yards']))
    return player_data

#IGNore This
    # Grouped is defined the same as in getSpecifiedStatisticsForSpecificPlayerGrouped()
    # This method will iterate through all the features a current player has, remove the non numerical ones, then calculate a descriptive statitistic for each
    # Additional Loop will provided time valued data as needed for multinomimial linear regression
def iterateAndReceiveStats():
    for item, value in grouped:
        xFeatures = set(list(value.columns))
        xFeatures = xFeatures.difference(['Name', 'Opponent', 'Position', 'Opponent_x', 'Position_x',
                                          'Team', 'Opponent_y', 'Position_y'])
        for feature in xFeatures:
            valFeature = value[feature]
            # if valFeature.at[1].isnumeric()==True or valFeature.at[1].isdecimal()==True:
            print(np.mean(valFeature))

            # This does what said needed to be done in iterateAndReceiveStats() method
            # It loops and calculates a rolling mean, returning each new mean value a every week
            # SHOULD CHECK THAT IT STARTS AT WEEK 1, INSTEAD OF COUNTING DOWN
            # Also, these methods need to be returning a list or some usable datastructure instead of printing the results
            # recommend creating empty list and appending where my print statements are

#Ignore this
def cumulativeWeeklyMeans(grouped):
    # item is the feature we grouped on and value is it's assocciated dataframe
    for item, value in grouped:
        if item == 2012:
            xFeatures = set(list(value.columns))
            xFeatures = xFeatures.difference(['Name', 'Opponent', 'Position', 'Opponent_x', 'Position_x',
                                              'Team', 'Opponent_y', 'Position_y'])
            for feature in xFeatures:
                valFeature = value[feature]
                # if valFeature.at[1].isnumeric()==True or valFeature.at[1].isdecimal()==True:
                # print(np.mean(valFeature))
                # Here we begin finding the effect of time for features
                if feature == 'Rush Yards':
                    print('Weekly Values before means:, ', valFeature)
                    for val in range(0, len(valFeature)):
                        print(np.mean(valFeature[0:val]))


# Takes columns of a DF and removes the currently known nonNumerical values
def purgeAlphas(unCleaned):
    cleaned = (set(list(unCleaned))).difference(['Name', 'Opponent', 'Position', 'Opponent_x', 'Position_x',
                                                 'Team', 'Opponent_y', 'Position_y'])
    return cleaned


# Removes the nonNumerical columns of a DF
def removeAlphaData(unCleanedDF):
    return unCleanedDF[list(purgeAlphas(unCleanedDF))]

#Ignore this
def createInputData():
    os.chdir("/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI")
    player_data = pd.read_csv("RBAllYears.csv", index_col=['Year','Week'])
    ingram_data = player_data[player_data.Name == 'Mark Ingram']
    # brees_data.drop("Unnamed: 0", axis=1, inplace=True)
    grouped = ingram_data.groupby(['Year'])
    listOfAveragedList = []
    for year, yearData in grouped:

        newValuesAsDict = []

        for dataColumn in purgeAlphas(yearData.columns):
            dataVector = yearData[dataColumn]
            averagedList = []
            for val in range(0, len(dataVector)):
                averagedList.append(np.mean(dataVector[0:val]))
            newValuesAsDict.append(dict({dataColumn: averagedList}))
        listOfAveragedList.append((newValuesAsDict))
    updatedDF = pd.DataFrame(listOfAveragedList)
    print(updatedDF.head(10))
    updatedDF.to_csv("Averaged_Ingram_Values.csv")

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
    # simple = pd.DataFrame({'Hey':[1]})
    # simple.to_csv('simple.csv')
    # combineOffensePositionWithDefense('RB', save=True)
    # exit(0)
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
    #exporting this dataframe to CSV to make sure it is correct
    players_data.to_csv("enhanced_player.csv")
    #trimming down to just 1 players stats
    # players_data = getSpecifiedStatisticsForSpecificPlayerGrouped(players_data, player_name)
    # players_data.to_csv("just_individual_player_data.csv")

    #Running the simulation
    resultTuples.append(multivariateLinearRegression(statsUsing, statistic_to_predict, players_data))
    mean2Errs, variances = zip(*resultTuples)
    ourPredictions.write("Predicting: \t" + statistic_to_predict)
    ourPredictions.write('\nMean^2 Error Avg\t' + str(sum(mean2Errs) / float(len(mean2Errs))))
    ourPredictions.write('\nVariances Avg\t' + str(sum(variances) / float(len(variances))))
    # change to stat in stats in order to predict all relevant stats, instead of only the stats I'm currently caring about
    # for stat in statsUsing:
    #     resultTuples = []
    #     for i in range(10):
    #         resultTuples.append(multivariateLinearRegression(
    #             statsUsing, 'Rush Yards', players_data=combineOffensePositionWithDefense('RB'), file=None))
    #     mean2Errs, variances = zip(*resultTuples)
    #     ourPredictions.write("\n\n\nPredicting: \t" + stat)
    #     ourPredictions.write('\nMean^2 Error Avg\t' + str(sum(mean2Errs) / float(len(mean2Errs))))
    #     ourPredictions.write('\nVariances Avg\t' + str(sum(variances) / float(len(variances))))

    ourPredictions.close()
    print("Predicting: \t" + statistic_to_predict)
    print('\nMean^2 Error Avg\t' + str(sum(mean2Errs) / float(len(mean2Errs))))
    print('\nVariances Avg\t' + str(sum(variances) / float(len(variances))))

if __name__ == "__main__": main()