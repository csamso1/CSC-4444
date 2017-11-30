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




#?
## for scaling data before input use
    #data = model.Scale(data, data, scale=float(1./256))
    # see mnist file line 78

#This is a file for experimenting with Fantasy Foootball data!
# Documentation for pyplot
# https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.html
def getQbWithName(qbName):
    os.chdir('/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI')
    players_data = pd.read_csv("RBAllYears.csv")
    players_data.reindex(index=['Year', 'Week', 'Name'])
    print(players_data.columns)
    print(players_data.index)
    players_data[players_data.Name == qbName].to_csv(qbName+".csv")
def sortIndividualQBData():
    dBrees_data = pd.read_csv('RBAllYears.csv',index_col=['Year','Week'])
    dBrees_data.sort_index(inplace=True)
    dBrees_data.drop('Unnamed: 0',axis=1,inplace=True)
    dBrees_data[dBrees_data.Name == 'Mark Ingram'].to_csv("Mark Ingram With Out Defense.csv")
def combineOffenseWithDefense():
    os.chdir('/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI')
    players_data = pd.read_csv("RBAllYears.csv")
    defense_data = pd.read_csv("DSTAllYears.csv")
    players_data.reindex(index=['Year', 'Week', 'Name'])
    print(players_data.columns)
    print(players_data.index)
    players_data[players_data.Name == 'Mark Ingram'].to_csv("Mark Ingram Data.csv")

    pd_data = pd.merge(players_data, defense_data, left_on=['Year', 'Week', 'Opponent'],
                       right_on=['Year', 'Week', 'Team'])
    print(pd_data.head())
    pd_data.to_csv("RB_with_Defense.csv")
def heatMap():

        heatmap, xedges, yedges = np.histogram2d(X, Y)#, bins=(64, 64))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.clf()
        plt.imshow(heatmap, extent=extent)
        plt.show()

def multivariateLinearRegression(file, yfeature):
    #works best if you pass the full file path instead of just the file name
        #Same exact thing as above except there are waaayyy more features. Above involved only one
        os.chdir('/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI')
        try:
            players_data = pd.read_csv(file)
        except:
            print("what")
            players_data = pd.read_csv('AllPositionsAllYears.csv')

        featuresList = list(players_data.columns)

        #Creates a set, phenominal for membership checking, unions, intersections, etc
        setList = set(featuresList)

        #Test if the yFeature argument is a valid column
        if (yfeature in setList) == False:
            raise Exception(str(yfeature)+'Dude, put in a valid parameter\nPlease try again with a valid feature')

        #Removes the yfeature from the group of Xs
        xFeatures = setList.difference([yfeature])
        # Remove the Fantasy Football Point Values from Data
        xFeatures = xFeatures.difference(['Points_x','Points_y','Points'])
        # Removes non numerical values from data
        xFeatures = xFeatures.difference(['Name','Opponent','Position','Opponent_x','Position_x',
                            'Team', 'Opponent_y', 'Position_y'])
        # Removes Features I feel are unimportant for Predicting QB stats
        xFeatures = xFeatures.difference(['Fumble TD', 'Fumbles Lost','Away Games', 'Away Games_x',
                            'Blocks', 'Fumble Returns',
                             'Safety', 'Away Games_y'])

        Y = players_data[yfeature]
        # Y = preprocessing.scale(Y)
        X = players_data[list(xFeatures)].values.reshape(-1,len(xFeatures))
        # X = preprocessing.scale(X)
        # XSquared = list(map(lam))
        # X = X[:, np.newaxis]
        # Y = Y[:, np.newaxis]

        # X_train, X_test, Y_train, Y_test = train_test_split(preprocessing.scale(X), preprocessing.scale(Y), test_size=.2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=.2)
        coeff_titles = list(xFeatures)
        coeff_titles.insert(0,'Intercept')


        reg = linear_model.LinearRegression()
        # reg = linear_model.PassiveAggressiveRegressor
        model = reg.fit(X_train, Y_train)

        # model = reg.fit(X_train, Y_train,Y_test)
        PassAttempts_prediction = model.predict(X_test)
        # print("Predicting: ",yfeature)
        print("The Coeffecients:\n ")
        #Let's format the list of float point coeffs so they're easier to read
        pattern = "%.2f"
        floatsstrings = [pattern % i for i in list(reg.coef_)]
        floats = list(float(i) for i in floatsstrings)
        hmmm = dict(zip(coeff_titles,floats))

        # hmmmDF = pd.DataFrame(hmmm)
        # print(hmmm);exit(0)
        mean2Err = mean_squared_error(Y_test, PassAttempts_prediction)
        # print("Score: ",model.score(X_test,Y_test))
        varianceScore = r2_score(Y_test, PassAttempts_prediction)
        # print("The Mean Squared Error\n", mean2Err)
        # Explained Variance Score: 1 is perfect prediction
        # print('Variance Score: %2.2f' % varianceScore)

        # pStr = str("\n\n\nPredicting: \t"+ yfeature)
        # cStr = str('The Coeffecients: %2.2f' %  reg.coef_)
        # sStr = str("\nScore: \t" + str(model.score(X_test,Y_test)))
        # mStr = str("\nMean^2 Error\t" + str(mean2Err))
        # vStr = str('\nVariance Score: \t%2.2f' % varianceScore)
        # return pStr + sStr + mStr + vStr
        # return (mean2Err,varianceScore)
        return (mean2Err,varianceScore,hmmm)
def groupPlayersByNameYear(players_data):
    os.chdir("/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI")
    player_data = pd.read_csv('QB_with_Defense.csv')
    #where players_data is a pandas dataframe
    their_stats = players_data.groupby(['Name', 'Year']).agg({'Pass Yards': [np.size, np.mean, np.sum ]})
    #Now to get a specific player using this format
    #dBrees_data = their_stats.loc[['Drew Brees', ], :]
    mark_ingram_data = their_stats.loc[['Mark Ingram', ], :]

#So here we can get a specific player then group their data into whatever we want, iterate through those groupings and get whatever stats we want on each column
#Yeah that's kinda a rant
def getSpecifiedStatisticsForSpecificPlayerGrouped():
    player_data = pd.read_csv("QB_with_Defense.csv", index_col=['Year', 'Week'].sort())
    ingram_data = player_data[player_data.Name == 'Mark Ingram']
    ingram_data.drop("Unnamed: 0", axis=1, inplace=True)
    grouped = brees_data.groupby(['Year'])
    for item, value in grouped:
        if item == 2012:
            print(np.mean(value['Pass Yards']))

# Grouped is defined the same as in getSpecifiedStatisticsForSpecificPlayerGrouped()
#This method will iterate through all the features a current player has, remove the non numerical ones, then calculate a descriptive statitistic for each
#Additional Loop will provided time valued data as needed for multinomimial linear regression
def iterateAndReceiveStats():
    for item, value in grouped:
        xFeatures = set(list(value.columns))
        xFeatures = xFeatures.difference(['Name', 'Opponent', 'Position', 'Opponent_x', 'Position_x',
                                          'Team', 'Opponent_y', 'Position_y'])
        for feature in xFeatures:
            valFeature = value[feature]
            # if valFeature.at[1].isnumeric()==True or valFeature.at[1].isdecimal()==True:
            print(np.mean(valFeature))

#This does what said needed to be done in iterateAndReceiveStats() method
#It loops and calculates a rolling mean, returning each new mean value a every week
#SHOULD CHECK THAT IT STARTS AT WEEK 1, INSTEAD OF COUNTING DOWN
#Also, these methods need to be returning a list or some usable datastructure instead of printing the results
    #recommend creating empty list and appending where my print statements are

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
                if feature == 'Pass Yards':
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


#<<<<<<<<<------------If you fell asleep, continue work here------------>>>>>
#Here I'm following the steps written in the piece of paper that's attatched to your clipboard.
#<<<<<<<<<------------If you fell asleep, continue work here------------>>>>>

def createInputData():
    os.chdir("/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI")
    player_data = pd.read_csv("RBAllYears.csv", index_col=['Year', 'Week'].sort())
    brees_data = player_data[player_data.Name == 'Mark Ingram']
    # brees_data.drop("Unnamed: 0", axis=1, inplace=True)
    grouped = brees_data.groupby(['Year'])
    newValuesAsDict = []
    for year, yearData in grouped:
        listOfAveragedList = []

        averagedList = []
        for dataColumn in purgeAlphas(yearData.columns):
            dataVector = yearData[dataColumn]

            for val in range(0, len(dataVector)):
                averagedList.append(np.mean(dataVector[0:val]))
        newValuesAsDict.append(dict({dataColumn:averagedList}))
    updatedDF = pd.DataFrame(newValuesAsDict)
    print(updatedDF.head(10))
    updatedDF.to_csv("Averaged_Ingram_Values.csv")


def main():
    ourPredictions = open('/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI/Mark Ingram Testing_this.txt','w')

    #These are the stats to predict. All of them have a nonzero coefficient in Fantasy Football Points algorithm for an Offensive player
    stats = ['Fumble TD', 'Fumbles Lost', 'Pass 2PT',
     'Pass Attempts', 'Pass Completions', 'Pass Interceptions', 'Pass TD',
     'Pass Yards', 'Receiving 2PT', 'Receiving TD',
     'Receiving Yards', 'Receptions', 'Rush 2PT', 'Rush Attempts', 'Rush TD',
     'Rush Yards']
    # These are the stats I'm actually concerned about accurately predicting right now
    statsUsing = ['Rush Attempts', 'Rush Yards', 'Rush TD', 'Receptions' ]


    # change to stat in stats in order to predict all relevant stats, instead of only the stats I'm currently caring about
    for stat in statsUsing:
        resultTuples = []
        for i in range(10):
            resultTuples.append(multivariateLinearRegression("/mnt/c/Users/Clayton/Documents/GitHub/CSC-4444/FF-AI/RBAllYears.csv",stat))
        mean2Errs, variances, hmmm = zip(*resultTuples)
        ourPredictions.write("\n\n\nPredicting: \t"+ stat)
        ourPredictions.write('\nMean^2 Error Avg\t'+str(sum(mean2Errs)/float(len(mean2Errs))))
        ourPredictions.write('\nVariances Avg\t' + str(sum(variances) / float(len(variances))))
        # What the hell was the point of the below code
        # please = hmmm[0]
        # print(please)
        # for huh in hmmm[1:]:
        #     for who in huh:
        #         please[who] += huh[who]

        # print(please)
        # for hm in please:
        #     ourPredictions.write(str(hm)+':\t'+str(sum(please[hm])/float(len(please[hm]))))
    ourPredictions.close()
    # multivariateLinearRegression("AllPositionsAllYears.csv", "Pass Attempts")

if __name__ == "__main__": main()