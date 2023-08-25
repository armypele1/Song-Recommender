import numpy as np
import pandas as pd
import dask as dd
import torch
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import random
import time
import multiprocessing as mp
import gc

# Non-personalised Popularity based Recommender System model
class PopularityRecommender():
    def __init__(self, userID='userID', itemID='song'):     
        self.userID = userID
        self.itemID = itemID
        self.popularityRecommendations = None
        
    #Create the popularity based recommender system model
    def fit(self, trainingData):
        self.trainingData = trainingData
        # Get a count of userIDs for each unique song as recommendation score
        groupedTrainingData = self.trainingData.groupby([self.itemID]).agg({self.userID: 'count'}).reset_index()
        groupedTrainingData.rename(columns = {self.userID: 'score'},inplace=True)
    
        # Sort songs based upon recommendation score
        sortedTrainingData = groupedTrainingData.sort_values(['score', self.itemID], ascending = False)
        sortedTrainingData['Ranking'] = sortedTrainingData['score'].rank(ascending=False, method='first')
        
        # Get the sorted recommendations
        self.popularityRecommendations = sortedTrainingData

    #Use the popularity based recommender system model to
    #make recommendations
    # self, u, k=5, removeListened=True, itemsToIgnore=None, output='list'
    def recommend(self, userID, k=10, itemsToIgnore=None, output='list'):    
        recommendations = self.popularityRecommendations.head(k)
        recommendations = recommendations.loc[:,'song'].values  
        if not itemsToIgnore == None:
            recommendations = [x for x in recommendations if x not in itemsToIgnore]
        if output=='df':
            recommendations = pd.DataFrame(recommendations, columns=["song"])

        return recommendations
    
    def getName(self):
        return "Non-personalised Recommender (Popularity)"
    
    @staticmethod
    def getDirectory():
        path = "./savedModels"
        if not os.path.isdir(path): 
            os.makedirs(path)
        return path
    
    def save(self, topn):
        path = self.getDirectory()
        topnPopularSongs = self.popularityRecommendations.head(topn)
        topnPopularSongs.to_csv(path+"/popularityModel.csv")
    
    def load(self):
        path = self.getDirectory()
        self.popularityRecommendations = pd.read_csv(path+"/popularityModel.csv")
    
    # Save some dataframe properties for the purposes of the console application
    def saveDataProperties(self, data):
        path = self.getDirectory()
        numUsers = data["userID"].nunique()
        numItems = data["song"].nunique()
        df = pd.DataFrame({'numUsers':numUsers, 'numItems':numItems}, index=[0])
        df.to_csv(path+"dfProperties.csv")
    
    def loadDataProperties(self):
        path = self.getDirectory()
        df = pd.read_csv(path+"dfProperties.csv")
        numUsers = df._get_value(0, 'numUsers')
        numItems = df._get_value(0, 'numItems')
        return numUsers, numItems

# Personalised Recommender System model using 'Embarrassingly Shallow Autoencoders for Sparse Data (EASE)'
class EASErecommender():
    def __init__(self, userCol='userID', itemCol='song', listenCount='plays', reg=0.05):
        self.reg = reg
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder() 
        self.listenCount = listenCount    
        self.userCol = userCol
        self.itemCol = itemCol 
        
    def fit(self, trainingData, useSavedEncoders=False):
        #(0) Load the dataset
        if useSavedEncoders:
            self.loadEncoderLabels()
        else:
            self.users = self.user_enc.fit_transform(trainingData.loc[:, self.userCol])
            self.items = self.item_enc.fit_transform(trainingData.loc[:, self.itemCol])

        self.users = self.user_enc.transform(trainingData.loc[:, self.userCol])
        self.items = self.item_enc.transform(trainingData.loc[:, self.itemCol])
        self.values = trainingData[self.listenCount].to_numpy() / trainingData[self.listenCount].max()
        #(1) Build the X matrix      
        X = csr_matrix((self.values, (self.users, self.items)))
        self.X = X
        #(2) Build the Gram matrix: G = X^t * X
        G = self.X.T.dot(self.X).toarray()  
        #(3) Apply regularization value (lambda) along the diagonal of G
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.reg
        #(4) P = G.inverse()
        P = np.linalg.inv(G)
        G = None
        gc.collect()
        #(5) B = -P/diagonals(P)
        B = (-1 * P) / np.diag(P)
        P = None
        gc.collect()
        #(6) Set diagonals to 0
        B[diagIndices] = 0
        diagIndices = None
        gc.collect()

        self.B = B
        self.pred = self.X.dot(B)

    def recommend(self, u, k=5, removeListened=True, itemsToIgnore=None, output='list'):
        r = self.X[u, :] @ self.B

        # Make previously listened songs the last to be recommended if removeListened=True
        if removeListened:
            r += -1.0 * self.X[u, :]
        scores = np.array(r).flatten()
        recommendations = list(np.argsort(-scores)[:k])

        #if output == "list":
        recommendations = self.item_enc.inverse_transform(recommendations) 
        
        # Used when measuring model performance
        if not itemsToIgnore == None:
            recommendations = [x for x in recommendations if x not in itemsToIgnore]
        if output=='df':
            recommendations = pd.DataFrame(recommendations, columns=["song"])
        return recommendations
    
    @staticmethod
    def getDirectory():
        path = "./savedModels"
        if not os.path.isdir(path): 
            os.makedirs(path)
        return path
    
    def save(self):
        path = self.getDirectory()
        save_npz(path+'/X.npz', self.X)
        np.save(path+'/B.npy', self.B, allow_pickle=True)
        with open(path+'/user_enc.pkl', 'wb') as f:
            pickle.dump(self.user_enc, f)
        with open(path+'/item_enc.pkl', 'wb') as f:
            pickle.dump(self.item_enc, f)
        print("Model saved to "+path)
    
    def load(self):
        path = self.getDirectory()  
        self.X = load_npz(path+'/X.npz')
        self.B = np.load(path+'/B.npy', allow_pickle=True)
        with open(path+'/user_enc.pkl', 'rb') as f:
            self.user_enc = pickle.load(f)
        with open(path+'/item_enc.pkl', 'rb') as f:
            self.item_enc = pickle.load(f)
    
    def loadEncoderLabels(self):
        path = self.getDirectory()
        with open(path+'/user_enc.pkl', 'rb') as f:
            self.user_enc = pickle.load(f)
        with open(path+'/item_enc.pkl', 'rb') as f:
            self.item_enc = pickle.load(f)
    
    def getName(self):
        return "Personalised Recommender (EASE)"

# Evaluation Model to measure the effectiveness of the two recommendation models
# I looked at this example as a basis for the top-n recall:
# https://www.kaggle.com/code/gspmoreira/recommender-systems-in-python-101/notebook#Evaluation
class ModelEvaluator():
    def __init__(self, allData, trainingData, testData, userCol='userID', itemCol='song', listenCount='plays'):
        self.allData = allData
        self.trainingData = trainingData
        self.testData = testData
        self.userCol = userCol
        self.itemCol = itemCol

        self.allDataIndexed = self.allData.set_index(userCol)
        self.trainingDataIndexed = self.trainingData.set_index(userCol)
        self.testDataIndexed = self.testData.set_index(userCol)
        path = "./savedModels"
        with open(path+'/user_enc.pkl', 'rb') as f:
            self.user_enc = pickle.load(f)
    
    # Get all items listened to by the user
    def getListenedItems(self, userID, df):
        listenedItems = df.loc[userID][self.itemCol]
        # make sure still works even if only one listened item
        return set(listenedItems if type(listenedItems) == pd.Series else [listenedItems])

    # Get all the items the user hasn't interacted with
    def getNotListenedItemsSample(self, userID, sampleSize, seed=12):
        listenedItems = self.getListenedItems(userID, self.allDataIndexed)
        allItems = set(self.allData[self.itemCol])
        notListenedItems = allItems - listenedItems

        random.seed(seed)
        notListenedItemsSample = random.sample(notListenedItems, sampleSize)
        return set(notListenedItemsSample)
    
    # Check the first n recommendations to see if they match a given item
    def checkIfTopN(self, itemID, recommendations, topn):
        try:
            index = next(i for i, c in enumerate(recommendations) if c == itemID) 
        except:
            index = -1
        # hit can only be true or false
        hit = int(index in range(0, topn))
        #print(hit)
        return hit, index
    
    def evaluateForUser(self, model, userID):
        # Get items in the test set
        listenedValuesTestSet = self.testDataIndexed.loc[userID]
        if type(listenedValuesTestSet[self.itemCol]) == pd.Series:
            userListenedItemsTestSet = set(listenedValuesTestSet[self.itemCol]) 
        else:
            userListenedItemsTestSet = set([listenedValuesTestSet[self.itemCol]]) 

        listenedItemsTestSetCount =  len(userListenedItemsTestSet)

        # Get the ranked recommendation list for given user
        if model.getName() == "Personalised Recommender (EASE)":
            userIDinput = model.user_enc.transform([userID])
        else:
            userIDinput = userID

        userRecommendationsdf = model.recommend(userIDinput, itemsToIgnore=self.getListenedItems(userID,self.trainingDataIndexed), k=1000, output='df')
        #print(userRecommendationsdf)
        hitsTop5Counter = 0
        hitsTop10Counter = 0

        for itemID in userListenedItemsTestSet:
            # Get random sample of 100 items user has not interacted with
            # Assumed to not be wanted by user
            notListenedItemsSample = self.getNotListenedItemsSample(userID, 100)
            # Combine current interacted item with the 100 unlistened items
            itemsForFilter = notListenedItemsSample.union(set([itemID]))

            # Filtering only for listened item or the sample of unlistened 
            validRecommendationsdf = userRecommendationsdf[userRecommendationsdf[self.itemCol].isin(list(itemsForFilter))]
            #print(validRecommendationsdf[itemID])
            validRecommendations = validRecommendationsdf[self.itemCol]#.values

            # Check if current item is in top N recommendations
            hitAt5, indexAt5 = self.checkIfTopN(itemID, validRecommendations, 5)
            hitsTop5Counter += hitAt5
            hitAt10, indexAt10 = self.checkIfTopN(itemID, validRecommendations, 10)
            hitsTop10Counter += hitAt10

        # Measure recall using these hits
        recallAt5 = hitsTop5Counter / float(listenedItemsTestSetCount)
        recallAt10 = hitsTop10Counter / float(listenedItemsTestSetCount)

        userMetrics = {
            'hitsAt5Count':hitsTop5Counter,
            'hitsAt10Count':hitsTop10Counter,
            'listenedCount':listenedItemsTestSetCount,
            'recallAt5':recallAt5,
            'recallAt10':recallAt10
        }

        return userMetrics

    # Calculates the top n recall for given model
    def evaluateModel(self, model):
        allUsersMetrics = []
        totalAmount = len(list(self.testDataIndexed.index.unique().values))
        i = 1
        print(totalAmount)
        for idx, userID in enumerate(list(self.testDataIndexed.index.unique().values)):
            t0 = time.time()
            userMetrics = self.evaluateForUser(model, userID)
            userMetrics['userID'] = userID
            allUsersMetrics.append(userMetrics)
            diff = time.time() - t0
            if i % 1000 == 0:
                print("Completed user "+str(i)+" of "+ str(totalAmount) + " in " + str(diff)+"s")
            i += 1
 
        detailedResultsdf = pd.DataFrame(allUsersMetrics).sort_values('listenedCount', ascending=False)
        globalRecallAt5 = detailedResultsdf['hitsAt5Count'].sum() / float(detailedResultsdf['listenedCount'].sum())
        globalRecallAt10 = detailedResultsdf['hitsAt10Count'].sum() / float(detailedResultsdf['listenedCount'].sum())

        global_metrics = {'modelName': model.getName(),
                          'recallAt5': globalRecallAt5,
                          'recallAt10': globalRecallAt10}    
        return global_metrics, detailedResultsdf


