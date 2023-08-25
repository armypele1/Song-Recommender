print("loading Pretrained Recommender Models...")
import numpy as np
import pandas as pd
import Recommenders as rs
import os

personalisedRS = rs.EASErecommender()
personalisedRS.load()

popularityRS = rs.PopularityRecommender()
popularityRS.load()

numUsers, numItems = popularityRS.loadDataProperties()
numUsers -= 1
numItems -= 1

def getNumUsers(df, userCol):
    numUsers = len(df[userCol].unique())
    return numUsers

def getNumItems(df, itemCol):
    numItems = len(df[itemCol].unique())
    return numItems

def getValidInputs(numUsers, numItems):
    while True:
        user = input("Select a user number (1-"+str(numUsers)+") ")
        if user.isdigit() and int(user) > 0 and int(user) <= numUsers:
            break
        print("invalid input, try again.")

    while True:
        k = input("Select number of items to recommend (max "+str(numItems)+") ")
        if k.isdigit() and int(k) > 0 and int(k) <= numItems:
            break
        print("invalid input, try again.")
    
    return int(user), int(k)

def outputRecommendations(recommendations):
    print("\n### RECOMMENDATIONS ###")
    for i in range(len(recommendations)):
        print("("+str((i+1))+") " + recommendations[i])

menuString = "Select an option: \n\
    1. Get personalised recommendation \n\
    2. Get non-personalised recommendation \n\
    3. Clear Screen \n\
    4. Exit \n"

userInput = "0"
while userInput != "5":
    userInput = input(menuString)

    if userInput == "1":
        user, k = getValidInputs(numUsers, numItems)              
        recommendations = personalisedRS.recommend(user, k, removeListened=True)
        outputRecommendations(recommendations)
  
    elif userInput == "2":
        user, k = getValidInputs(numUsers, numItems)
        recommendations = popularityRS.recommend(user, k)
        outputRecommendations(recommendations)

    elif userInput == "3":
        os.system('cls')

    elif userInput == "4":
        break
    else:
        print("invalid input")
    print("")

