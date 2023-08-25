#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tables
import os
import hdf5_getters
import fnmatch
import sys
import h5py
import gc
import multiprocessing as mp
import time


# In[2]:


class Song:
    def __init__(self, fileLocation):
        self.h5 = hdf5_getters.open_h5_file_read(fileLocation)  
        self.id = None
        self.title = None
        self.year = None
        self.duration = None
        #self.genreList = None
        self.danceability = None
        self.tempo = None
        self.timeSignature = None
        self.timeSignatureConfidence = None
        self.artistID = None
        self.artistName = None
        self.artistLocation = None
        self.artistLongitude = None
        self.artistLatitude = None
        self.albumID = None
        self.albumName = None
    
    # Given a song h5 file, extract all wanted attributes from it using hdf5_getters
    def getAttributes(self):
        h5 = self.h5
        self.id = hdf5_getters.get_song_id(h5).decode('UTF-8')
        self.title = hdf5_getters.get_title(h5).decode('UTF-8')
        self.year = hdf5_getters.get_year(h5)
        self.duration = hdf5_getters.get_duration(h5)
        #self.genreList = hdf5_getters.
        self.danceability = hdf5_getters.get_danceability(h5)
        self.tempo = hdf5_getters.get_tempo(h5)
        self.timeSignature = hdf5_getters.get_time_signature(h5)
        self.timeSignatureConfidence = hdf5_getters.get_time_signature(h5)
        self.artistID = hdf5_getters.get_artist_id(h5).decode('UTF-8')
        self.artistName = hdf5_getters.get_artist_name(h5).decode('UTF-8')
        self.artistLocation = hdf5_getters.get_artist_location(h5).decode('UTF-8')
        self.artistLongitude = hdf5_getters.get_artist_longitude(h5)
        self.artistLatitude = hdf5_getters.get_artist_latitude(h5)
        self.albumID = hdf5_getters.get_release_7digitalid(h5)
        self.albumName = hdf5_getters.get_release(h5).decode('UTF-8')
    
    # Convert the song object into a row that can be appended to csv
    def convertToRow(self):
        attributes = list((self.__dict__).values())[1:]
        outputString = ""
        for attribute in attributes:
            outputString += str(attribute).replace(',',' ') + ","
        outputString = outputString[0:len(outputString)-1]
        outputString += "\n"
        return outputString
    
    # Get the name of all attributes to easily construct the first row of the csv file
    def getHeaders(self):
        attributes = list((self.__dict__).keys())[1:]
        outputString = ""
        for attribute in attributes:
            outputString += str(attribute).replace(',',' ') + ","
        outputString = outputString[0:len(outputString)-1]
        outputString += "\n"
        return outputString
    
    # Close the h5 file to prevent memory leak
    def closeSongFile(self):
        self.h5.close()


# In[3]:


#gc.enable()
def walkThroughFolder(subfolder):
    letter = subfolder.split("/")[-1]
    outputName = 'SongCSV'+letter+'.csv'
    f = open(outputName, 'w', encoding="utf-8")
    hasHeader = False
    for root, dirnames, filenames in os.walk(subfolder):
        for filename in fnmatch.filter(filenames, '*.h5'):
            currentFilePath = os.path.join(root, filename)     
            s = Song(currentFilePath)
            s.getAttributes()  
            # If file is empty write in the headers, use boolean flag so doesn't check file every iteration (slow)
            if (not hasHeader) and os.stat(outputName).st_size == 0:
                f.write(s.getHeaders())  
                hasHeader = True
            f.write(s.convertToRow())
            #sys.exit(0)
            # Close file, delete object and manually collect garbage 
            s.closeSongFile()
            del s
            gc.collect()
    f.close()

    return outputName

def mergeFiles(filenames):
    outputName = "SongCSV.csv"
    firstFileAppended = False
    for filename in filenames:
        newSection = pd.read_csv(filename,encoding="utf-8")
        if firstFileAppended:          
            newSection.to_csv(outputName, encoding="utf-8", mode='a', header=False, index=False)
        else:
            newSection.to_csv(outputName, encoding="utf-8", mode='w', index=False)


#'./MSD/millionsongsubset'
if __name__ == "__main__":
    t0 = time.time()
    #numProcessors = mp.cpu_count()
    numProcessors = 8
    print("Using " + str(numProcessors) + " processors in parallel for scraping.")
    pool = mp.Pool(numProcessors)

    path = './MSD/millionsongsubset'
    subFolders =  next(os.walk('./MSD/millionsongsubset'))[1]
    subFolders = [path+"/"+x for x in subFolders]
    print(subFolders)
    params1 = subFolders[0:8]
    params2 = subFolders[9:16]
    params3 = subFolders[17:24]

    params1 = subFolders
    results = pool.map(walkThroughFolder, params1)
    results = pool.map(walkThroughFolder, params2)
    results = pool.map(walkThroughFolder, params3)
    
    pool.terminate()
    pool.join()
    diff = time.time() - t0

    mergeFiles(results)
    print(diff)



