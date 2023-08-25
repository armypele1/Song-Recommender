# Song-Recommender

The purpose of my recommender system is to recommend new, unheard music to users that they will react 
to most positively based on the music they have previously listened to. The performance of this RS should be adequate such that recommendations are accurate and can be 
obtained within a reasonable time frame. Furthermore, these recommendations should be original and occasionally somewhat unusual, so as not to constrain the user to a 
narrow genre of music. This recommender system performs this task using both a personalised and non-personalised approach.

Please note that this repo's main purpose is to demonstrate my understanding of recommender systems. The code above will not run without a .csv file that is not in this repo due to its large file size (~1GB). If anyone would like to run this recommender system, please contact me and I can send you the appropriate files.

## Data Description

The dataset used in this project is the Million Song Dataset (MSD), which is a collection of audio features and 
metadata for a million contemporary music tracks. The dataset was created in December 2010, so the songs are not current with what is released today. Despite this, 
the dataset serves as a solid proof of concept to verify the suitability of the recommender system before applying it to a 
real-world problem. The dataset is stored in an HDF5 format, with each song 
being its own separate .h5 file.
On analysing the data, it was found that 12 of the 55
fields are consistently available across all songs in the dataset. Only these were considered when determining the 
appropriate features for the recommendation algorithm.
Alongside this, a complementary dataset called the Taste Profile Subset was utilized, which contains user play counts of songs that are all found within the MSD. This 
dataset is formatted as rows of triplets, where the 3 features are the user ID, song ID and listen count for that pair.
There is a website dedicated to the MSD, which contains useful information on various features of the dataset: http://millionsongdataset.com/. This includes important information 
on some matching errors between the Taste Profile and MSD, which are addressed when cleaning the data.

## Data Preparation

The large amount of data in the MSD made it unreasonable to immediately begin testing on the entirety of the dataset. Instead, initial data preparation and model training were performed on a random subset of the data, which is readily available on the MSD website.
The first stage of data preparation was getting the dataset in a suitable format to begin feature selection. The MSD is exclusively available in a hdf5 file format [2]. The preferred file type for a dataset in this recommender system workflow is a .csv file because the Python module ‘Pandas’ can read the file type into memory very efficiently.
Consequently, a Python script was used to merge and convert all .h5 files into a single .csv file (<strong>See MSDtoCSV.py</strong>). 

This script followed a chunking approach, which allowed for its re-use on the larger dataset later.
The next stage of data preparation was merging the newly generated songs.csv file with the Taste Profile Subset. This was done using a dedicated Python script, again using a chunking approach to allow for reuse with the full dataset. It was at this stage that the match-ing errors discussed in section 2.1 were addressed. The MSD website gave a list of song IDs that contain match-ing errors. Any time an ID from this list was found, the row was discarded.
Initially, 12 fields were extracted for each user-song pair. However, this was reduced to 3 once the recommendation algorithm was properly determined. These fields are (1) userID, (2) listenCount, and (3) song, where song is a concatenation of the fields ‘Title’ and ‘artistName’ which essentially acts as both an output for the system and a key for the song. This reduction in data was done because the algorithm follows a collaborative approach and does not require extensive content information.


Once the recommendation algorithm was implemented and tentatively evaluated, the full MSD was extracted from an AWS EC2 instance snapshot [3]. The same process as described was used to extract the necessary features. However, the implementation was improved to use the multiprocessing module when building the initial .csv file. This improved the efficiency of this process by 12 times.

## Recommendation Techniques

The algorithm that was used to construct the personalised recommender system is the EASE (Embarrassingly Shallow Autoencoders for Sparse Data) model [4]. This is a state-of-the-art linear model that is geared towards sparse data – particularly implicit feedback data. This makes it well suited to the problem at hand.
The model has been shown to achieve better ranking accuracy than many state-of-the-art collaborative filtering approaches such as deep non-linear models. It takes the premise of fewer hidden layers in an autoencoder being beneficial to recommendation accuracy and pushes it to an extreme. It is an autoencoder without a hidden layer, where an input vector containing items the user has interacted with is linearly encoded to an output of fewer dimensions. Self-similarity of an item in the input layer to the same item in the output layer is forbidden. This is what forces the model to generalize.
To acquire the parameters for this model, an item-item weight matrix needs to be produced. This is similar to a neighborhood-based approach. The algorithm that constructs this matrix can be found in <strong>Recommenders.py</strong>.

The algorithm used to construct the non-personalised recommender system is popularity-based. This means that the total listen counts for every song in the database are calculated, and songs are ranked according to this count. The actual recommendation to the user is the head of this ranking list. The length of this head is determined by the user. It should be noted that this recommender gives the same results for every user in the system. Please also see <strong>Recommenders.py</strong> for the full implementation.

## User Interface

The input interface is an easy-to-understand set of numbered options. Upon choosing a recommendation type, the user is requested to enter a user number to receive recommendations for that user. It was chosen to use the encoded number as opposed to the full user IDs because their length would make it tedious to enter them into the system.
There is an additional option to clear the screen, in the case that many recommendations have been made and the command line looks cluttered.

![input interface](Images/inputInterface.png?raw=true "input interface")


Recommendations are displayed in a simple list, which is labelled with rankings from 1 to n, where n is the number of recommendations requested by the user.

![output interface](Images/outputInterface.png?raw=true "output interface")
