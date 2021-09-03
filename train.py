#install pip packages
#pip install azureml-sdk[notebooks]
#pip install azureml-core
#pip install azure-storage-blob

# importing necessary libraries
from azureml.core import Workspace, Datastore, Dataset
from azureml.data.dataset_factory import DataType
from azureml.core.model import Model
from azure.storage.blob import BlobServiceClient
from azureml.core.run import Run

# importing sklearn libraries
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

# Useful for good split of data into train and test
from sklearn.model_selection import train_test_split

# import pandas
import pandas as pd

# linear algebra
import numpy as np

# import re package
import re

#import os package
import os

# import joblib
import joblib

#import time
import time

run = Run.get_context()

# get existing workspace
ws = Workspace.from_config()

# setup credentials for blob
STORAGEACCOUNTURL = "https://machinelearninxxxxxxxxxx.blob.core.windows.net/"
STORAGEACCOUNTKEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
LOCALFILENAME = "Titanic.csv"
CONTAINERNAME = "azureml-blobstore-xxxxxxx-xxxxxx-xxxx-xxxxx-xxxxxxxxxx"
BLOBNAME = "Titanic.csv"

# download from blob
t1=time.time()
blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
blob_client_instance = blob_service_client_instance.get_blob_client(CONTAINERNAME, BLOBNAME, snapshot=None)
with open(LOCALFILENAME, "wb") as my_blob:
    blob_data = blob_client_instance.download_blob()
    blob_data.readinto(my_blob)
t2=time.time()
print(("It takes %s seconds to download "+BLOBNAME) % (t2 - t1))

# LOCALFILE is the file path
titanic_ds = pd.read_csv(LOCALFILENAME)

# convert ‘Sex’ feature into numeric
genders = {"male": 0, "female": 1}
data = [titanic_ds]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

# since the most common port is Southampton the chances are that the missing one is from there
titanic_ds['Embarked'].fillna(value='S', inplace=True)

# convert ‘Embarked’ feature into numeric
ports = {"S": 0, "C": 1, "Q": 2}
data = [titanic_ds]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

# convert ‘Survived’ feature into numeric
ports = {False: 0, True: 1}
data = [titanic_ds]
for dataset in data:
    dataset['Survived'] = dataset['Survived'].map(ports)

# a cabin number looks like ‘C123’ and the letter refers to the deck.
# therefore we’re going to extract these and create a new feature, that contains a persons deck. 
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [titanic_ds]
for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

# drop cabin since we have a deck feature
titanic_ds = titanic_ds.drop(['Cabin'], axis=1)

# fix age features missing values
data = [titanic_ds]
for dataset in data:
    mean = titanic_ds["Age"].mean()
    std = titanic_ds["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = titanic_ds["Age"].astype(int)

# convert ‘age’ to a feature holding a category
data = [titanic_ds]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# create titles
data = [titanic_ds]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)

# drop name and title column since we have create a title
titanic_ds = titanic_ds.drop(['Name','Ticket'], axis=1)

# default missing fare rates
titanic_ds['Fare'].fillna(value=titanic_ds.Fare.mean(), inplace=True)

# convert fare to a feature holding a category
data = [titanic_ds]
for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

# create not_alone and relatives features
data = [titanic_ds]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

# create age class
data = [titanic_ds]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

# create fare per person
data = [titanic_ds]
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

# convert all data to numbers
le = preprocessing.LabelEncoder()
titanic_ds=titanic_ds.apply(le.fit_transform)

print("Show first records of all the features created")
titanic_ds.head(10)

# convert all data to numbers
le = preprocessing.LabelEncoder()
titanic_ds=titanic_ds.apply(le.fit_transform)

# split our data into a test (30%) and train (70%) dataset
test_data_split = 0.30
msk = np.random.rand(len(titanic_ds)) < test_data_split 
test = titanic_ds[msk]
train = titanic_ds[~msk]

# drop ‘PassengerId’ from the train set, because it does not contribute to a persons survival probability
train = train.drop(['PassengerId'], axis=1)

# train_test_split is a function to split the dataset X (inputs) and y (output) into X_train,X_test,y_train,y_test respectively.
# shows 0.2 for testing data, therefore 0.8 for training data. shuffle=True means shuffling data.
# X_train - This includes your all independent variables,these will be used to train the model, also as we have specified the test_size = 0.4, this means 60% of observations from your complete data will be used to train/fit the model and rest 40% will be used to test the model.
# X_test - This is remaining 40% portion of the independent variables from the data which will not be used in the training phase and will be used to make predictions to test the accuracy of the model.
# Y_train - This is your dependent variable which needs to be predicted by this model, this includes category labels against your independent variables, we need to specify our dependent variable while training/fitting the model.
# Y_test - This data has category labels for your test data, these labels will be used to test the accuracy between actual and predicted categories.
X_train, X_test, Y_train, Y_test = train_test_split(train.drop("Survived", axis=1), train["Survived"],test_size=0.4,random_state=54,shuffle=True)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

# Save model as pickle file
joblib.dump(random_forest, "outputs/random_forest.pkl")

# Predict and get result
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

#complete run
run.log("Random Forest accuracy", acc_random_forest)
run.complete()


