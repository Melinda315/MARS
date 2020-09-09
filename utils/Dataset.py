import scipy.sparse as sp
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from numpy import random
import time

class Dataset(object):
    def __init__(self, totalFilename, trainFilename, valFilename, testFilename, negativesFilename):

        print("Reading Dataset")
        self.totalData = pd.read_csv(totalFilename, sep='\t')[['uid','iid']]
        self.train = pd.read_csv(trainFilename, sep='\t')[['uid','iid']]
            
        self.trainMatrix = self.load_rating_file_as_matrix(trainFilename)
        self.valRatings = self.load_rating_file_as_list(valFilename)
        self.testRatings = self.load_rating_file_as_list(testFilename)
        self.negatives = self.load_negative_file(negativesFilename)

        assert len(self.testRatings) == len(self.negatives)
        self.numUsers, self.numItems = len(self.totalData.uid.unique()), len(self.totalData.iid.unique())
        
        self.userTsCache = {}
        self.userCache = self.getuserCache()
        self.itemTsCache = {}
        self.itemCache = self.getitemCache()
        self.userInterest = self.getUserInterest()
        self.totalTrainUsers = set(self.train.uid.unique())
        self.totalTrainItems = set(self.train.iid.unique())
        
        print("[Rating] numUsers: %d, numItems: %d, numRatings: %d]" %(self.numUsers, self.numItems, len(self.trainMatrix)))
        
        # Free memory
        self.totalData.drop(self.totalData.index, inplace=True)
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0
                line = f.readline()    
        return mat


    def getuserCache(self):
        train = self.train
        userCache = {}

        for uid in train.uid.unique():
            items = train.loc[train.uid == uid][['iid']].values
            userCache[uid] = items[:, 0].astype(np.int64).tolist()
        
        return userCache
    
    def getitemCache(self):
        train = self.train
        totalUsers = set(range(self.numUsers))
        itemCache = {}
        for iid in range(self.numItems):
            users = train.loc[train.iid == iid][['uid']].values
            itemCache[iid] = users[:, 0].astype(np.int64).tolist()

        return itemCache

    def getUserInterest(self):
        userInterest = []
        for uid in range(self.numUsers):
            userNeighborSet = set()
            for iid in self.userCache[uid]:
                userNeighborSet = userNeighborSet.union(set(self.itemCache[iid]))
            userInterest.append(1.0 * len(userNeighborSet) / self.numUsers)

        return userInterest


class Dataset_MARS(Dataset):
    def __init__(self, totalData):
        self.totalData = totalData

    def __len__(self):
        return len(self.totalData)

    def __getitem__(self, idx):
        result = {'u': self.totalData[idx, 0], 'i': self.totalData[idx, 1], 'j': self.totalData[idx, 2]}
        return result
