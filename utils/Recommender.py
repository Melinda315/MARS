import time
from utils.Dataset import Dataset
import numpy as np
import torch
from torch.autograd import Variable
import heapq
from utils import evaluation


class Recommender(object):
    def __init__(self, args):
        self.cuda_available = torch.cuda.is_available()
        self.recommender = args.recommender
        self.numEpoch = args.numEpoch
        self.batch_size = args.batch_size
        self.embedding_dim = args.embedding_dim
        self.lRate = args.lRate
        self.topK = eval(args.topK)
        self.pop_reg = args.pop_reg
        self.reg1 = args.reg1
        self.reg2 = args.reg2
        self.num_negatives = args.num_negatives
        self.dataset = args.dataset
        self.margin = args.margin
        self.rand_seed = args.rand_seed
        np.random.seed(self.rand_seed)
        self.mode = args.mode
        self.cuda = args.cuda
        self.batchSize_test = args.batchSize_test
        self.early_stop = args.early_stop
        
        self.totalFilename = 'data/'+self.dataset+'/ratings.dat'
        self.trainFilename = 'data/'+self.dataset+'/LOOTrain.dat'
        self.valFilename = 'data/'+self.dataset+'/LOOVal.dat'
        self.testFilename = 'data/'+self.dataset+'/LOOTest.dat'
        self.negativesFilename = 'data/'+self.dataset+'/LOONegatives.dat'
        
        dataset = Dataset(self.totalFilename, self.trainFilename, self.valFilename, self.testFilename, self.negativesFilename)

        self.trainRatings, \
        self.valRatings, \
        self.testRatings, \
        self.negatives, \
        self.numUsers, \
        self.numItems, \
        self.userCache, \
        self.itemCache, \
        self.userInterest = dataset.trainMatrix, dataset.valRatings, dataset.testRatings, dataset.negatives, dataset.numUsers, dataset.numItems, dataset.userCache, dataset.itemCache, dataset.userInterest

        self.train = dataset.train
        self.totalTrainUsers, self.totalTrainItems = dataset.totalTrainUsers, dataset.totalTrainItems

        total_count = 0
        for iid in self.itemCache:
            total_count += len(self.itemCache[iid])
        total_count = 1.0 * total_count ** self.pop_reg

        self.item_popularity = np.zeros(self.numItems, dtype=np.float32)
        self.user_popularity = np.zeros(self.numUsers, dtype=np.float32)

        for iid in range(self.numItems):
            if iid in self.itemCache:
                if len(self.itemCache[iid]) > 0:
                    self.item_popularity[iid] = (1.0 * len(self.itemCache[iid])) ** self.pop_reg / total_count
                # self.item_popularity[iid] = self.item_popularity[iid] ** 0.5
            else:
                self.item_popularity[iid] = 0

        for uid in range(self.numUsers):
            if uid in self.userCache:
                if len(self.userCache[uid]) > 0:
                    self.user_popularity[uid] = (1.0 * len(self.userCache[uid])) ** self.pop_reg / total_count

        self.item_popularity /= np.sum(self.item_popularity)
        self.user_popularity /= np.sum(self.user_popularity)
        # self.trainNeg, self.trainNegPopularity = self.getTrainNegatives()
        # Evaluation
        self.bestHR, self.bestNDCG, self.bestMRR = dict(), dict(), dict()
        for k in self.topK:
            self.bestHR[k] = 0
            self.bestNDCG[k] = 0
            self.bestMRR[k] = 0
        self.early_stop_metric = []
        
    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s

    def is_converged(self, model, epoch, totalLoss, topHits, topNdcgs, topMrrs):

        for k in self.topK:
            self.bestHR[k] = max(self.bestHR[k], topHits[k])
            self.bestNDCG[k] = max(self.bestNDCG[k], topNdcgs[k])
            self.bestMRR[k] = max(self.bestMRR[k], topMrrs[k])

        # if epoch % 10 == 0:
        print("[%s] [iter=%d %s] Loss: %.2f, margin: %.3f [%.4f, %.4f, %.4f] | [%.4f, %.4f, %.4f] | [%.4f, %.4f, %.4f]" %(self.recommender, epoch+1, self.currentTime(), totalLoss, self.margin,
                                                                                        self.bestHR[5],self.bestHR[10],self.bestHR[20],
                                                                                        self.bestNDCG[5],self.bestNDCG[10],self.bestNDCG[20],
                                                                                        self.bestMRR[5],self.bestMRR[10],self.bestMRR[20]))

        self.early_stop_metric.append(self.bestHR[10])
        if self.mode == 'Val' and epoch > self.early_stop and self.bestHR[10] == self.early_stop_metric[epoch-self.early_stop]:
            print("[%s] [Final (Early Converged)] [%.4f, %.4f, %.4f] | [%.4f, %.4f, %.4f] | [%.4f, %.4f, %.4f] "%(self.recommender,
                                                                                            self.bestHR[5],self.bestHR[10],self.bestHR[20],
                                                                                            self.bestNDCG[5],self.bestNDCG[10],self.bestNDCG[20],
                                                                                            self.bestMRR[5],self.bestMRR[10],self.bestMRR[20]))
            return True

    def printFinalResult(self):
        print("[%s] [Final] [%.4f, %.4f, %.4f] | [%.4f, %.4f, %.4f] | [%.4f, %.4f, %.4f] "%(self.recommender,
                                                                                            self.bestHR[5],self.bestHR[10],self.bestHR[20],
                                                                                            self.bestNDCG[5],self.bestNDCG[10],self.bestNDCG[20],
                                                                                            self.bestMRR[5],self.bestMRR[10],self.bestMRR[20]))

    def evalScore(self, model):
        topHits = dict(); topNdcgs = dict(); topMrrs = dict()
        for topK in self.topK:
            hits = []; ndcgs = []; mrrs = []
            for idx in range(len(self.test.keys())):
                users = Variable(self.test[idx]['u'])
                items = Variable(self.test[idx]['i'])
                offsets = self.test[idx]['offsets']

                vals, _ = model(users, items)

                if self.cuda_available:
                    items = items.cpu().data.numpy().tolist()
                    vals = vals.cpu().data.numpy().tolist()
                else:
                    items = items.data.numpy().tolist()
                    vals = vals.data.tolist()

                for i in range(len(offsets)-1):
                    from_idx = offsets[i]
                    to_idx = offsets[i+1]
                    cur_items = items[from_idx:to_idx]
                    cur_vals = vals[from_idx:to_idx]

                    gtItem = cur_items[-1]
                    
                    map_item_score = dict(zip(cur_items, cur_vals))

                    ranklist = heapq.nlargest(topK, map_item_score, key=map_item_score.get)
                    hr = evaluation.getHitRatio(ranklist, gtItem)
                    ndcg = evaluation.getNDCG(ranklist, gtItem)
                    mrr = evaluation.getMRR(ranklist, gtItem)

                    hits.append(hr)
                    ndcgs.append(ndcg) 
                    mrrs.append(mrr)

            hr, ndcg, mrr = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(mrrs).mean()
            topHits[topK] = hr; topNdcgs[topK] = ndcg; topMrrs[topK] = mrr
            
        return topHits, topNdcgs, topMrrs

    def getMarginOffset(self, uids):
        if self.cuda_available == True:
            uids = uids.cpu().data.numpy().tolist()
        else:
            uids = uids.data.numpy().tolist()

        offsets = []
        for uid in uids:
            offsets.append(self.userInterest[uid])

        if self.cuda_available:
            return Variable(torch.FloatTensor(offsets)).cuda(self.cuda)
        else:
            return Variable(torch.FloatTensor(offsets))

    def getTestInstances(self):
        trainItems = set(self.train.iid.unique())
        test=dict()
        # Make test data
        input = range(self.numUsers)
        bins = [input[i:i+self.batchSize_test] for i in range(0, len(input), self.batchSize_test)]

        for bin_idx, bin in enumerate(bins):
            userIdxs = []
            itemIdxs = []
            prevOffset = 0
            offset = [0]
            for uid in bin:
                if self.mode == 'Val':
                    rating = self.valRatings[uid]
                else:
                    rating = self.testRatings[uid]
                items = self.negatives[uid]
                items = list(trainItems.intersection(set(items)))
                u = rating[0]
                assert (uid == u)
                gtItem = rating[1]
                if gtItem not in trainItems:
                    continue
                items.append(gtItem)

                users = [u] * len(items)

                userIdxs += users
                itemIdxs += items
                offset.append(prevOffset + len(users))
                prevOffset += len(users)

            test.setdefault(bin_idx, dict())
            test[bin_idx]['offsets'] = offset
            if self.cuda_available == True:
                test[bin_idx]['u'] = torch.LongTensor(np.array(userIdxs)).cuda(self.cuda)
                test[bin_idx]['i'] = torch.LongTensor(np.array(itemIdxs)).cuda(self.cuda)

            else:
                test[bin_idx]['u'] = torch.LongTensor(np.array(userIdxs))
                test[bin_idx]['i'] = torch.LongTensor(np.array(itemIdxs))
                
        return test

    def getTrainInstancesByPopularity(self):
        trainItems = set(self.train.iid.unique())

        totalData = []
        for s in range(self.numUsers * self.num_negatives):
            # generate a random user u
            u = np.random.choice(self.numUsers, p=self.user_popularity)
            # u's positive items
            cu = self.userCache[u]
            if len(cu) == 0:
                continue

            # get a positive item at random
            t = np.random.choice(len(cu))
            i = cu[t]
            ci = self.itemCache[i]
            if len(ci) == 0:
                continue

            # random a negative item by populartiy
            j = np.random.choice(np.arange(self.numItems), p=self.item_popularity)
            while j in cu or j not in trainItems:
                j = np.random.choice(np.arange(self.numItems), p=self.item_popularity)
            totalData.append([u, i, j])

        totalData = np.array(totalData)
        return totalData
