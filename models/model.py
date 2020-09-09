import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils.Recommender import Recommender
from utils.Dataset import Dataset_MARS
from torch.utils.data import DataLoader
import os
import numpy as np


class modelFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, userEmbeds, itemEmbeds, is_pos):
        # batch_size * K * embedding_size
        output = torch.sum((userEmbeds * itemEmbeds), 2)
        # batch_size * K
        ctx.save_for_backward(userEmbeds, itemEmbeds, output)
        ctx.in1 = is_pos
        return output

    @staticmethod
    def backward(ctx, grad_output):
        userEmbeds, itemEmbeds, output = ctx.saved_tensors
        is_pos = ctx.in1
        uf = userEmbeds * output.view(-1, userEmbeds.shape[1], 1)
        vf = itemEmbeds * output.view(-1, itemEmbeds.shape[1], 1)

        if not is_pos:
            userEmbeds_grad = (itemEmbeds - uf)  # v - fu
        else:
            userEmbeds_grad = (uf - itemEmbeds)  # hu - v


        if not is_pos:
            itemEmbeds_grad = (userEmbeds - vf)  # u - hv
        else:
            itemEmbeds_grad = (vf - userEmbeds)  # u - hv

        mask = ((grad_output.view(userEmbeds.shape[0], userEmbeds.shape[1], 1)) != 0).float()
        userEmbeds_grad = userEmbeds_grad * mask
        itemEmbeds_grad = itemEmbeds_grad * mask
        return userEmbeds_grad, itemEmbeds_grad, None


class MARS(Recommender):
    def __init__(self, args):
        Recommender.__init__(self, args)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.rand_seed)
        torch.manual_seed(self.rand_seed)

        if self.cuda_available:
            self.clip_max = torch.FloatTensor([1.0]).cuda(self.cuda)
        else:
            self.clip_max = torch.FloatTensor([1.0])

        if self.cuda_available:
            self.margin_tensor = Variable(torch.FloatTensor([self.margin])).cuda(self.cuda)
        else:
            self.margin_tensor = Variable(torch.FloatTensor([self.margin]))

        self.test = self.getTestInstances()
        self.k = args.k
        self.eta = args.eta

        print('initial model')
        self.model = modeler(self.numUsers, self.numItems, self.k, self.embedding_dim, self.cuda_available, self.cuda, self.eta)
        if self.cuda_available:
            self.model = self.model.cuda(self.cuda)

    # Training
    def training(self):
        model = self.model
        criterion = torch.nn.MarginRankingLoss(margin=self.margin, reduction='sum')
        optimizer = optim.SGD(model.parameters(), lr=self.lRate)
        # Initial performance
        model.eval()
        topHits, topNdcgs, topMrrs = self.evalScore(model)

        model.train()

        print("[%s] [Initial %s] [%.4f, %.4f, %.4f] | [%.4f, %.4f, %.4f] | [%.4f, %.4f, %.4f] " % (
        self.recommender, self.currentTime(),
        topHits[5], topHits[10], topHits[20],
        topNdcgs[5], topNdcgs[10], topNdcgs[20],
        topMrrs[5], topMrrs[10], topMrrs[20]))

        for epoch in range(self.numEpoch):
            totalLoss = 0
            data_path = os.path.join('data', self.dataset, 'samples', 'sampling_' + str(epoch) + '.npy')
            if os.path.exists(data_path):
                totalData = np.load(data_path)
            else:
                totalData = self.getTrainInstancesByPopularity()

                np.save(data_path, totalData)
            train_by_dataloader = Dataset_MARS(totalData)
            train_loader = DataLoader(dataset=train_by_dataloader, batch_size=self.batch_size, shuffle=True)
            for batch_idx, batch in enumerate(train_loader):
                u = Variable(batch['u'])
                i = Variable(batch['i'])
                j = Variable(batch['j'])

                if self.cuda_available == True:
                    u = u.cuda(self.cuda)
                    i = i.cuda(self.cuda)
                    j = j.cuda(self.cuda)

                optimizer.zero_grad()

                margin_offsets = self.margin_tensor - self.getMarginOffset(u)

                # Observed (positive) interaction
                pos, facet_loss = model(u, i, True)
                neg, _ = model(u, j, False)

                if self.cuda_available == True:
                    loss = criterion(neg + margin_offsets, pos, Variable(torch.FloatTensor([-1])).cuda(self.cuda))
                else:
                    loss = criterion(neg + margin_offsets, pos, Variable(torch.FloatTensor([-1])))

                loss += self.reg1 * (torch.max(1 - pos, Variable(torch.FloatTensor([0])).cuda(self.cuda)).sum())

                loss += self.reg2 * facet_loss

                loss.backward()

                optimizer.step()
                totalLoss += loss.item()

            model.eval()
            topHits, topNdcgs, topMrrs = self.evalScore(model)
            model.train()

            if self.is_converged(model, epoch, totalLoss, topHits, topNdcgs, topMrrs):
                return

        self.printFinalResult()


class modeler(nn.Module):
    def __init__(self, numUsers, numItems, k, embedding_dim, cuda_available, gpunum, eta):
        super(modeler, self).__init__()
        self.embedding_dim = embedding_dim

        self.userEmbed = nn.Embedding(numUsers * k, embedding_dim)
        self.itemEmbed = nn.Embedding(numItems * k, embedding_dim)

        self.userProb = nn.Embedding(numUsers, k)

        self.cos_grad = modelFunction.apply
        self.k = k
        self.cuda_available = cuda_available
        self.init_weights()
        self.gpunum = gpunum

        self.eta = eta

    def k_normalize(self, tensor):
        return F.normalize(tensor, p=2, dim=1)

    def init_weights(self):
        nn.init.normal_(self.userEmbed.weight.data, mean=0.0, std=0.01)
        nn.init.normal_(self.itemEmbed.weight.data, mean=0.0, std=0.01)

        tensor = self.userEmbed.weight.data[:, :]
        self.userEmbed.weight.data[:, :] = self.k_normalize(tensor)

        tensor = self.itemEmbed.weight.data[:, :]
        self.itemEmbed.weight.data[:, :] = self.k_normalize(tensor)

    def _get_offsets_u(self, u):
        if self.cuda_available == True:
            uids = u.cpu().data.numpy().tolist()
        else:
            uids = u.data.numpy().tolist()

        k_uids = []
        for uid in uids:
            k_uids.append([(uid * self.k + offset) for offset in range(self.k)])

        if self.cuda_available == True:
            return Variable(torch.LongTensor(k_uids)).cuda(self.gpunum)
        else:
            return Variable(torch.LongTensor(k_uids))

    def _get_offsets_i(self, i):
        if self.cuda_available == True:
            iids = i.cpu().data.numpy().tolist()
        else:
            iids = i.data.numpy().tolist()

        k_iids = []
        for iid in iids:
            k_iids.append([(iid * self.k + offset) for offset in range(self.k)])  # item has k embedding
            # k_iids.append([iid for offset in range(self.k)])  # item has one embedding

        if self.cuda_available == True:
            return Variable(torch.LongTensor(k_iids)).cuda(self.gpunum)
        else:
            return Variable(torch.LongTensor(k_iids))

    def getUserEmbeds(self, u):
        user_idx = self._get_offsets_u(u)
        return self.userEmbed(user_idx)

    def getItemEmbeds(self, i):
        item_idx = self._get_offsets_i(i)
        return self.itemEmbed(item_idx)

    def forward(self, u, i, is_pos=True):

        facet_embed = []

        prob_u = self.userProb(u)
        prob_u = F.softmax(prob_u, dim=-1)
        k_probs = prob_u
        # N * K * D
        tensor = self.userEmbed.weight.data[:, :]
        self.userEmbed.weight.data[:, :] = self.k_normalize(tensor)
        tensor = self.itemEmbed.weight.data[:, :]
        self.itemEmbed.weight.data[:, :] = self.k_normalize(tensor)

        userEmbeds = self.getUserEmbeds(u)
        itemEmbeds = self.getItemEmbeds(i)
        for l in range(self.k):
            facet_embed.append(userEmbeds[:, l, :])
        # N * K

        k_dis = self.cos_grad(userEmbeds, itemEmbeds, is_pos)

        # N * 1
        facet_loss = 0
        out = torch.sum(k_probs * k_dis, 1)

        for l in range(self.k):
            for j in range(l + 1, self.k):
                facet_loss += (1 / self.eta) * torch.log(
                    1 + torch.exp(-0.1 * torch.sum(facet_embed[l] * facet_embed[j])))

        return out, facet_loss
