# Boltzmann Machines

# Importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # for
import torch.nn.parallel  # for parallel computation
import torch.optim as optim  # for optimizer
import torch.utils.data  # for the tools we are going to use
from torch.autograd import Variable  # Stochastic Gradient descent

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')

users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Preparing the training set and the test set

training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
# Total number of users
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]  # all the movies for a user
        id_ratings = data[:, 2][data[:, 0] == id_users]  # all the rating for a user
        ratings = np.zeros(nb_movies)  # create a new list of 1682
        ratings[
            id_movies - 1] = id_ratings  # id_movies start from 1 and so, we make sure indexes start from 0 & we assigned ratings for each id
        new_data.append(ratings)  # adding ratings of 1 particular user
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# converting the data into torch tensors
training_set = torch.FloatTensor(training_set)  # FloatTensor Expects lists of list
test_set = torch.FloatTensor(test_set)

# Convert the ratings into binary ratings 1 (Liked) or 0 (Not liked)
training_set[training_set == 0] = -1  # all the 0 values of training_set

# Ratings from 1-2 will be given 0 to mean the user did not like
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0

# Ratings bigger than or equal will get 1 to mean user liked it
training_set[training_set >= 3] = 1

# For test set
test_set[test_set == 0] = -1  # all the 0 values of training_set
# Ratings from 1-2 will be given 0 to mean the user did not like
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
# Ratings bigger than or equal will get 1 to mean user liked it
test_set[test_set >= 3] = 1


# Creating an architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):  # nv means number of visible nodes, nh means number of hidden nodes
        self.W = torch.randn(nh, nv) # weights
        self.a = torch.randn(1, nh) # bias for hidden nodes
        self.b = torch.randn(1, nv) # bias for visible nodes

    def sample_h(self, x): # x is visible neurons
        wx = torch.mm(x, self.W.t()) # transpose of W
        activation = wx + self.a.expand_as(wx) # expand_as is used to make sure a is added to each line of wx
        p_h_given_v = torch.sigmoid(activation) # probability of hidden neurons given visible neurons
        return p_h_given_v, torch.bernoulli(p_h_given_v) # bernoulli will return 0 or 1

    def sample_v(self, y): # y is hidden neurons
        wy = torch.mm(y, self.W) # transpose of W
        activation = wy + self.b.expand_as(wy) # expand_as is used to make sure b is added to each line of wy
        p_v_given_h = torch.sigmoid(activation) # probability of visible neurons given hidden neurons
        return p_v_given_h, torch.bernoulli(p_v_given_h) # bernoulli will return 0 or 1

    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


nv = len(training_set[0]) # number of visible nodes
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size): # 0 to 943, 100
        vk = training_set[id_user: id_user + batch_size] # visible neurons at k
        v0 = training_set[id_user: id_user + batch_size] # visible neurons at 0
        ph0, _ = rbm.sample_h(v0) # probability of hidden neurons at 0
        for k in range(10):
            _, hk = rbm.sample_h(vk) # hidden neurons at k
            _, vk = rbm.sample_v(hk) # visible neurons at k
            vk[v0 < 0] = v0[v0 < 0] # we don't want to update the missing ratings
        phk, _ = rbm.sample_h(vk) # probability of hidden neurons at k
        rbm.train(v0, vk, ph0, phk) # training the RBM
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])) # mean of absolute difference between v0 and vk
        s += 1. # increment s
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s)) # print epoch and loss

## Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users): # for each user
    v = training_set[id_user:id_user + 1] # visible neurons
    vt = test_set[id_user:id_user + 1]
    if len(vt[vt >= 0]) > 0: # if there are ratings
        _, h = rbm.sample_h(v) # sample hidden neurons
        _, v = rbm.sample_v(h) # sample visible neurons
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0])) # mean of absolute difference between vt and v
        s += 1.
print('test loss: ' + str(test_loss / s))         # print test loss

