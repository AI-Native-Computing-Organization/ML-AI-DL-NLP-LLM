"""##Importing the libraries"""

import numpy as np #to install numpy
import pandas as pd #to install pandas
import torch #to install pytorch
import torch.nn as nn #to install pytorch
import torch.nn.parallel #to install pytorch
import torch.optim as optim #to install pytorch
import torch.utils.data #
from torch.autograd import Variable #to install pytorch

"""## Importing the dataset"""

# We won't be using this dataset.
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') #to import the dataset
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

"""## Preparing the training set and the test set"""

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

"""## Getting the number of users and movies"""

nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

"""## Converting the data into an array with users in lines and movies in columns"""

def convert(data):
  new_data = []
  for id_users in range(1, nb_users + 1): #to convert the data into an array with users in lines and movies in columns
    id_movies = data[:, 1] [data[:, 0] == id_users] #to convert the data into an array with users in lines and movies in columns
    id_ratings = data[:, 2] [data[:, 0] == id_users] #to convert the data into an array with users in lines and movies in columns
    ratings = np.zeros(nb_movies) #to convert the data into an array with users in lines and movies in columns
    ratings[id_movies - 1] = id_ratings #to convert the data into an array with users in lines and movies in columns
    new_data.append(list(ratings)) #to convert the data into an array with users in lines and movies in columns
  return new_data
training_set = convert(training_set) #to convert the data into an array with users in lines and movies in columns
test_set = convert(test_set) #to convert the data into an array with users in lines and movies in columns

"""## Converting the data into Torch tensors"""

training_set = torch.FloatTensor(training_set) #to convert the data into torch tensors
test_set = torch.FloatTensor(test_set) #to convert the data into torch tensors

"""## Creating the architecture of the Neural Network"""

class SAE(nn.Module): #to create the SAE class
    def __init__(self, ): #to initialize the class
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20) #to set the first fully connected layer
        self.fc2 = nn.Linear(20, 10) #to set the second fully connected layer
        self.fc3 = nn.Linear(10, 20) #to set the third fully connected layer
        self.fc4 = nn.Linear(20, nb_movies) #to set the fourth fully connected layer
        self.activation = nn.Sigmoid()#to set the activation function
    def forward(self, x):  #to define the forward function
        x = self.activation(self.fc1(x)) #to set the activation function
        x = self.activation(self.fc2(x)) #to set the activation function
        x = self.activation(self.fc3(x)) #to set the activation function
        x = self.fc4(x) #to set the activation function
        return x #to return the output
sae = SAE() #to set the SAE
criterion = nn.MSELoss() #to set the criterion
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) #to set the optimizer

"""## Training the SAE"""

nb_epoch = 200 #Setting the number of epochs
for epoch in range(1, nb_epoch + 1): #for each epoch
  train_loss = 0  #Setting the train loss to 0
  s = 0.  #Setting the s to 0
  for id_user in range(nb_users): #for each user in the dataset
    input = Variable(training_set[id_user]).unsqueeze(0) #to convert the data into torch tensors
    target = input.clone() #to clone the input
    if torch.sum(target.data > 0) > 0: #if the target data is greater than 0
      output = sae(input) #to get the output
      target.require_grad = False #to set the target data to false
      output[target == 0] = 0 #to set the output to 0
      loss = criterion(output, target) #to get the loss
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #to get the mean corrector
      loss.backward() #to get the loss
      train_loss += np.sqrt(loss.data*mean_corrector) #to get the train loss
      s += 1. #to increment s
      optimizer.step() #to update the weights
  print('epoch: '+str(epoch)+' loss: '+ str(train_loss/s)) #to print the epoch and loss

"""## Testing the SAE"""

test_loss = 0 #Setting the test loss to 0
s = 0. #Setting the s to 0
for id_user in range(nb_users): #for each user in the dataset
  input = Variable(training_set[id_user]).unsqueeze(0) #to convert the data into torch tensors
  target = Variable(test_set[id_user]).unsqueeze(0) #to convert the data into torch tensors
  if torch.sum(target.data > 0) > 0: #if the target data is greater than 0
    output = sae(input) #to get the output
    target.require_grad = False #to set the target data to false
    output[target == 0] = 0 #to set the output to 0
    loss = criterion(output, target) #to get the loss
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #to get the mean corrector
    test_loss += np.sqrt(loss.data*mean_corrector) #to get the test loss
    s += 1. #to increment s
print('test loss: '+str(test_loss/s)) #to print the test loss