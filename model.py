#!/usr/bin/env python3
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import torch.utils.data
import torchvision.transforms as transforms
import random

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024
batch_size = 64

######## ADVERSARIAL ATTACKS #############################################################################

########## Fast Gradient Sign Method attack ####################################
### model : the network to attack
### x : the vanilla inputs
### y : the true classes (untargeted) / the target classes (targeted)
### eps : the stepsize
### targeted : wether to use the targeted attack or not
def attack_FGSM(model, x, y, eps, targeted=False):
    model.eval()            # We put the model in eval mode to avoid any stochasticity
    x.requires_grad_(True)  # We track the gradients of the input

    logits = model(x)
    loss   = F.cross_entropy(logits, y)
    loss.backward()

    if not targeted:    x_adv  = x.detach() + eps * torch.sign(x.grad)  # We increase the loss wrt true classes
    else:               x_adv  = x.detach() - eps * torch.sign(x.grad)  # We decrease the loss wrt target classes

    x = x.detach()
    model.zero_grad()
    return x_adv.detach()


########## Projects points onto the lp-norm if outside ###########
### x : points to be projected
### eps : radius of norm-ball
### p : Lp norm of ball
def project(x, eps, p=2):
    if p=="inf":
        x = torch.clamp(x,-eps,eps)
    else:
        norms = torch.norm(x.view(x.size(0),-1),dim=1,p=p)
        norms[norms==0] = 1                          # To avoid division by zero
        mask  = (norms > eps)                        # We select only inputs of the batch with a norm > eps
        x[mask] /= norms[mask].view(-1,1,1,1)        # Project onto p-norm ball of radius 1
        x[mask] *= eps                               # Multiply by eps
    return x

########## Projected Gradient Descent attack ###################################
### model : the network to attack
### x : the vanilla inputs
### y : the true classes (untargeted) / the target classes (targeted)
### eps : the radius of the norm-ball in which we want our perturbation
### stepsize : the stepsize
### iterations : the number of iterations
### p : the norm to use (l2, linf...)
### targeted : wether to use the targeted attack or not
def attack_PGD(model, x, y, eps, stepsize, iterations, p=2, targeted=False):
    model.eval()  # We put the model in eval mode to avoid any stochasticity

    # Start with a random point inside norm-ball
    if p=="inf":
        delta = (torch.rand(x.size()) * 2 - 1) * eps    # Each values ~ uniform(-eps,eps)
    else:
        delta = torch.randn_like(x)                                             # Generate random direction
        norms = torch.norm(delta.view(delta.size(0),-1), dim=1, p=p)
        norms[norms==0] = 1                                                     # To avoid division by zero
        delta /= norms.view(-1,1,1,1)                                           # Project onto p-norm ball of radius 1
        delta *= torch.rand((delta.size(0),1,1,1)).to(device) * eps                        # Multiply each value by f ~ uniform(0,eps)

    delta = delta.to(device)
    # Iterately take a step of fixed norm, the project on p-norm ball if necessary
    for i in range(iterations):
        x_adv = x + delta
        x_adv.requires_grad_(True)
        logits = model(torch.clamp(x_adv,0,1))
        loss   = F.cross_entropy(logits,y)
        loss.backward()

        # Update image with a fixed stepsize
        if p=="inf":
            gradient = torch.sign(x_adv.grad) * stepsize
        else:
            gradient  = x_adv.grad                                                # Take the gradient direction
            norms     = torch.norm(gradient.view(gradient.size(0),-1),dim=1,p=p)
            norms[norms==0] = 1                                                   # To avoid division by zero
            gradient /= norms.view(-1,1,1,1)                                      # Project onto p-norm ball of radius 1
            gradient *= stepsize                                                  # Multipy by stepsize

        if not targeted :   x_adv = x_adv.detach() + gradient
        else:               x_adv = x_adv.detach() - gradient

        # Project delta onto p-norm ball of radius eps if necessary
        delta = (x_adv - x).detach()
        delta = project(delta, eps, p)

    x = x.detach()
    model.zero_grad()
    return torch.clamp(x + delta,0,1)
###############################################################################################################################

'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)  # 32 * 32
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2) # 16 * 16
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2) # 8 * 8
        #self.conv4 = nn.Conv2d(64, 64, 5, padding=2) # 4 * 4
        self.fc1 = nn.Linear(64 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1  = nn.BatchNorm2d(32)
        self.bn2  = nn.BatchNorm2d(64)
        self.bn3  = nn.BatchNorm2d(64)
        #self.bn4  = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        #x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))


    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.

           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''
        self.load(os.path.join(project_dir, Net.model_file))



def train_model(net, train_loader, pth_filename, num_epochs, adv=""):
    '''Basic training function (TODO)'''
    print("Starting training")
    optimizer = optim.SGD(net.parameters(),lr=0.1, momentum=0.0005)
    scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=0)
    mean_loss = 0

    net.train()

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        print("--- Training epoch {}".format(epoch))
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            if adv=="PGD":
              inputs = attack_PGD(net,inputs,labels,8/255,2/255,20,"inf",False)
              net.train()

            optimizer.zero_grad()
            logits = net(inputs)
            loss   = F.cross_entropy(logits,labels)
            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            if i>0 and i % 50 == 0:
              print(f"* Batch {i}/{len(train_loader)} : {mean_loss/50}")
              mean_loss = 0

            # implement training procedure here...

        scheduler.step()
        mean_loss = 0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def test_adv(net, test_loader, attack_fn):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    for i,data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)

        images_adv = attack_fn(net, images, labels)

        # calculate outputs by running images through the network
        outputs = net(images_adv)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=128):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid

def main():


    #### Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([ transforms.RandomHorizontalFlip(),
                                               transforms.RandomRotation(10),
                                               transforms.RandomCrop(size=32, padding=4),
                                               transforms.ToTensor()])

        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        train_model(net, train_loader, args.model_file, args.num_epochs,"PGD")
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()
