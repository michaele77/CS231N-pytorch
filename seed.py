#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:13:41 2020

@author: ershov
"""

#Imports
import shutil 
import os
import matplotlib.pyplot as plt
import random
import subprocess


##INPUT HERE##
#We first want to get inputs for grid bounds and number for each type of grid
numRuns = 20
numEpochs = 50
numEpochsDecay = 50

#For learning rate
lr_list = []
for x in range(numRuns):
#  lr_list.append(random.randint(1*100,3*100)*1e-4*1e-2)
    lr_list.append(random.randint(1.5*100,4*100)*1e-4*1e-2)
print(lr_list)

#gan mode
gan_list = []
#gan_ref = ['vanilla', 'lsgan', 'wgangp']
gan_ref = ['wgangp']
for x in range(numRuns):
#  gan_list.append(gan_ref[random.randint(0,2)])
  gan_list.append(gan_ref[0])
print(gan_list)

#number of epochs
epoch_list = []
for x in range(numRuns):
  epoch_list.append(numEpochs)
print(epoch_list)


#number of epochs decay
epochDecay_list = []
for x in range(numRuns):
  epochDecay_list.append(numEpochsDecay)
print(epochDecay_list)


#beta1 size
beta_list = []
for x in range(numRuns):
#  beta_list.append(random.randint(3*100,7*100)*1e-1*1e-2)
  beta_list.append(random.randint(2*100,8*100)*1e-1*1e-2)
print(beta_list)


#policy type
policy_list = []
policy_ref = ['linear', 'step', 'plateau', 'cosine']
for x in range(numRuns):
  policy_list.append(policy_ref[random.randint(0,3)])
print(policy_list)




#Now create the seed directory
#Always choose the highest seed directory value
#basePath = '/content/drive/My Drive/CS231N/'
autoPath = 'automated_tests/'

dirList = os.listdir(autoPath)

currNum = len(dirList)
seedFile = 'seed_' + str(currNum)
command = ['mkdir', autoPath + seedFile]
subprocess.run(command)

seedPath = autoPath + seedFile + '/'

print('Created seed directory ' + seedFile)


#Now create the seed.txt and check_run.txt

#The seed.txt file has the following structure per number or iterations, has 7 inputs per iteration
#Run ID
#learning rate
#gan mode
#number of epochs
#number of epochs decay
#beta1 size
#policy type

with open(seedPath + 'seed.txt', 'w') as filehandle:
    #It's a list of 2 numbers for epoch
    for i in range(len(lr_list)):    
        filehandle.writelines("%d\n" % i)
        filehandle.writelines("%.6f\n" % lr_list[i])
        filehandle.writelines("%s\n" % gan_list[i])
        filehandle.writelines("%d\n" % epoch_list[i])
        filehandle.writelines("%d\n" % epochDecay_list[i])
        filehandle.writelines("%.6f\n" % beta_list[i])
        filehandle.writelines("%s\n" % policy_list[i])


#The check_run.txt file has the following structure per number or iterations, has 7 inputs per iteration
#Run ID
#learning rate
#gan mode
#number of epochs
#number of epochs decay
#beta1 size
#policy type

with open(seedPath + 'check_run.txt', 'w') as filehandle:
    #It's a list of 2 numbers for epoch
    for i in range(len(lr_list)):    
        filehandle.writelines("%d\n" % i)
        filehandle.writelines("%d\n" % 0)
        filehandle.writelines("%d\n" % 0)



#Now we're done!