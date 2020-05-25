#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:28:38 2020

@author: ershov
"""


#We want to flesh out 3 aspects during testing:
#-Inception Scores for validation and testing images
#-Image creation scheme (tiles 3-5 images, sketch + BW + gen, for some model)
#-being able to choose between different models (not just latest one)
#-plotting losses


#Imports
import shutil 
import os
import matplotlib.pyplot as plt
import subprocess


#command = ['python', 'datasets/combine_A_and_B.py', '--fold_A', 'path/to/data/A', \
#           '--fold_B', 'path/to/data/B', '--fold_AB', 'path/to/data']
#subprocess.run(command)






#Define loading and plotting functions

def lossDataLoad(inStr):

  G_GAN = []
  G_loss = []
  D_real = []
  D_fake = []

  with open(inStr + '/loss.txt', 'r') as filehandle:
      #It's a list of lists of floats
      fullFile = filehandle.readlines()
  cntr = 0
  for i in fullFile:
    cntr += 1
    if i == '-\n':
      cntr = 0
      continue
    elif cntr == 1:
      G_GAN.append(float(i[:-2]))
    elif cntr == 2:
      G_loss.append(float(i[:-2]))
    elif cntr == 3:
      D_real.append(float(i[:-2]))
    elif cntr == 4:
      D_fake.append(float(i[:-2]))

  epoch = []
  epoch_iter = [] 
  with open(inStr + '/epoch.txt', 'r') as filehandle:
      #It's a list of 2 numbers for epoch
      fullFile = filehandle.readlines()
  cntr = 0
  for i in fullFile:
    if i == '-\n':
      cntr = 0
      continue
    else:
      cntr += 1
      if cntr == 1:
        epoch.append(float(i[:-2]))
      else:
        epoch_iter.append(float(i[:-2]))
  combEpoch = []
  for i in range(len(epoch)):
    bigE = epoch[i]
    endE = epoch_iter[-1]

    temp = (bigE - 1)*endE + epoch_iter[i]
    combEpoch.append(temp)

  return G_GAN, G_loss, D_real, D_fake, combEpoch



#plotting function
def lossPlotFunc(G_GAN, G_loss, D_real, D_fake, combEpoch, path, name):


  fig, axs = plt.subplots(1, 2, figsize=(15,5))
  axs[0].plot(combEpoch, G_GAN)
  axs[0].set_title('Generator Error', fontsize = 20)
  axs[0].set_xlabel('Batch Number', fontsize = 20)
  axs[0].set_ylabel('Error', fontsize = 20)
  axs[0].grid(True)

  axs[1].plot(combEpoch, G_loss)
  axs[1].set_title('Generator Loss', fontsize = 20)
  axs[1].set_xlabel('Batch Number', fontsize = 20)
  axs[1].set_ylabel('Loss', fontsize = 20)
  axs[1].grid(True)
  plt.savefig(path + name + 'loss_G.jpg')

  fig, axs = plt.subplots(1, 2, figsize=(15,5))
  axs[0].plot(combEpoch, D_real)
  axs[0].set_title('Discr Real Confidence', fontsize = 20)
  axs[0].set_xlabel('Batch Number', fontsize = 20)
  axs[0].set_ylabel('Confidence', fontsize = 20)
  axs[0].grid(True)

  axs[1].plot(combEpoch, D_fake)
  axs[1].set_title('Discr Fake Confidence', fontsize = 20)
  axs[1].set_xlabel('Batch Number', fontsize = 20)
  axs[1].set_ylabel('Confidence', fontsize = 20)
  axs[1].grid(True)
  plt.savefig(path + name + 'loss_D.jpg')


#First, set the seed directory and make directories if they dont exist
#subproc will return 0 is successful, 1 if exists
####INPUT HERE####
seedNum = 1
####INPUT HERE####
subSeedPath = 'automated_tests_gcp/seed_' + str(seedNum)
seedPath = 'automated_tests_gcp/seed_' + str(seedNum) + '/test_result'
command = ['mkdir', seedPath]
a = subprocess.run(command)

compPath = seedPath + '/image_comparisons'
lossPath = seedPath + '/loss_plots'
isPath = seedPath + '/inception_plots'
resultPath = seedPath + '/saved_results'
miscPath = seedPath + '/misc'
tempPath = seedPath + '/temp'

if a.returncode == 0:
    #Means the directory has not been created yet
    print('Creating subdirectories...')
    
    command = ['mkdir', compPath]
    a = subprocess.run(command)
    
    command = ['mkdir', lossPath]
    a = subprocess.run(command)
    
    command = ['mkdir', isPath]
    a = subprocess.run(command)
    
    command = ['mkdir', resultPath]
    a = subprocess.run(command)
    
    command = ['mkdir', miscPath]
    a = subprocess.run(command)
    
    command = ['mkdir', tempPath]
    a = subprocess.run(command)
else:
    pass
    

####IMPLEMENTING RETRIEVAL OF PROPER MODELS AND IMAGES BELOW#####
#First, need to get a list of run directories
#Then need to retrieve the correct models
seedDirList = os.listdir(subSeedPath) 
runDirList = []

for i in seedDirList:
    if i[0:4] == 'run_':
        runDirList.append(i)
    else:
        pass
    
#Makes sure that the list is sorted alphabetically/by length with 2 sorts
runDirList.sort()
runDirList.sort(key=len) 
runNumList = [int(i[4:]) for i in runDirList]


#Now we choose the model to look at for all of the runs
####INPUT HERE####
#chosenModel = '5'
#For the latest model, use:
chosenModel = 'latest'
####INPUT HERE####


###IMPLEMENTING LOSS SCORES 






    

    
    




#####Ref code below for plotting stuff
#
#ershStr = 'test0_nom_iter1/ershov_lossFolder_2'
#fullString = '/content/drive/My Drive/CS231N/' + ershStr
#
#G_GAN, G_loss, D_real, D_fake, combEpoch = lossDataLoad(fullString)
#
#print(G_loss)
#
#lossPlotFunc(G_GAN, G_loss, D_real, D_fake, combEpoch)


