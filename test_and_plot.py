#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:28:38 2020

@author: ershov
"""


##NEEDS WORK##
##COME BACK TO THIS##

#Imports
import shutil 
import os
import matplotlib.pyplot as plt
import subprocess


command = ['python', 'datasets/combine_A_and_B.py', '--fold_A path/to/data/A', \
           '--fold_B path/to/data/B', '--fold_AB path/to/data']
subprocess.run(command)




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
def lossPlotFunc(G_GAN, G_loss, D_real, D_fake, combEpoch):


  plotPath = 'automated_tests/'
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
  plt.savefig('/content/drive/My Drive/CS231N/tempPlot1.jpg')

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
  plt.savefig('/content/drive/My Drive/CS231N/tempPlot2.jpg')



ershStr = 'test0_nom_iter1/ershov_lossFolder_2'
fullString = '/content/drive/My Drive/CS231N/' + ershStr

G_GAN, G_loss, D_real, D_fake, combEpoch = lossDataLoad(fullString)

print(G_loss)

lossPlotFunc(G_GAN, G_loss, D_real, D_fake, combEpoch)


