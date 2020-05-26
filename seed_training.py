#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:22:08 2020

@author: ershov
"""

#Imports
import shutil 
import os
import matplotlib.pyplot as plt
import random
import subprocess

##INPUT HERE##
#Need to determine which seed we want to run
#currNum = 1
txtIn = input('What seed number to train with? Please input: ')
currNum = int(txtIn)

txtIn = input('New Data? Need to join directories [1 for yes]? Please input: ')
willJoin = int(txtIn)

autoPath = 'automated_tests/'
seedFile = 'seed_' + str(currNum)
seedPath = autoPath + seedFile + '/'

#Extract the seed.txt and check_run.txt file information into lists
#First, pull seed.txt into a full file
with open(seedPath + 'seed.txt', 'r') as filehandle:
  fullFile = filehandle.readlines()

numParams = 7
numRuns = len(fullFile) // numParams

#populate the lists, make sure to remove the newline /n from strings
id_list = []
lr_list = []
gan_list = []
epoch_list = []
epochDecay_list = []
beta_list = []
policy_list = []
for i in range(numRuns):
  id_list.append(int(fullFile[numParams*i + 0]))
  lr_list.append(float(fullFile[numParams*i + 1]))
  gan_list.append(str(fullFile[numParams*i + 2][:-1]))
  epoch_list.append(int(fullFile[numParams*i + 3]))
  epochDecay_list.append(int(fullFile[numParams*i + 4]))
  beta_list.append(float(fullFile[numParams*i + 5]))
  policy_list.append(str(fullFile[numParams*i + 6][:-1]))



#Now, pull check_run.txt into a full file
with open(seedPath + 'check_run.txt', 'r') as filehandle:
  check_run_file = filehandle.readlines()
  #will use this file as reference for re-writing stuff back later

#populate the lists, 
started_run = []
finished_run = []
numChecks = 3
for i in range(numRuns):
  #Already know the id list
  started_run.append(int(check_run_file[numChecks*i + 1]))
  finished_run.append(int(check_run_file[numChecks*i + 2]))


#reference for future code:
# temp = check_run_file.copy()
# temp[numChecks*2 + 1] = str(700000) + '\n'

# with open(seedPath + 'check_run.txt', 'w') as filehandle:
#   filehandle.writelines(temp)
  
  #Make sure the folds are combined
  if willJoin == 1:
    print('combining datasets...')
    command = ['python', 'datasets/combine_A_and_B.py', '--fold_A', 'path/to/data/A', \
               '--fold_B', 'path/to/data/B', '--fold_AB', 'path/to/data']
    temp = subprocess.run(command)
    print('finished combining, output = ' + str(temp))
  else:
    print('datasets already combined...')

#!python datasets/combine_A_and_B.py --fold_A path/to/data/A --fold_B path/to/data/B --fold_AB path/to/data



##!!MAIN TRAINING LOOP!!##
#Now we start our main training loop

#See the following logic:
#   itrate through started/finished list, continue if both are finished; if either is not start counter there
#   Once counter is set, loop in range of the start counter to the end of list
#   Inside the loop, we create directory of the run ID, we set the correct command list based on params and execute the command
#   Once done, copy over the the ershov loss log file as loss and the checkpoints file in checkpoints
#   Be sure to overwrite the check_run.txt log at the start of the loop and at the end!

firstRun = 0
while started_run[firstRun] == 1 and finished_run[firstRun] == 1:
  firstRun += 1
  if firstRun == numRuns:
    break

#Now we have the first run index
for i in range(firstRun,numRuns):
  print('~~~')
  print('Currently on Run ' + str(i))
  print('~~~')
  print()

  #Now create the directory for the runID and overwrite the started flag
  runPath = seedPath + 'run_' + str(i)
  runName = 'run_' + str(i)
  command = ['mkdir', runPath]
  subprocess.run(command)
  print('Created directory at ' + runPath)

  with open(seedPath + 'check_run.txt', 'r') as filehandle:
    check_run_file = filehandle.readlines()

  overWTemp = check_run_file.copy()
  overWTemp[i*numChecks + 1] = str(1) + '\n' 
  with open(seedPath + 'check_run.txt', 'w') as filehandle:
    filehandle.writelines(overWTemp)
  

  
  #Now set the main train command
  print('Starting training for ' + str(epoch_list[i]) + ' epochs')
  #To train on CPUs:
#  command = ['python', 'train.py', '--dataroot', './path/to/data/', '--name', runName, \
#             '--model', 'pix2pix', '--batch_size=8', '--direction', 'AtoB', '--n_epochs='+str(epoch_list[i]), \
#             '--n_epochs_decay='+str(epochDecay_list[i]), '--gan_mode='+str(gan_list[i]), '--lr='+str(lr_list[i]), \
#             '--beta1='+str(beta_list[i]), '--lr_policy='+str(policy_list[i]), '--gpu_ids', '-1']

   #To train on GPUs:
#   python train.py --dataroot path/to/data --name run_test --model pix2pix --batch_size=8 --direction AtoB
             
  command = ['python', 'train.py', '--dataroot', 'path/to/data/', '--name', runName, \
              '--model', 'pix2pix', '--batch_size=8', '--direction', 'AtoB', '--n_epochs='+str(epoch_list[i]), \
              '--n_epochs_decay='+str(epochDecay_list[i]), '--gan_mode='+str(gan_list[i]), '--lr='+str(lr_list[i]), \
              '--beta1='+str(beta_list[i]), '--lr_policy='+str(policy_list[i]),\
              '--save_epoch_freq=10', '--display_id', '0']

  subprocess.run(command)


  #Once done, copy over the checkpoints directory into my own checkpoints directory
  shutil.copytree('checkpoints/' + runName, runPath + '/checkpoints')
  command = ['rm', '-r', 'checkpoints/' + runName]
  subprocess.run(command)

  #Also copy over the loss file, remove the created loss folder bc of my dumb script originally
  shutil.copytree('ershov_lossFolder_0', runPath + '/loss')
  command = ['rm', '-r', 'ershov_lossFolder_0']
  subprocess.run(command)
#  !rm -r ershov_lossFolder_0

  #Finally, set the finished running to 1
  with open(seedPath + 'check_run.txt', 'r') as filehandle:
    check_run_file = filehandle.readlines()
  
  overWTemp = check_run_file.copy()
  overWTemp[i*numChecks + 2] = str(1) + '\n' 
  with open(seedPath + 'check_run.txt', 'w') as filehandle:
    filehandle.writelines(overWTemp)
  print('Finished run ' + str(i))


#'python train.py --dataroot ./path/to/data/ --name testest --model pix2pix --batch_size=8 --direction AtoB ')
  





#with open(seedPath + 'check_run.txt', 'r') as filehandle:
#    check_run_file = filehandle.readlines()
#
#overWTemp = check_run_file.copy()
#overWTemp[14*numChecks + 2] = str(1) + '\n' 
#overWTemp[14*numChecks + 1] = str(1) + '\n' 
#overWTemp[15*numChecks + 2] = str(0) + '\n' 
#overWTemp[15*numChecks + 1] = str(0) + '\n' 
#overWTemp[16*numChecks + 2] = str(0) + '\n' 
#overWTemp[16*numChecks + 1] = str(0) + '\n' 
#overWTemp[17*numChecks + 2] = str(0) + '\n' 
#overWTemp[17*numChecks + 1] = str(0) + '\n' 
#overWTemp[18*numChecks + 2] = str(0) + '\n' 
#overWTemp[18*numChecks + 1] = str(0) + '\n' 
#overWTemp[19*numChecks + 2] = str(0) + '\n' 
#overWTemp[19*numChecks + 1] = str(0) + '\n' 
#with open(seedPath + 'check_run.txt', 'w') as filehandle:
#  filehandle.writelines(overWTemp)
#print('Finished run ' + str(0))