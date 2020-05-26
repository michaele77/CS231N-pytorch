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
from distutils.dir_util import copy_tree
import cv2
import numpy as np


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
  
  
#Image display function
def tileCompFunc(inDict, runNameList, imgNum, subpath, preStr='val', save=False):
  rowSize = 5
  colSize = 5
  fig, axs = plt.subplots(rowSize, colSize, figsize=(15,15))
  
  stCol = 1
  stRow = 0
  
  #Get the image
  
  imPath = subpath + '/' + 'run_0' + '/saved_results/model_' + str(chosenModel) + '/' + preStr + '/'
  skPath = imPath + inDict[runNameList[0]][0][imgNum]
  bwPath = imPath+ inDict[runNameList[0]][1][imgNum]
  genPath = imPath + inDict[runNameList[0]][2][imgNum]
  
  skData = cv2.imread(skPath, 0)
  bwData = cv2.imread(bwPath, 0)
  
  #Set the top two as the original figures
  axs[0,0].imshow(skData, cmap='gray') #Sketch
  axs[0, 0].set_title('Sketch for ' + str(imgNum), fontsize = 12)
  
  axs[0,colSize-1].imshow(bwData, cmap='gray') #B&W
  axs[0, colSize-1].set_title('B&W for ' + str(imgNum), fontsize = 12)
  
  
  for runI in runNameList:
      
      imPath = subpath + '/' + runI + '/saved_results/model_' + str(chosenModel) + '/' + preStr + '/'
  
      
      genPath = imPath + inDict[runI][-1][imgNum]
      genData = cv2.imread(genPath, 0)
      
      axs[stCol, stRow].imshow(genData, cmap='gray')
      axs[stCol, stRow].set_title(runI, fontsize = 12)
      
      stRow += 1
      stCol += 1*(stRow % rowSize == 0)
      
      stRow = stRow % rowSize
      
  iterAx = axs.flatten()
  for ax in iterAx:
      ax.set_xticks([])
      ax.set_yticks([])
      
  
  if save:
      plt.savefig(resultPath + '/M' + str(chosenModel) + '_' + \
                  preStr + '_' + str(imgNum) + '.png')
      
      
def calcHeuristic(inDict, runID, imgNum, subpath, preStr='val'):
    imPath = subpath + '/' + runID + '/saved_results/model_' + str(chosenModel) + '/' + preStr + '/'
    skPath = imPath + inDict[runID][0][imgNum]
    bwPath = imPath+ inDict[runID][1][imgNum]
    genPath = imPath + inDict[runID][2][imgNum]
    
    
    skData = np.float32(cv2.imread(skPath, 0))
    bwData = np.float32(cv2.imread(bwPath, 0))
    genData = np.float32(cv2.imread(genPath, 0))
    
    
    
    heurVal = 1 - np.sum(np.abs(bwData - genData)) / np.sum(np.abs(bwData - skData))
    return heurVal
    
      
  
          
   
    

#First, set the seed directory and make directories if they dont exist
#subproc will return 0 is successful, 1 if exists
####INPUT HERE####
seedNum = 1
####INPUT HERE####
subSeedPath = 'automated_tests_gcp/seed_' + str(seedNum)
seedPath = 'automated_tests_gcp/seed_' + str(seedNum) + '/test_result'
checkPath = 'checkpoints/'
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
chosenModel = 30
valNum = 200
testNum = 200
####INPUT HERE####

modelStr = str(chosenModel) + '_net_G.pth'


###IMPLEMENTING INCEPTION SCORE CALCULATION HERE###
#First, use chosenModel to go through runDirList 
#Create checkpoint files for all things
#Run the test on all of the files inside using both validation and testing 
#Then, go inside the results folder, load each image, calculate inception score, and store in a list
#Then make some plots based on dimensionality and result

#First, remove the directory in checkpoints if it exists
#Then, copy desired model to the checkpoints as "latest_blahblah.pth"
#Then run the script and see the output in results

imgDictVal = {} 
imgDictTest = {} 

usrInRerun = input('Do you want rerun test/val image output [1 for yes]: ')

usrInDir = input('Do you want to delete the old directories if they pop up \
              and replace them with the new images? [1 for yes]: ')

for runPathIter in runDirList:
    
    #destroy/create new directory
    path = checkPath + runPathIter
    command = ['rm', '-r', path]
    subprocess.run(command)
    #again, returns 1 if unsuccessful and 0 if ok
    
    command = ['mkdir', path]
    subprocess.run(command)
    
    
    #copy over the desired model as latest
    srcPath = subSeedPath + '/' + runPathIter + '/checkpoints/' + modelStr
    sinkPath = 'checkpoints/' + runPathIter + '/latest_net_G.pth'
    print('Copying over from ' + runPathIter)
    shutil.copy2(srcPath,sinkPath)
    
    
    #Now, run the both the test and validation sets
    #For now, do 200 of each (remember, validation has like 700 and is different, test is from training)
    if int(usrInRerun) == 1 and runPathIter == 'run_0':
    
        command = ['rm', '-r', 'results/' + runPathIter]
        
        print('Running validation phase for ' + runPathIter + '...')
        subprocess.run(command)
        command = ['python', 'test.py', '--dataroot', 'path/to/data', '--name', runPathIter, \
                   '--model', 'pix2pix', '--direction', 'AtoB', '--gpu_ids', 
                   '-1', '--num_test='+str(valNum), '--phase','val']
        subprocess.run(command)
        
        
        
        
        print('Running testing phase for ' + runPathIter + '...')
        command = ['python', 'test.py', '--dataroot', 'path/to/data', '--name', runPathIter, \
                   '--model', 'pix2pix', '--direction', 'AtoB', '--gpu_ids', 
                   '-1', '--num_test='+str(testNum), '--phase','test']
        subprocess.run(command)
    
    
    #Now copy and parse over the files; copy their contents into lists to process
    testrunPath = subSeedPath + '/' + runPathIter + '/saved_results/model_' + str(chosenModel)
    command = ['mkdir', subSeedPath + '/' + runPathIter + '/saved_results']
    temp = subprocess.run(command)
    
    command = ['mkdir', testrunPath]
    temp = subprocess.run(command)
    
    if temp.returncode == 0:
        print('Created directory at ' + testrunPath)
    elif temp.returncode == 1:
        if int(usrInDir) == 1:
            command = ['rm', '-r', testrunPath]
            subprocess.run(command)
            
            command = ['mkdir', testrunPath]
            subprocess.run(command)
        else:
            print('Using old directory (not rewriting images)...')
            
    
    if temp.returncode == 0 or (temp.returncode == 1 and int(usrInDir) == 1):
        #First remove the DS_store file so we dont get any 
        srcPath = 'results/' + runPathIter + '/test_latest/images/'
        sinkPath = testrunPath + '/test'
        
#        command = ['rm', srcPath + '.DS_Store']
#        subprocess.run(command)
        
        shutil.copytree(srcPath,sinkPath)
        
        print('Test directory copied!')
        
        
        srcPath = 'results/' + runPathIter + '/val_latest/images/'
        sinkPath = testrunPath + '/val'
        
#        command = ['rm', srcPath + '.DS_Store']
#        subprocess.run(command)
        
        shutil.copytree(srcPath,sinkPath)
        
        print('Val directory copied!')
        
        
    
    #Now run extraction script to read individual files and store inside lists
    #Structure is as follows:
    #imgDict is a dictionary containing a list
    #list is [sketch img, black and white image, generated image]
    #each listentry is a list of the names of the images
    #Thus, for a given run, we can query the dictionary, and iterate through one of the lists
    path = testrunPath + '/val'
    dirList = os.listdir(path)
    dirList.sort()
    try:
        dirList.remove('.DS_Store')
    except:
        print('os_dir wasnt in the dir')
    
    #When sorted, order is: xxx_fake_B.png, xxx_real_A.png, xxx_real_B.png, etc
    skL = []
    bwL = []
    genL = []
    for i in range(len(dirList)//3):
        skL.append(dirList[i*3+1])
        bwL.append(dirList[i*3+2])
        genL.append(dirList[i*3])
        
    imgDictVal[runPathIter] = [skL,bwL,genL]
        
        
    path = testrunPath + '/test'
    dirList = os.listdir(path)
    dirList.sort()
    try:
        dirList.remove('.DS_Store')
    except:
        print('os_dir wasnt in the dir')
    
    #When sorted, order is: xxx_fake_B.png, xxx_real_A.png, xxx_real_B.png, etc
    skL = []
    bwL = []
    genL = []
    for i in range(len(dirList)//3):
        skL.append(dirList[i*3+1])
        bwL.append(dirList[i*3+2])
        genL.append(dirList[i*3])
        
    imgDictTest[runPathIter] = [skL,bwL,genL]
    
    
    
##FINISHED WITH RUN DIRECTORY LOOP!##
    
####NOW, CALCULATE IS SCORES!!###
    ##COME BACK TO THIS, DONT KNOW HOW TO DO INCEPTION SCORE!!!!
#ISDictVal = {} #creating a dictionary for the inception scores
#ISDictTest = {}
#for runD in runDirList:
#    valPath = testrunPath + '/val/'
#    
#    currList = imgDictVal[runD]
#    skPath = currList[0]
#    bwPath = currList[1]
#    genPath = currList[2]
#    
#    for i in range(len(skPath)):
#        skIm = cv2.imread(valPath + skPath, 0)
#        bwIm = cv2.imread(valPath + bwPath, 0)
#        genIm = cv2.imread(valPath + genPath, 0)
    
    
##NOW IMPLEMENT IMAGE CONCTENATION##
#Make sure to replace BOTH preStr variables and the imgDict for the tiling function!
dispImList = [7,69,100,138,77,1,89,193]

ifRun = input('Do you want to run tile images? [1 for yes]: ')

if int(ifRun) == 1:
    ifSave = input('Do you want to save the tile images? [1 for yes]: ')
    saveIn = False
    if int(ifSave) == 1:
        saveIn = True
    
    for dispIm in dispImList:
        tileCompFunc(imgDictVal, runDirList, dispIm, subSeedPath, preStr='val', save=saveIn)
    
    for dispIm in dispImList:
        tileCompFunc(imgDictTest, runDirList, dispIm, subSeedPath, preStr='test', save=saveIn)
        

##IMPLEMENT CUSTOM HEURISTIC FUNCTION BELOW##
#Test on run0
print('Calculating heuristic index...')



plt.figure(0)
for runID in runDirList:
    heurArr = []
    
    for i in range(len(imgDictVal[runID][0])):
        heurArr.append(calcHeuristic(imgDictVal, runID, i, subSeedPath, preStr='val'))
    plt.plot(heurArr)
    print('Mean for ' + runID + ' = ' + str(mean(heurArr)))


        
    
    
    
    
    
        
        
        
        
        
            
    
            



    
    

#python test.py --dataroot path/to/data --name run_0 --model pix2pix --direction AtoB --gpu_ids -1 --num_test=300 --phase val
    
    
    







    

    
    




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


