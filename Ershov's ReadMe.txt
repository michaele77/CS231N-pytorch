Ershov's ReadMe

This will describe how the data processing goes.

When you wish to train more models, do the following:

1) Copy the pix2pix.ipynb file and place it into the 'CS231N' file in berkeley.edu google drive
2) Open it with colab
3) Change runtime to GPU
4) Run the code
	-When you run it, it will clone a repository and will mount your drive
	-When the training is done, it will place images into the drive folder
5) Once done with the ipynb, make sure to save it, download it, and replace the current pix2pix.ipynb

If you will change any of the actual .py files or training files:

1) Implement changes locally
2) Add, commit, and push the code do your repo
	-Need to do this before you run the pix2pix.ipynb since it clones your repo everytime
	-Also need to do this if you change the training images




FOR INTERFACING WITH GCLOUD

-Assuming you are starting from an empty VM instance:

1) Copy over the gitClone.py file using: -->
gcloud compute scp gitClone.py cs231n-pytorch-project-vm:

2) Now the file should be in youe VM, open a new terminal 

3) To SSH into your VM, use: -->
gcloud beta compute ssh --zone "us-west1-b" "cs231n-pytorch-project-vm" --project "cs231n-276805"

4) Now you should see the gitClone.py file, run it with: -->
python gitClone.py
-This will clone the repo if you dont have it
-Otherwise, it will pull the repo, run this command inside the CS231N folder thats created

5) Now seed, train seed, and test all inside the repo

6) To copy files back, MAKE SURE YOU ARE IN THE LOCAL DIRECTORY,use: -->
gcloud compute scp --recurse cs231n-pytorch-project-vm:PATHTOCOPY PATHTOPASE

7) If running into an issue with DS_STORE files (ie merging photos), run: (will delete DS_Store files in all sub directories from current directory) -->
find . -name ".DS_Store" -delete

