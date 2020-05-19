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