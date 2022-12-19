The tutorial on GitHub
1. Installation and data preparation
Download the code.
Windows or Linux platform could both run. Firstly, you should install Anaconda
We tested on the NVIDIA GeForce GTX 1080 Ti and NVIDIA GeForce GTX 2080 Ti. The version of cuda must be 10.1 and cudnn is 7.6.5. So the version of the graphics card should be compatible with cuda10.1.
conda create -y -n rest -c conda-forge python=3.7
conda activate rest
pip install -r requirements.txt
2. How to generate the training data:
Strategy1：
1). Do STA use Relion or other software
2). The subtomograms that participated in STA were extracted using Relion as the input of the training data. 
3). According to alignment parameters in star files. used e2proc3d.py program to rotate and shift the averaged map to generate the ground truth. The command is like 
‘e2proc3d.py --rot=spider:phi=-64:theta=28:psi=62 --trans=6.2,0.4,28.6ribosome.mrc y_0.mrc’
Here, ‘--rot=spider:phi=-64:theta=28:psi=62 --trans=6.2,0.4,28.6’ is the parameters of orientation which has been calculated from Relion star file, ‘ribosome.mrc’is the averaged map from STA and ‘y_0.mrc’ is the ground truth of the particle according to the parameter.
4). Generate the training pairs of all particles and train them later.
Strategy2：
You can directly use the pipeline of HEMNMA_3D in Scipion to generate the simulated subtomograms as the input and the volume used to generate the subtomograms as ground truth. 
1). You need to import a PDB for the task of NMA.
2). Using the program of ‘Modesanalysis&visualization’ to generate the Normal modes
3). Using the program ‘Synthesize volumes’ to generate the simulated data, in this program, you can set the parameters of volume number, voxel size, voxel size, SNR, tilt range, SNR, and settings in CTF.
4). Extract the ground truth in the output named ‘*_df.vol’ and use e2proc3d.py program to convert the file format from .vol to .mrc.
5) Extract the simulated data in the output named ‘*_subtomogram.vol’and use e2proc3d.py program to convert the file format from .vol to .mrc.
3. Training
Copy all files downloaded here in the project folder and execute them in the project folder. If you change a new batch of data for training, do it again. 
1). Rename the raw particles or simulated particles to 1_00****.mrc and place them into subtomo folder. The name of each file should write in the subtomo.star.
2). Rename the ground truth to 1_00****.mrc and place them into subtomo folder in process2. The name of each file should write in the subtomo.star in process2.
3). Back to the project directory
4). Preprocess:
python process1.py;
cd process2/;
python process2.py;
cd ..;
python process3_linux.py;
python splite_trainset.py;
5). train:
python process_train.py --epoch 2 --step 1500 --num_gpu 4 --new_model_name new_model_name.h5 --initial_model initial_model_name.h5 ;
If you want to know each parameters meaning:
python process_train.py -h
usage: process_train.py [-h]
 [--initial_model INITIAL_MODEL]
[--new_model_name NEW_MODEL_NAME] 
[--epoch EPOCH] 
[--step STEP] 
[--num_gpu NUM_GPU]
refine your model using simulated data
optional arguments:
  -h, --help            show this help message and exit

  --initial_model INITIAL_MODEL  
Name of the pre-trained model. Leave empty to train a new one.

  --new_model_name NEW_MODEL_NAME
Name of the model you want to rename. Leave empty to use results/model_iter00.h5.

  --epoch EPOCH         
The epoch for training. Maybe you need find a suitable parameters to train your data

  --step STEP           
The step of each epoch. Maybe you need find a suitable parameter to train your data

  --num_gpu NUM_GPU     
The number of gpu you want to use
4. Predicting
1). Copy the input tomograms or sub-tomograms into tomoset folder
2). Execute the command : 
python process5_generate_predict.py;
python rest.py  predict for_predict1.star new_maodel_name.h5  --gpuID 0,1,2,3 --tomo_idx 0;