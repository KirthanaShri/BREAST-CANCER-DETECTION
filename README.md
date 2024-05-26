## Breast Cancer Classification

In this project, we compare the performance of multiple deep learning architectures, including MLP, CNN-based ResNet with the BreakHis dataset, which contains breat cancer histopathological images at varying magnification levels. 

## Setup
Install Pytorch and other dependencies:

	pip install -r requirements.txt


Copy dataset in the folder `data_model`.
The struture is shown as below

![alt text](https://raw.githubusercontent.com/khanhvynguyen/Breast_Cancer_NN_Project/main/pics/dataset.png?token=GHSAT0AAAAAACDIHFIJUAXM7CWX7MAO6QX4ZLIEMEA)


## Usage

Run the following command to train and evaluate the model:

	python main.py


## Structure of files

- utils.py, consisting of these following functions:
	+ `create_dataset_csv()`: create a csv file containing metadata of the dataset
	+ `compute_accuracy()`: compute accuracy of 1 batch
	+ `find_images()`: get all image paths recursively from a folder (used in `create_dataset_csv()`)
	+ `get_mean_std()`: TODO: compute mean and std of the training set (not done yet)
	+ `get_dataloaders()`: create trainloader, validloader, testloader
	
- dataset.py: implementation for class `BreastCancer`. 
We create dataset objects (trainset, validset, testset) from this class. Then, we use these objects to create trainloader, validloader and testloader using DataLoader (DataLoader is used to load the data in a format that can be easily fed into a model for training and valuation).

- train.py:
    + `train_one_epoch()`: train 1 epoch on trainloader, return train_acc by magnification and train_loss of 1 epoch

- eval.py:
    + `eval_model()`: evaluate model, return eval_acc by magnification and eval_loss of 1 epoch
    
- main.py: 
    + `main()`: train and evaluate model over each epoch. The result will show the table summary of training and evaluation accuracy and loss by magnification as well as the line chart of these metrics
