# cs6910_assignment1

Description: The repository contains the implementation of feed forward neural network, aimed to solve multiclass classification problem over fashion_mnist dataset.
Following Files are a part of Repository
1. train.py - This file contains the entire implemention of the neural network at hand
2. question1.py - This file contains the code to display all the labels with their images
3. sweep.txt: contains the sweep configuration used
4. Sample images: Contains all the sample images over all the classes

Best Configuration Achieved:

Loss: cross entropy, 
number of hidden layers: 2,
size of hidden layers: 128, 
batch size: 64, 
learning rate: 0.01,
activation: sigmoid,
weight initialization: Xavier,
optimizer: Nadam,
weight decay: 0.0001,
epochs: 10 

Results Summarized:

Train Accuracy: 89.3 
Train Loss: 0.4278
Validation Accuracy: 88.7 
Validation Loss:  0.4707
Test Accuracy: 86.92

