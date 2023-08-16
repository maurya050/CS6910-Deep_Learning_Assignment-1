# CS6910:Fundamentals of Deep Learning_Assignment-1
## Feedforward Neural Network with Backpropagation
*This repository contains the implementation of a feedforward neural network and the associated backpropagation algorithm using only the numpy library. The network is designed to classify images from the Fashion-MNIST dataset into 10 different classes.*
### Introduction:
*In this project, we implement a feedforward neural network from scratch without using any automatic differentiation libraries. We train the network using the Fashion-MNIST dataset, which consists of grayscale images of clothing items belonging to 10 different classes.*
## For Running train.py file
* At line-19 : Change the wandb login key to and start the implementation.
* Project Name (wandb_project) default set to 'CS6910_Assignment_1'
* Entity Name (wandb_entity) in project (CS6910_Assignment_1) default set to 'Shivam_Maurya'

  ### String passed for selecting Function
  
  **Choices For Loss Function**
  * 'squared_loss' : For Mean Square Error
  * 'cross_entropy' : For Cross Entropy

  **Choices For Optimization Function**
  * 'sgd' : For Stochastic Gradient Descent
  * 'momentum' : For Momentum Based Gradient Descent
  * 'nag' : For Nesterov Accelerated Gradient Descent
  * 'rmsprop' : For Rmsprop
  * 'adam' : For Adam
  * 'nadam' : For Nadam

  **Choices For Activation Function**
  * 'sigmoid' : For Sigmoid Function
  * 'tanh' : For Tanh Function
  * 'relu' : For ReLu Function

  **Choices For Output-Activation Function**
  * 'softmax' : SoftmaxFunction
  
  **Choices For Weight Initilization** (String Passed For selecting Weight Initializer)
  * 'Random' 
  * 'Xavier'
  
## For Running 'CS6910_Assignment1.ipynb' File
* To run this file , change  the wandb login key and just run the file from the start .
* Run from start sequencially, it will give all the outputs which i am getting in my wandb Report.
