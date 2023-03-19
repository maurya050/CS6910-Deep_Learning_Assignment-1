# CS6910:Fundamentals of Deep Learning_Assignment-1
**Assignment_1 submission for the course Fundamentals of Deep Learning (CS6910).**
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
