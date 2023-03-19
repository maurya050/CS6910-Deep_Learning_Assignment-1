import argparse
import numpy as np
import pandas as pd
from keras.datasets import mnist
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from tensorflow.keras.datasets.fashion_mnist import load_data
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import wandb
import warnings
warnings.filterwarnings("ignore")
wandb.login(key='b032cc059132c9aac4f1b317f6f9ad007ef9e4d4')
wandb.init(project="CS6910_Assignment")
parser = argparse.ArgumentParser()
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='myprojectname')
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='myname')
parser.add_argument('-d', '--dataset', help='choices: ["mnist", "fashion_mnist"]', type=str, default='fashion_mnist')
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=1)
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=4)
parser.add_argument('-l','--loss', help = 'hoices: ["mean_squared_error", "cross_entropy"]' , type=str, default='cross_entropy')
parser.add_argument('-o', '--optimizer', help = 'choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]', type=str, default = 'sgd')
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.1)
parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.5)
parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.5)
parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.5)
parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=0.000001)
parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=.0)
parser.add_argument('-w_i', '--weight_init', help = 'choices: ["Random", "Xavier"]', type=str, default='Random')
parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=1)
parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', nargs='+', type=int, default=4, required=False)
parser.add_argument('-a', '--activation', help='choices: ["identity", "sigmoid", "tanh", "ReLU"]', type=str, default='sigmoid')
# parser.add_argument('--hlayer_size', type=int, default=32)
parser.add_argument('-oa', '--output_activation', help = 'choices: ["softmax"]', type=str, default='softmax')
# parser.add_argument('-oc', '--output_size', help ='Number of neurons in output layer used in feedforward neural network.', type = int, default = 10)
arguments = parser.parse_args()

#Load the fashion MNIST data 
if(arguments.dataset=="fashion_mnist"):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    enc = OneHotEncoder()
    x_train = x_train/255.0
    x_test = x_test/255.0
    y_OH_test = enc.fit_transform(np.expand_dims(y_test, 1)).toarray()
    y_OH_train = enc.fit_transform(np.expand_dims(y_train, 1)).toarray()
    print(y_OH_train.shape, y_OH_test.shape)
    print(x_train.shape)
else:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    enc = OneHotEncoder()
    x_train = x_train/255.0
    x_test = x_test/255.0
    y_OH_test = enc.fit_transform(np.expand_dims(y_test, 1)).toarray()
    y_OH_train = enc.fit_transform(np.expand_dims(y_train, 1)).toarray()
    print(y_OH_train.shape, y_OH_test.shape)
    print(x_train.shape)

class FFNN():
 
    def __init__(self,opt, hidden_sizes, loss_fun, activation_fun, output_act, b_s= 1024, epochs = 10, initialization = "Random", log=0, train_losses = None, train_accuracy = None, test_losses = None, test_accuracy = None):
        
        self.train_accuracy_list = train_accuracy
        self.batch_size = b_s
        self.layer_sizes = []
        self.output_activation = output_act
        self.dw = {}
        self.output_layer_size=1
        self.input_layer_size=0
        self.A = {}
        self.hidden_layer_sizes = hidden_sizes
        self.dH = {}
        self.loss_function = loss_fun
        self.H = {}
        self.train_losses_list = train_losses
        self.optimizer = opt
        self.dA = {}
        self.activation_function = activation_fun
        self.weights = {}
        self.db = {}
        self.biases = {}
        #Layer sizes array will be initialzed after input and output layer size is obtained
        self.val_accuracy_list = test_accuracy
        self.val_losses_list = test_losses
        self.wan_log = log
        self.epochs = epochs
        self.initialization = initialization


    def initialize_weights(self):
      
        #Function to initialize the weights between the layers of the network. Weights are initialized randomly.
        self.layer_sizes = [self.input_layer_size] + self.hidden_layer_sizes + [self.output_layer_size]
        np.random.seed(137)
        np.random.RandomState(137)
        ln = len(self.hidden_layer_sizes)
        self.optimizer.initialize(self.layer_sizes)
        weight_counts =ln  +1
        for i in range(weight_counts):
            
            s_size = (self.layer_sizes[i], self.layer_sizes[i+1])
            lsize_i = self.layer_sizes[i]
            lsize_i1 = self.layer_sizes[i+1]
        
            if self.initialization == "Xavier":
               
                self.weights[i+1] = np.random.normal(0.0, np.sqrt(2 / float(lsize_i + lsize_i1)) , size = s_size)

            elif self.initialization == "Random":
               
                self.weights[i+1] = np.random.randn(lsize_i, lsize_i1)
            self.biases[i+1] = np.zeros((1, lsize_i1))


    def forward_propogation(self, X):
        
        ln = len(self.hidden_layer_sizes)
        
        self.H = {}
        #Initialize the output from input layer as H[0] into a single row(row vector)
        self.H[0] = X.reshape(1,-1)
        self.A = {}
        #Compute a(x) = W_x*h(x-1) and h(a(x)) = activation(a(x)) for hidden layer
        for i in range(ln):
            weight = self.weights[i+1]
            bias = self.biases[i+1]
            self.A[i+1] = np.matmul(self.H[i], weight) + bias
            self.H[i+1] = self.activation_function.compute_activation(self.A[i+1])

        ln = ln +1
        weight = self.weights[ln]
        bias = self.biases[ln]
        self.A[ln] = np.matmul(self.H[ln-1], weight) + bias
        #Compute a(x) and h(a(x)) = softmax(a(x)) for output layer
        self.H[ln] = self.output_activation.compute_activation(self.A[ln]) 
        return  self.H[ln]

    def backward_propogation(self, X, Y,  dw_i, db_i):
        
        ln = len(self.hidden_layer_sizes)
        #Compute the gradient of loss wrt the activation of output layer
        self.dA[ln +1] = self.loss_function.last_output_derivative(self.H[ln +1], Y, self.output_activation.compute_derivative(self.A[ln +1]))

        #Compute the partial derivatives for the weights and biases of the layers
        # change in weight : (assuming 2 hidden layers) dL/dW3 = dA3/dW3 * dL/dA3 = d(W3*H2 + B3)/dW3 * dL/dA3 = H2 * dL/dA3
        # change in biase : (assuming 2 hidden layers) dL/dB3 = dA3/dB3 * dL/dA3 = d(W3*H2 + B3)/dB3 * dL/dA3 = 1 * dL/dA3
        for i in range(ln, -1, -1):
            weight = self.weights[i+1].T
            db_i[i+1] = self.dA[i+1]
            dw_i[i+1] = np.matmul(self.H[i].T, self.dA[i+1])
            
            if i!=0:
                self.dH[i] = np.matmul(self.dA[i+1],weight)  
                act_grad = self.activation_function.compute_derivative(self.A[i])
                self.dA[i] = np.multiply(act_grad, self.dH[i])    
               
        return dw_i, db_i


    def fit(self, X, Y, X_val, Y_val, console = 1):
        
        #Function to fit the data (X,Y) on the model. This performs forward + backward pass for epoch number of times. Gradient is updated after each batch is processed.
        self.output_layer_size = Y.shape[1] # Number of columns in output (label count)
        self.console_log = console
        self.input_layer_size = X.shape[1]*X.shape[1] # Number of features in data(features)
        hidden_ls = self.hidden_layer_sizes
        self.initialize_weights()
        ln = len(self.hidden_layer_sizes)+1
        for e in range(self.epochs):
            count = -1
            for i in range(ln):
                layer_size = self.layer_sizes[i+1]
                self.db[i+1] = np.zeros((1, layer_size))
                self.dw[i+1] = np.zeros((self.layer_sizes[i], layer_size))
                
            y_preds = []

            for x, y in zip(X, Y):
                count += 1
                db_i = {}
                dw_i = {}

                if count==self.batch_size:
                    #Done wih current batch
        
                    if self.optimizer.optimizer_name()=="nag":
                        b_lookahead = {}
                        w_lookahead = {}
                        
                        for i in range(ln):
                            b_upd = self.optimizer.gamma*self.optimizer.update_history_b[i+1]
                            w_upd = self.optimizer.gamma*self.optimizer.update_history_w[i+1]
                            b_lookahead[i+1] = self.biases[i+1] - b_upd
                            w_lookahead[i+1] = self.weights[i+1] - w_upd
                            
                        
                        weights_old = self.weights
                        self.weights = w_lookahead
                        self.forward_propogation(x)
                        dw_lookahead, db_lookahead = self.backward_propogation(x,y, dw_i, db_i) 
                        self.biases = b_lookahead
                        biases_old = self.biases
                        self.weights, self.biases = self.optimizer.update_parameters(weights_old, biases_old, dw_lookahead, db_lookahead, hidden_ls)

                    else: 
                        self.weights, self.biases = self.optimizer.update_parameters(self.weights, self.biases, self.dw, self.db, hidden_ls)
                    
                    for i in range(ln):
                        lsize1 = self.layer_sizes[i+1]
                        lsize = self.layer_sizes[i]
                        self.db[i+1] = np.zeros((1, lsize1))
                        self.dw[i+1] = np.zeros((lsize, lsize1))

                    count = 0
                #Forward Propogation
                self.forward_propogation(x)

                #Predictions
                y_preds.append(self.H[ln])

                #Backward Propogation using Loss funtion
                self.backward_propogation(x,y, dw_i, db_i) 

                for i in range(ln):
                    b_i = db_i[i+1]
                    w_i = dw_i[i+1]
                    self.db[i+1] = self.db[i+1] + b_i
                    self.dw[i+1] = self.dw[i+1] + w_i
                    
           
            #Update weights based on loss(GD hence once every epoch update)
            if self.optimizer.optimizer_name()=="nag":
                b_lookahead = {}
                w_lookahead = {}
                
                for i in range(ln):
                    bias = self.biases[i+1]
                    weight = self.weights[i+1]
                    b_upd = self.optimizer.gamma*self.optimizer.update_history_b[i+1]
                    w_upd = self.optimizer.gamma*self.optimizer.update_history_w[i+1]
                    b_lookahead[i+1] = bias - b_upd
                    w_lookahead[i+1] = weight - w_upd     
                
                biases_old, weights_old= self.biases, self.weights
                self.biases, self.weights = b_lookahead , w_lookahead
                
                self.forward_propogation(x)
                dw_lookahead, db_lookahead = self.backward_propogation(x, y, dw_i, db_i) 
                self.weights, self.biases = self.optimizer.update_parameters(weights_old, biases_old, dw_lookahead, db_lookahead, hidden_ls)

            else:  
                self.weights, self.biases = self.optimizer.update_parameters(self.weights, self.biases, self.dw, self.db, hidden_ls)
            
            y_preds = np.array(y_preds).squeeze()
            y_preds_validation = self.predict(X_val)
            
            validation_loss = self.loss_function.compute_loss(Y_val, y_preds_validation, self.batch_size)
            if self.val_losses_list != None:
                self.val_losses_list.append(validation_loss) 
            
            training_loss = self.loss_function.compute_loss(Y, y_preds, self.batch_size)
            if self.train_losses_list != None:
                self.train_losses_list.append(training_loss)
                
            validation_accuracy = accuracy_score(np.argmax(Y_val,1), np.argmax(y_preds_validation,1))
            if self.val_accuracy_list != None:
                self.val_accuracy_list.append(validation_accuracy)
                
            training_accuracy = accuracy_score(np.argmax(Y,1), np.argmax(y_preds,1))
            if self.train_accuracy_list != None:
                self.train_accuracy_list.append(training_accuracy)
           
              
            if self.console_log == 1: #For Printing Log results on Console 
                print("Training Loss: ",round(training_loss,3),"Val_loss:", round(validation_loss, 3), " Training Accuracy: ",round(training_accuracy,3), "Val_accuracy:", round(validation_accuracy,3), " <-:Epoch:",e+1,)
            
            
            elif self.wan_log==1:#For Log metrics on wandb
                
                wandb.log({"Training_accuracy": training_accuracy, "Validation_accuracy": validation_accuracy, "Training_loss": training_loss, "Validation_loss": validation_loss, 'Epoch': e+1})

    
        return training_loss, validation_loss, training_accuracy, validation_accuracy

    def predict(self, X):
        y_pred = []
        for x in X:
            pred = self.forward_propogation(x)
            y_pred.append(pred)

        y_pred = np.array(y_pred).squeeze()
        return y_pred

class Optimizer():
    
    def __init__(self, optimizer,  learning_rate = 0.001,  gamma = 0.001, beta1 = 0.9, beta2 = 0.999, weight_decay = 0.0, epsilon = 1e-8):
        
        self.update_history_b = {}
        self.m_b = {}
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.db_look_ahead = {}
        self.b_look_ahead = {}
        self.weight_decay = weight_decay
        self.v_b = {}
        self.dw_look_ahead ={}
        self.gamma = gamma
        self.w_look_ahead = {}
        self.update_history_w = {}
        self.lr =0
        self.epsilon = epsilon
        self.m_w = {}
        self.v_w ={}
        if( self.optimizer == "sgd"):
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            
        if( self.optimizer == "momentum" ):
            self.weight_decay = weight_decay
            self.initialized = False
            self.update_history_w = {}
            self.learning_rate = learning_rate
            self.update_history_b = {}
            self.gamma = gamma
        
        if( self.optimizer == "nag"):
            self.initialized = False
            self.update_history_b = {}
            self.update_history_w = {}
            self.learning_rate = learning_rate
            self.b_look_ahead={}
            self.w_look_ahead={}
            self.gamma = gamma
            self.db_look_ahead={}
            self.dw_look_ahead={}

        if( self.optimizer == "rmsprop"):
            self.initialized = False
            self.weight_decay = weight_decay
            self.v_b = {}
            self.v_w = {}
            self.epsilon = epsilon
            self.gamma = gamma
            self.learning_rate = learning_rate
            
            

        if( self.optimizer == "adam" or self.optimizer == "nadam"):
            self.initialized = False
            self.epsilon = epsilon
            self.v_b = {}
            self.v_w = {}
            self.beta1 = beta1
            self.beta2 = beta2
            self.m_b = {}
            self.m_w = {}
            self.iterations = 1
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
        

    def set_learning_rate(self, learning_rate):

        if(self.optimizer == "sgd"):
            self.learning_rate = learning_rate

        if( self.optimizer == "momentum" ):
            self.learning_rate = learning_rate
        
        if( self.optimizer == "nag"):
            self.learning_rate = learning_rate

        if( self.optimizer == "rmsprop"):
            self.learning_rate = learning_rate

        if( self.optimizer == "adam" or self.optimizer == "nadam"):
            self.learning_rate = learning_rate



    def set_weight_decay(self, weight_dec):
        self.weight_decay = weight_dec
        
    def set_initial_parameters(self, parameters):
        
        if(self.optimizer == "sgd"):
            self.weight_decay = parameters["weight_decay"]
            self.learning_rate = parameters["learning_rate"]

        if( self.optimizer == "momentum" ):
            self.weight_decay = parameters["weight_decay"]
            self.gamma = parameters["gamma"]
            self.learning_rate = parameters["learning_rate"]
            
        
        if( self.optimizer == "nag"):
            self.gamma = parameters["gamma"]
            self.learning_rate = parameters["learning_rate"]

        if( self.optimizer == "rmsprop"):
            
            self.weight_decay = parameters["weight_decay"]
            self.learning_rate = parameters["learning_rate"]
            self.lr = 0.01
            self.epsilon = parameters["epsilon"]
            self.gamma = parameters["gamma"]
            

        if( self.optimizer == "adam" or self.optimizer == "nadam"):
            self.weight_decay = parameters["weight_decay"]
            self.lr = 0.01
            self.epsilon = parameters["epsilon"]
            self.beta1, self.beta2  = parameters["beta1"], parameters["beta2"]
            self.learning_rate = parameters["learning_rate"]
            
    def initialize(self, all_layers):
        
        self.b_look_ahead.clear()
        self.update_history_b.clear()
        ln = len(all_layers)
        self.m_b.clear()
        self.dw_look_ahead.clear()
        self.v_b.clear()
        self.w_look_ahead.clear()
        ln = ln -1
        self.update_history_w.clear()
        self.m_w.clear()
        self.db_look_ahead.clear()
        self.v_w.clear()
        
        if(self.optimizer == "sgd"):
            return

        if( self.optimizer == "momentum" ):
            
            for i in range(ln):
                all_layer = all_layers[i+1]
                self.update_history_b[i+1] = np.zeros((1, all_layer))
                self.update_history_w[i+1] = np.zeros((all_layers[i], all_layer))
                
        if( self.optimizer == "nag"):
            
            for i in range(ln):
                all_layer = all_layers[i+1]
                self.update_history_w[i+1] = np.zeros((all_layers[i], all_layer))
                self.dw_look_ahead[i+1] = np.zeros((all_layers[i], all_layer))
                self.w_look_ahead[i+1] = np.zeros((all_layers[i], all_layer))
                self.update_history_b[i+1] = np.zeros((1, all_layer))
                self.db_look_ahead[i+1] = np.zeros((1, all_layer))
                self.b_look_ahead[i+1] = np.zeros((1, all_layer))

        if( self.optimizer == "rmsprop"):
            
            for i in range(ln):
                all_layer = all_layers[i+1]
                self.v_w[i+1] = np.zeros((all_layers[i], all_layer))
                self.v_b[i+1] = np.zeros((1, all_layer))

        if( self.optimizer == "adam" or self.optimizer == "nadam"):
            
            for i in range(ln):
                all_layer = all_layers[i+1]
                self.m_b[i+1] = np.zeros((1, all_layer))
                self.v_b[i+1] = np.zeros((1, all_layer))
                self.m_w[i+1] = np.zeros((all_layers[i], all_layer))
                self.v_w[i+1] = np.zeros((all_layers[i], all_layer))
                
    def optimizer_name(self):

        if(self.optimizer == "sgd"):
            return "sgd"

        if( self.optimizer == "momentum" ):
            return "momentum"
        
        if( self.optimizer == "nag"):
            return "nag"

        if( self.optimizer == "rmsprop"):
            return "rmsprop"

        if( self.optimizer == "adam"):
            return "adam"
        if( self.optimizer == "nadam"):
            return "nadam"

    def update_parameters(self, weights, biases, dw, db, layers):
        nlayer = len(layers)+1
        if(self.optimizer == "sgd"):
            for i in range(nlayer):
                decay_wt = self.weight_decay*weights[i+1]
                dw[i+1] = dw[i+1] + decay_wt
                grad_b = self.learning_rate * db[i+1]
                grad_w = self.learning_rate * dw[i+1]
                biases[i+1] = biases[i+1] - grad_b # b_t = b_{t-1} - eta*(dL/db)
                weights[i+1] = weights[i+1] - grad_w # w_t = w_{t-1} - eta*(dL/dw)
            
            return weights, biases

        if( self.optimizer == "momentum" ):
            """
            Function to perform the weight update step based on optimizer algorithm
            """

            for i in range(nlayer):
                
                wt_decay = self.weight_decay*weights[i+1]
                dw[i+1] = dw[i+1] + wt_decay
                upd_bias = self.learning_rate*db[i+1]
                upd_weight = self.learning_rate*dw[i+1]
                self.update_history_b[i+1] =self.gamma*self.update_history_b[i+1] + upd_bias 
                self.update_history_w[i+1] =self.gamma*self.update_history_w[i+1] + upd_weight
                biases[i+1] = biases[i+1] - self.update_history_b[i+1]
                weights[i+1] = weights[i+1] - self.update_history_w[i+1]
            
            return weights, biases
        
        if( self.optimizer == "nag"):
            """
            Function to perform the weight update step based on optimizer algorithm
            """
            for i in range(nlayer):
                
                grad_db = self.learning_rate*db[i+1]
                grad_dw = self.learning_rate*dw[i+1]
                self.update_history_b[i+1] = self.gamma*self.update_history_b[i+1] 
                self.update_history_b[i+1] += grad_db
                biases[i+1] = biases[i+1] - self.update_history_b[i+1]
                self.update_history_w[i+1] = self.gamma*self.update_history_w[i+1] 
                self.update_history_w[i+1] += grad_dw
                weights[i+1] = weights[i+1] - self.update_history_w[i+1]
                

            return weights, biases

        if( self.optimizer == "rmsprop"):
            for i in range(nlayer):
                eps = self.epsilon
                lr = self.learning_rate
                wt_decay = self.weight_decay*weights[i+1]
                dw[i+1] = dw[i+1] + wt_decay
                
                v_weight = self.gamma*self.v_w[i+1] 
                self.v_w[i+1] = v_weight + (1-self.gamma)* ((dw[i+1])**2)
                
                v_bias = self.gamma*self.v_b[i+1]
                self.v_b[i+1] = v_bias + (1-self.gamma)* ((db[i+1])**2)
                
                biase_upd = ((lr)/np.sqrt(self.v_b[i+1] + eps))*db[i+1]
                weight_upd = ((lr)/np.sqrt(self.v_w[i+1] + eps))*dw[i+1]

                biases[i+1] = biases[i+1] - biase_upd
                weights[i+1] = weights[i+1] - weight_upd
            
            return weights, biases

        if( self.optimizer == "adam" or self.optimizer == "nadam"):
            #Function to perform the weight update step based on optimizer algorithm
           
            for i in range(nlayer):
                eps = self.epsilon
                lr = self.learning_rate
                upd_wt = self.weight_decay*weights[i+1]
                dw[i+1] = dw[i+1] + upd_wt

                m_bias = self.beta1*self.m_b[i+1] 
                self.m_b[i+1] = m_bias + (1-self.beta1)* (db[i+1])
                m_weight = self.beta1*self.m_w[i+1] 
                self.m_w[i+1] = m_weight + (1-self.beta1)* (dw[i+1])
                v_weight = self.beta2*self.v_w[i+1] 
                self.v_w[i+1] = v_weight + (1-self.beta2)* ((dw[i+1])**2)
                v_bias = self.beta2*self.v_b[i+1] 
                self.v_b[i+1] = v_bias + (1-self.beta2)* ((db[i+1])**2)
                
                v_hat_div = (1-(self.beta2**self.iterations))
                m_hat_div = (1-(self.beta1**self.iterations))
                v_b_hat = self.v_b[i+1] / v_hat_div
                m_b_hat = self.m_b[i+1] / m_hat_div
                v_w_hat = self.v_w[i+1] / v_hat_div
                m_w_hat = self.m_w[i+1] / m_hat_div
                nadam_impl = (1-self.beta1)/(1-(self.beta1**self.iterations))
                if(self.optimizer == "nadam"):
                    weight_upd = ((lr)/(np.sqrt(v_w_hat) + eps))*(self.beta1 * m_w_hat + (nadam_impl * dw[i+1]))
                    weights[i+1] = weights[i+1] - weight_upd
                    biases_upd = ((lr)/(np.sqrt(v_b_hat) + eps))*(self.beta1 * m_b_hat + (nadam_impl * db[i+1]))
                    biases[i+1] = biases[i+1] - biases_upd
                elif(self.optimizer == "adam"):
                    weights_upd = ((lr)/(np.sqrt(v_w_hat) + eps))*(m_w_hat)
                    biases_upd = ((lr)/(np.sqrt(v_b_hat) + eps))*(m_b_hat)
                    
                    weights[i+1] = weights[i+1] - weights_upd
                    biases[i+1] = biases[i+1] - biases_upd
                
                nadam_impl = (1-self.beta1)/(1-(self.beta1**self.iterations))
                
            self.iterations = self.iterations + 1
            
            return weights, biases
        
class ActivationFunction():
    def __init__(self, fun = "sigmoid"):
        self.act_fun = fun

    def compute_activation(self, X):

        if(self.act_fun == "sigmoid"):
            return 1.0/(1.0+np.exp(-X)) # sigmoid function
        if(self.act_fun == "softmax"):
            exponentials = np.exp(X) # softmax function
            return exponentials / np.sum(exponentials)
        if(self.act_fun == "tanh"):
            return np.tanh(X) # tanh function
        if(self.act_fun == "relu"):
            return X * (X > 0) # relu function
            
    def compute_derivative(self, X):
        if(self.act_fun == "sigmoid"):
            val = self.compute_activation(X)
            return val*(1-val)
        if(self.act_fun == "softmax"):
            softmax = self.compute_activation(X)
            return softmax*(1-softmax)
        if(self.act_fun == "tanh"):
            return 1 - (np.tanh(X) ** 2)
        if(self.act_fun == "relu"): 
            X[X > 0.0] = 1.0
            X[X <= 0.0] = 0.0
            return X


class LossFunction():
    def __init__(self, fun = "cross_entropy"):
        self.loss_fun = fun
    
    def compute_loss(self, Y_true, Y_pred, batch_size):
        ln = len(Y_true)
        lss = 0
        if(self.loss_fun == "squared_loss"):
            return (1/2) * np.sum((Y_pred-Y_true)**2) / ln
        
        if(self.loss_fun == "cross_entropy"):
            for p in Y_pred[0]:
                if p < 10e-8  or np.isnan(p):
                    p = 10e-8
                    lss += 1
            loss = np.multiply(Y_pred,Y_true)
            loss = loss[loss != 0]
            lss  = loss
            loss = -np.log(loss)
            loss = np.mean(loss)
            return loss

    def name(self):
        if(self.loss_fun == "squared_loss"):
            return "squared_loss" 
        
        if(self.loss_fun == "cross_entropy"):
            return "cross_entropy_loss"

    def compute_derivative(self, Y_pred,Y_true):
        
        if(self.loss_fun == "squared_loss"):
            return (Y_pred)*(Y_pred-Y_true)/len(Y_true)
        
        if(self.loss_fun == "cross_entropy"):
            return -Y_true/(Y_pred)
            

    def last_output_derivative(self, Y_pred,Y_true,activation_derivative):
        ln = len(Y_true)
        lss = 0
        for p in Y_pred[0]:
                if (np.isnan(p) or p < 10e-8):
                    lss += 1
                    p = 10e-8
        if(self.loss_fun == "squared_loss"):
            return (Y_pred - Y_true)*activation_derivative / ln
        
        if(self.loss_fun == "cross_entropy"):
            return -(Y_true - Y_pred)
# Parameters Selection for Different Optimization Algorithm
parameters_sgd = {"learning_rate":arguments.learning_rate, "weight_decay":arguments.weight_decay} #sgd
parameters_momentum = {"learning_rate":arguments.learning_rate, "gamma":arguments.momentum, "weight_decay":arguments.weight_decay} #momentum
parameters_nag = {"learning_rate":arguments.learning_rate, "gamma":arguments.momentum} #nag
parameters_rmsprop = {"learning_rate":arguments.learning_rate, "gamma":arguments.momentum, "epsilon":1e-8, "weight_decay":arguments.weight_decay} #rmsprop
parameters_adam = {"learning_rate":arguments.learning_rate, "beta1":arguments.beta1, "beta2":arguments.beta2, "epsilon":arguments.epsilon, "weight_decay":arguments.weight_decay} #adam
parameters_nadam = {"learning_rate":arguments.learning_rate, "beta1":arguments.beta1, "beta2":arguments.beta2, "epsilon":arguments.epsilon, "weight_decay":arguments.weight_decay} #nadam


# "sgd" : gradient_descent, "momentum" : momentum_gd, "nag": nag , "rmsprop":  RMSProp, "adam": Adam "nadam": Nadam

optimizer = Optimizer(arguments.optimizer)
optimizer.set_initial_parameters(parameters_adam)


#  "cross_entropy" : Cross Entropy Loss Function,  "squared_loss" : Squared Error Loss Function
loss_fun = LossFunction(arguments.loss)

#Select activation-function(hidden layers) pass below respective string to select any Activation Fuction Eg:"tanh" in ActivationFunction
# "sigmoid": SigmoidFunction, "softmax": SoftmaxFunction, "tanh": TanhFunction, "relu":ReLUFunction
act_fun_hidden = ActivationFunction(arguments.activation)

#Select activation - function for output layer
act_fun_output = ActivationFunction("softmax")

#Add layer sizes for the hidden layers
layers = [arguments.hidden_size] * arguments.num_layers
batch_size = arguments.batch_size
model = FFNN(optimizer, layers, loss_fun, act_fun_hidden, act_fun_output, batch_size, arguments.epochs, initialization = arguments.weight_init, log=1)  # log=1 for enabling to make push on wandb. 
train_loss, val_loss, train_accuracy, val_accuracy = model.fit(x_train, y_OH_train, x_test, y_OH_test)

# layers = [arguments.hidden_size] * arguments.num_layers
# batch_size = arguments.batch_size
# model = FFNN(optimizer, layers, loss_fun, act_fun_hidden, act_fun_output, batch_size, 1, initialization = arguments.weight_init)
# train_loss, val_loss, train_accuracy, val_accuracy = model.fit(x_train, y_OH_train, x_test, y_OH_test)
y_preds = model.predict(x_test)
accuracy_train = accuracy_score(np.argmax(y_OH_test,1), np.argmax(y_preds,1))
print("Training accuracy", round(accuracy_train, 3))
wandb.finish()