import numpy as np
import cv2
import os
def load_data(filename):
  fh=open(filename,"r",encoding="utf-8")
  lines=fh.readlines()
  data=[]
  label=[]
  for line in lines:
      line=line.strip("\n")
      line=line.strip()
      words=line.split()
      imgs_path=words[0]
      labels=words[1]
      label.append(labels)
      data.append(imgs_path)
  return data,label 

# Define the two-layer neural network class

def load_mydata(filename,width,height): 
  data,label=load_data(filename)
  xs = []
  
  for i in range(len(label)):
    image_dir="C:/Users/user/Desktop/"
    img_path=os.path.join(image_dir,data[i])
    image=cv2.imread(img_path)
    
    if image.ndim==2:
        image=cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
    X=cv2.resize(image,(width, height), interpolation=cv2.INTER_AREA)
    xs.append(X)
    

  Xtr = np.array(xs)
  Ytr = np.asarray(label,dtype=int)
  
  return Xtr, Ytr
# Define the neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Calculate the output of the first layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)

        # Calculate the output of the second layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)

        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Calculate the gradient of the second layer
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Calculate the gradient of the first layer
        dZ1 = np.dot(dZ2, self.W2.T) * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update the weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
input_width=32
input_height=32
num_classes=50
X_train, y_train =load_mydata("C:/Users/user/Desktop/train.txt",input_width,input_height)
X_test, y_test =load_mydata("C:/Users/user/Desktop/test.txt",input_width,input_height)
X_val, y_val =load_mydata("C:/Users/user/Desktop/val.txt",input_width,input_height)
print(np.shape(y_train))

# Load the data and preprocess it
X = X_train # Your input data as a numpy array of shape (num_examples, 32, 32, 3)
y = np.matmul(np.reshape(y_train,(y_train.shape[0],-1)),np.ones((1,num_classes))) # Your labels as a numpy array of shape (num_examples, num_classes)
print(np.shape(y))

X = X.reshape(X.shape[0], -1) / 255.0

# Initialize the neural network
input_size = X.shape[1]
hidden_size = 128
output_size = y.shape[1]
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
num_epochs = 1000
learning_rate = 0.000001
for epoch in range(num_epochs):
    # Forward pass
    y_pred = nn.forward(X)
    
    

    # Compute the loss
    loss = np.sum(np.sum(y * np.log(y_pred), axis=1),axis=0)

    # Backward pass
    nn.backward(X, y, learning_rate)
    

    # Print the loss every 100 epochs
    if epoch % 10 == 0:
        
        result = np.argmax(y_pred, axis=1) - y_train
        result = list(result)
        acc=str(result.count(0)/X_test.shape[0])
        x_labels = "epoch: {}  loss: {}  acc: {} ".format(epoch, loss,acc)
        print(x_labels)

X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
Y_pred = nn.forward(X_test)
result = np.argmax(Y_pred, axis=1) - y_test
result = list(result)
print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(result.count(0)/X_test.shape[0]))

# TEST SET ACC
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0
Y_pred = nn.forward(X_val)
result = np.argmax(Y_pred, axis=1) - y_val
result = list(result)
print("VALIDATION--> Correct: " + str(result.count(0)) + " out of " + str(X_val.shape[0]) + ", acc=" + str(result.count(0)/X_val.shape[0]))
