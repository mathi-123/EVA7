# Assignment: Session 4 - Backpropagation and Architectural Basics

## PART 1[250]: Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github. 

1.Use exactly the same values for all variables as used in the class

2.Take a screenshot, and show that screenshot in the readme file

3.Excel file must be there for us to cross-check the image shown on readme (no image = no score)

Explain each major step

Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 

Upload all this to GitHub and then write all above as part 1 of your README.md file. 

Submit details to S4 - Assignment QnA. 

## PART 2 [250]:  Question Description

We have considered many points in our last 4 lectures. Some of these we have covered directly and some indirectly. They are:

How many layers,

MaxPooling,

1x1 Convolutions,
3x3 Convolutions,
Receptive Field,
SoftMax,
Learning Rate,
Kernels and how do we decide the number of kernels?
Batch Normalization,
Image Normalization,
Position of MaxPooling,
Concept of Transition Layers,
Position of Transition Layer,
DropOut

When do we introduce DropOut, or when do we know we have some overfitting
The distance of MaxPooling from Prediction,
The distance of Batch Normalization from Prediction,
When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
How do we know our network is not going well, comparatively, very early
Batch Size, and effects of batch size
etc (you can add more if we missed it here)

Refer to this code: COLABLINK https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx
WRITE IT AGAIN SUCH THAT IT ACHIEVES

99.4% validation accuracy
Less than 20k Parameters
You can use anything from above you want. 
Less than 20 Epochs
Have used BN, Dropout, a Fully connected layer, have used GAP. 
To learn how to add different things we covered in this session, you can refer to this code: https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99 DONT COPY ARCHITECTURE, JUST LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC.
This is a slightly time-consuming assignment, please make sure you start early. You are going to spend a lot of effort into running the programs multiple times
Once you are done, submit your results in S4-Assignment-Solution
You must upload your assignment to a public GitHub Repository. Create a folder called S4 in it, and add your iPynb code to it. THE LOGS MUST BE VISIBLE. Before adding the link to the submission make sure you have opened the file in an "incognito" window. 
If you misrepresent your answers, you will be awarded -100% of the score.
If you submit Colab Link instead of notebook uploaded on GitHub or redirect the GitHub page to colab, you will be awarded -50%

Submit details to S4 - Assignment QnA. 
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Solution -Assgnment Part-A

Neural Network BackPropogation using Excel
![Excel_NN](https://user-images.githubusercontent.com/91079217/137771680-50552506-ab9a-408c-b0c9-b60791869f26.jpg)



Backpropagation is a method for training a neural network. Backpropagation soptimize the weights so that the neural network can learn how to correctly map arbitrary inputs to outputs. We will see backpropagation with calculations and populate it in an excel sheet.
Here, we wll use a neural network with two inputs, two hidden neurons, two output neurons and ignore the bias.

### Here are the initial weights, to work with:

w1 = 0.15	w2 = 0.2	w3 = 0.25	w4 = 0.3
w5 = 0.4	w6 = 0.45	w7 = 0.5	w8 = 0.55

We’re going to work with a single training set: given inputs 0.05 and 0.10 and the expected outputs are  0.01 and 0.99.

### Forward Propogation

We will first pass the above inputs through the network by multiplying the inputs to the weights and calculate the h1 and h2

  h1 =w1*i1+w2+i2
  h2 =w3*i1+w4*i2

### The output from the hidden layer neurons (h1 and h2) are passed to activated neurons using a activation function (here we are using sigmoid activation),
this helps in adding non linearity to the network.

  a_h1 = σ(h1) = 1/(1+exp(-h1))
  a_h2 = σ(h2) = 1/(1+exp(-h2))
  
### Repeat this process for the output layer neurons, using the output from the hidden layer actiavted neurons as inputs.

  o1 = w5 * a_h1 + w6 * a_h2
  o2 = w7 * a_h1 + w8 * a_h2
  
  a_o1 = σ(o1) = 1/(1+exp(-o1))
  a_o2 = σ(o2) = 1/(1+exp(-o2))
  
### Calculate the error for each output neurons (a_o1 and a_o2) using the squared error function and sum them up to get the total error (E_total)

Calculating the Error (Loss)

E1 = ½ * ( t1 - a_o1)²
E2 = ½ * ( t2 - a_o2)²
E_Total = E1 + E2

Note: 1/2 is included so that exponent is cancelled when we differenciate the error term.

### Back Propogation

During back propogation, we would help the network learn and get better by updating the weights such that the total error is minimum

### First calculate the partial derivative of E_total with respect to w5

δE_total/δw5 = δ(E1 +E2)/δw5

δE_total/δw5 = δ(E1)/δw5       
             = (δE1/δa_o1) * (δa_o1/δo1) 
             
             = (δ(½ * ( t1 - a_o1)²) /δa_o1= (t1 - a_o1) * (-1) = (a_o1 - t1))   
                   * (δ(σ(o1))/δo1 = σ(o1) * (1-σ(o1)) = a_o1                       
                   * (1 - a_o1 )) * a_h1                                            
             = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
             
Similarly, calculate the partial derivative of E_total with respect to w6, w7, w8.

δE_total/δw5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
δE_total/δw6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2
δE_total/δw7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1
δE_total/δw8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2

Next, we continue back propogation through the hidden layers i.e we need to find how much the hidden neurons change wrt total Error

δE_total/δa_h1 = δ(E1+E2)/δa_h1 
               = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7
               
δE_total/δa_h2 = δ(E1+E2)/δa_h2 
               = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8
               
### Calculate the partial derivative of E_total with respect to w1, w2, w3 and w4 using chain rule

δE_total/δw1 = δE_total/δw1 = δ(E_total)/δa_o1 * δa_o1/δo1 * δo1/δa_h1 * δa_h1/δh1 * δh1/δw1
             = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i1
             

δE_total/δw2 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i2

δE_total/δw3 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i1

δE_total/δw4 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i2

Once we have gradients for all the weights with respect to the total error, we subtract this value from the current weight by multiplying with a learning rate

    w1 = w1 - learning_rate * δE_total/δw1
    w2 = w2 - learning_rate * δE_total/δw2
    w3 = w3 - learning_rate * δE_total/δw3
    w4 = w4 - learning_rate * δE_total/δw4
    w5 = w5 - learning_rate * δE_total/δw5
    w8 = w6 - learning_rate * δE_total/δw6
    w7 = w7 - learning_rate * δE_total/δw7
    w8 = w8 - learning_rate * δE_total/δw8
    
Repeat this entire process for forward and backward pass until we get minimum error.

### Error Graph for different Learning rates

Below is the error graph when we change the learning rates 0.1, 0.2, 0.5, 0.8, 1.0, 2.0

![119750792-a31fe080-beb7-11eb-948a-fe1f6d4c74c7](https://user-images.githubusercontent.com/91079217/137772103-65dedfc1-8b8a-4790-bf81-2c918e3542f6.png)

Observe that with small learning rate the loss is going to drop very slowly and takes lot of time to converge, so we should always be choosing optimal learning rate neither too low nor too high (if its too high it never converges).

----------------------------------------------------------------------------------------------------------------------------------------------------------------

## Part2-Solution
### https://github.com/mathi-123/EVA7/blob/main/S4/Session_4_Assignment2.ipynb

### Model parameters

Total params: 17,882
Trainable params: 17,882
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.26
Params size (MB): 0.07
Estimated Total Size (MB): 1.33
-------------------


### Accuracy logs of 20 epochs

oss=0.01379500050097704 batch_id=937: 100%|██████████| 938/938 [00:29<00:00, 32.15it/s]
Test set: Average loss: 0.0514, Accuracy: 9850/10000 (98.50%)

loss=0.037570904940366745 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.55it/s]
Test set: Average loss: 0.0368, Accuracy: 9895/10000 (98.95%)

loss=0.08293569087982178 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 33.13it/s]
Test set: Average loss: 0.0298, Accuracy: 9899/10000 (98.99%)

loss=0.011683406308293343 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.57it/s]
Test set: Average loss: 0.0263, Accuracy: 9914/10000 (99.14%)

loss=0.02773825265467167 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 33.24it/s]
Test set: Average loss: 0.0259, Accuracy: 9916/10000 (99.16%)

loss=0.049558673053979874 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.92it/s]
Test set: Average loss: 0.0263, Accuracy: 9917/10000 (99.17%)

loss=0.0351916141808033 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.36it/s]
Test set: Average loss: 0.0197, Accuracy: 9937/10000 (99.37%)

loss=0.04750240966677666 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.44it/s]
Test set: Average loss: 0.0206, Accuracy: 9931/10000 (99.31%)
loss=0.12883694469928741 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.72it/s]
Test set: Average loss: 0.0189, Accuracy: 9941/10000 (99.41%)

loss=0.1224433034658432 batch_id=937: 100%|██████████| 938/938 [00:29<00:00, 32.32it/s]
Test set: Average loss: 0.0173, Accuracy: 9941/10000 (99.41%)

loss=0.003706124145537615 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.46it/s]
Test set: Average loss: 0.0183, Accuracy: 9934/10000 (99.34%)

loss=0.008542485535144806 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.53it/s]
Test set: Average loss: 0.0177, Accuracy: 9937/10000 (99.37%)

loss=0.0080399289727211 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.81it/s]
Test set: Average loss: 0.0179, Accuracy: 9944/10000 (99.44%)

loss=0.09832636266946793 batch_id=937: 100%|██████████| 938/938 [00:29<00:00, 32.13it/s]
Test set: Average loss: 0.0166, Accuracy: 9944/10000 (99.44%)

loss=0.007235725875943899 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.79it/s]
Test set: Average loss: 0.0176, Accuracy: 9940/10000 (99.40%)

loss=0.000455019879154861 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 33.17it/s]
Test set: Average loss: 0.0172, Accuracy: 9949/10000 (99.49%)

loss=0.025439731776714325 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.45it/s]
Test set: Average loss: 0.0177, Accuracy: 9938/10000 (99.38%)

loss=0.07479388266801834 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.38it/s]
Test set: Average loss: 0.0166, Accuracy: 9948/10000 (99.48%)



