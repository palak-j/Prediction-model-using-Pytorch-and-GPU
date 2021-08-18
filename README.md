# Prediction-model-using-Pytorch-and-GPU

Aim: To predict outcome(diabetic or non-diabetic) based on variuos input parameters such as glucose, BMI, pregnancies, blood pressure, Insulin etc. <br/>
Approach: Here, I have used simple ANN model using pytorch with two hidden layers, one output layer and relu activation function. <br/>

We can do this with CPU as well as GPU. By default, it will execute on CPU but if we want to use GPU, we have to give .cuda() fuction to all tensor inputs, models and outputs. <br/>

I have used GPU to run my model. <br/>

This project consist of various parts:
### Part1: Import training data <br/>
I have used Pima Indians Diabetes Database from Kaggle [https://www.kaggle.com/uciml/pima-indians-diabetes-database]
### Part2: Visualization of data <br/> 
For Visualization, various plots have been used such as frequency plots, comparison plot of outcome with all parameters seperately, correlation matrix and pair plots.
### Part 3: Modelling the data <br/> 
Used ANN model using pytorch with two hidden layers, one output layer and relu activation function.
Dont forget to use .cuda() function if you want to use GPU.
### Part 4: Backward Propogation <br/>
We have to set an optimizer and loss function for backward propogation. 
### Part 5: Initiate the model <br/>
Select number of required epochs and run forward function, calculate loss for each epoch and use backward function with loss(the one we want to minimize). 
### Part 6: Analysis of model and Prediction <br/>
Plot loss function, get predicted values for test data. We can use confusion matrix to analyze the results. 

### Part 6: Save the model <br/>
To save the model, simply use : torch.save(model,'model_name.pt')
To access it again, use : model = torch.load('model_name.pt')


