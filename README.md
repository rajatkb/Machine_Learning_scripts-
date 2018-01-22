# Machine_Learning_scripts-
Machine_learning_Scripts will be a collection of all small time analytics that i do with datasets. The repo includes simple mathematical methods implementations or sklearn based one analysis. 
# 1. Gradient Descent
  In order to use the script just simply call the script while passing simple parameters as follow  
  
  $ gradent_descent.py nameOfdatasetFile.csv feauture1 feature2 learningrate_or_alpha epoc  
  
  all of the parameters are mandatory.Proper learning rate should chosen. i.e below 1 obviously you dont want your function  
  to overshoot.
# 2. K-Nearest Neighbour
  Ya its knn nothing really interesting... why is it even here !! Ahh i know for people looking out for simple implementation example too   look into
# 3. Logistic Regression
  An extermely simple Logistic Regression variant. Will uddate and make it proper classification worthy with added layers
# 4. Keras MNIST
  So i took the handwritting number dataset of MNIST and put it through a conv net made using Keras with tensorflow backend. Its just for code refference for anyone looking for any the architecture used is given below <br/>
  (CONV2d,RELU) => (CONV2d ,RELU) => MaxPooling => dropout => fully connected => dropout => fullconnected_10
# 5. MNIST Fashion Data using keras
  Model used for the analysis <br/>
  Conv2d(3,3,c = 64) => Conv2d(4,4,c=128) => Conv2d(2,2,256) => AveragePool(2,2) =>dropoout=> faltten=>Dense=>dropout=> dense => dense =>softmax output
  given is a trained model giving 90% accuracy , did an early stopping since limited by my computing power will continue with my Desktop grad GPU.
  * UPDATE : New GPU Weights are here with accuracy 0.9932 accuracy in train data and 0.93 in test. Should change the model maybe since the model is overfitting to the data
