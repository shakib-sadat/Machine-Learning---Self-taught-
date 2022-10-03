#Data Preprocessing

#Importing the dataset
dataset = read.csv('50_Startups.csv')
#dataset = dataset[,2:3]

#Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York','California','Florida'),
                         labels= c(1,2,3))

#Spliting Dataset into Traing set and Test set
#import library
#install.packages('caTools')
library(caTools)
set.seed(124)
split = sample.split(dataset$Profit, SplitRatio = 0.7)#split ratio for training set
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Fitting Multiple Linear Regression to The traing set
regressor = lm(formula = Profit ~ .,
               data = training_set)
#predicting the test set results
y_pred = predict(regressor, newdata = test_set)

#Building Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)

summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)

summary(regressor)