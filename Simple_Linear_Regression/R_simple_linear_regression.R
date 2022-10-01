#Simple Linear Regression
#Data Preprocessing

#Importing the dataset
dataset = read.csv('Salary_Data.csv')

#Spliting Dataset into Traing set and Test set
#import library
#install.packages('caTools')
library(caTools)
set.seed(124)
split = sample.split(dataset$Salary, SplitRatio = 2/3)#split ratio for training set
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Fitting Simple Linear Regression to the training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#Predict Test set result

y_pred = predict(regressor, newdata = test_set)

#Visualising the training set results
#install.packages('ggplot2')
#library(ggplot2)
ggplot()+
  geom_point(aes(x=training_set$YearsExperience,y = training_set$Salary), 
             color = 'red') +
  geom_line(aes(x=training_set$YearsExperience,y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience(Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

#Visualising the test set results
ggplot()+
  geom_point(aes(x=test_set$YearsExperience,y = test_set$Salary), 
             color = 'red') +
  geom_line(aes(x=training_set$YearsExperience,y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Experience(Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')