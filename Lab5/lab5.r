#Group 3 - Luke Adamson, Cole Taylor, Charrick Pierce, Logan Muzina, Brian Britton, Pepper Reasnover 

#Lab 5

#The first time you run this code, un-comment the two next lines to install the packages
install.packages('caret')
install.packages('e1071')
install.packages("randomForest")
install.packages('mlbench')
install.packages('Hmisc')
install.packages('corrplot')
install.packages('ggcorrplot')
install.packages('https://cran.r-project.org/src/contrib/Archive/randomForest/', repos=NULL, type="source") 


#load libraries
library(caret)
library(e1071)
library(mlbench)
library(caret)
library(randomForest)
library(Hmisc)
library(corrplot)
library(ggcorrplot)

# ensure the results are repeatable
set.seed(7)

spambase <- read.csv('spambase.csv', sep = ',')
spambase$x <- NULL

#run before correlation matrix to get proper results
# Change type to non spam 0 and spam 1 
spambase[spambase =='nonspam'] <- as.numeric(0)              
spambase[spambase =='spam'] <- as.numeric(1)                 
spambase$type = as.numeric(spambase$type)                    
str(spambase)

# calculate correlation matrix
correlationMatrix <- cor(spambase[,1:58])
# summarize the correlation matrix
print(correlationMatrix)

#get correlation matrix - representation 1
visCorMatrix1 <- corrplot(cor(spambase))

#get correlation matrix - representation 2
visCorMatrix2<-ggcorrplot(cor(spambase))

visCorMatrix2

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5,verbose = FALSE,names = TRUE )
# print highly correlated attributes
print(highlyCorrelated)

# ensure results are repeatable
set.seed(7)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(as.factor(type)~., data=spambase, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
# This will take a minute
importance <- varImp(model, scale=FALSE)
# summarize importance
# This will take a minute
print(importance)
# plot importance
plot(importance)

#Recursive Feature Elimination (RFE) ---------------------------------
# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data
# define the control using a random forest selection function
# This will take a moment.FYI.
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# run the RFE algorithm
results <- rfe(spambase[,1:56], spambase[,57], sizes=c(1:56), rfeControl=control)

# summarize the results
# this will take a hot minute
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))


# svm classification model using full data ---------------------------------

set.seed(7)
trainIndex <- createDataPartition(spambase$type, p=0.7, list= FALSE)
Train <- spambase[ trainIndex, ]
Test <- spambase[ -trainIndex, ]
trainctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
svm.model <- train(type~., data=Train, method="svmRadial",
                   tuneLength = 10,
                   trControl = trainctrl,
                   metric="Accuracy")
svm.model$results
svm.predict <- predict(svm.model, Test)
confusionMatrix(svm.predict, as.factor(Test$type), mode = "prec_recall")

# svm classification model using step 6 ---------------------------------
set.seed(7)
varstep6 <- c("charExclamation", "your", "num000", "remove", "charDollar", "you", "free", 
            "business", "hp", "capitalTotal", "our", "receive", "hpl", "over", "order", "money", 
            "capitalLong", "internet", "email", "all", "type")
step6 <- spambase[varstep6]

step6$type = as.factor(step6$type)

trainIndex6 <- createDataPartition(step6$type, p=0.7, list = FALSE)
Train6 <- step6[ trainIndex6, ]
Test6 <- step6[ -trainIndex6, ]
trainctrl6 <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
svm.model6 <- train(type~., data=Test6, method="svmRadial",
                   tuneLength = 10,
                   trControl = trainctrl6,
                   metric="Accuracy")
svm.model6$results
svm.predict6 <- predict(svm.model, Test6)
confusionMatrix(svm.predict6, as.factor(Test6$type), mode = "prec_recall")

# svm classification model using step 8 ---------------------------------
varstep8 <- c("capitalLong", "report", "order", "num1999", "charHash", "type")
step8 <- spambase[varstep8]
step8$type = as.factor(step8$type)

trainIndex8 <- createDataPartition(step8$type, p=0.7, list = FALSE)
Train8 <- step6[ trainIndex8, ]
Test8 <- step6[ -trainIndex8, ]
trainctrl8 <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
svm.model8 <- train(type~., data=Test8, method="svmRadial",
                   tuneLength = 10,
                   trControl = trainctrl8,
                   metric="Accuracy")
svm.model8$results
svm.predict8 <- predict(svm.model8, Test8)
confusionMatrix(svm.predict8, as.factor(Test8$type), mode = "prec_recall")
