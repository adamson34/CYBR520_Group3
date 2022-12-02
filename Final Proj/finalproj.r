#------Group 3 - Final Project---------
#------ML with a reduced version of the 'Hikari' Dataset-----

#readxl package allows XLSX to be loaded in 
install.packages('readxl')
install.packages('caret')
install.packages('rpart.plot')
install.packages('e1071')

library(readxl)
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)

#loading dataset into Rstudio 
networkeventbase <- read.csv('datasetreducedhikari.csv' , sep = ',')

#set seed to ensure repeatability (can be any random #)
set.seed(34)

#------Decision Tree Model--------
trainIndex <- createDataPartition(networkeventbase$Label, p=0.70, list=FALSE)

#70% of the networkeventkbase Data becomes "the training dataset"
train <- networkeventbase[ trainIndex, ]

#30% of the networkeventkbase Data becomes "the testing dataset"
test <- networkeventbase[-trainIndex, ]

#Build the Decision Tree model
treemodel <- rpart(Label~., data = train)

#Visualize the treemodel
treeplot <- rpart.plot(treemodel)

treemodel$variable.importance

Prediction <- predict(treemodel, test, type = 'class')  

Prediction

Confmatrix <- table(test$Label, Prediction)

Confmatrix 

Acc <- (sum(diag(Confmatrix)) / sum(Confmatrix)*100)
Acc

TN=Confmatrix[1]
FN=Confmatrix[2]
FP=Confmatrix[3]
TP=Confmatrix[4]

Accuracy <- (TP+TN)/sum(Confmatrix)*100
Precision <- (TP/ (TP+FP))*100
Recall <- (TP/(TP+FN))*100

Accuracy
Precision
Recall
#------SVM--------
trainctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

svm.model <- train(Label~., data=train, method="svmRadial", tuneLength = 10, trControl = trainctrl, metric="Accuracy")

svm.model$results

svm.predict <- predict(svm.model, test)

confusionMatrix(svm.predict, as.factor(test$Label), mode = "prec_recall")

svm.model$results

class(svm.model$results)

plot(y=svm.model$results$Accuracy,x=svm.model$results$C,
     ylab="Observed Accuracy",
     xlab="Cost Parameter")

plot(svm.model)


#------K-NN--------
knn.model <- train(Label~., data=train, method="knn",
                   tuneLength = 10,
                   trControl = trainctrl,
                   metric="Accuracy")

knn.model$results

knn.predict <- predict(knn.model, test)

confusionMatrix(knn.predict, as.factor(test$Label), mode = "prec_recall")

knn.model$results
class(knn.model$results)

plot(x=knn.model$results$Accuracy,y=knn.model$results$k,
     xlab="Observed Accuracy",
     ylab="Neighbour")

plot(knn.model)
