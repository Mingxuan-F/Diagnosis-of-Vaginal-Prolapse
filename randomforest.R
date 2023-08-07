# Load the required libraries
library(randomForest)
library(pROC)
library(caret)

rm(list = ls()) 
setwd('/Users/fanmingxuan/Desktop')
data<-read.csv('GAIN_imputed.csv',header=T)
X <- data[,2:(ncol(data)-1)]
y <- as.factor(data[,ncol(data)])
# 共线性诊断
#XX<-cor(data[,-42])
#kappa(XX,exact = TRUE)
n_repeat=100
auc_val=vector()
f1_scores=vector()
accuracy=vector()

# Assuming you have your data in the form of X (features) and y (labels/targets)
for (i in 1:n_repeat){
  # Step 1: Split the data into training and testing sets
  train_indices <- sample(nrow(X), 0.9 * nrow(X))  # 80% of data for training
  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[-train_indices, ]
  y_test <- y[-train_indices]
  
  # Step 2: Initialize the Random Forest Classifier
  # You can adjust hyperparameters like ntree (number of trees) and mtry (number of features considered for splitting)
  rf_classifier <- randomForest(x = X_train, y = y_train, ntree = 30, mtry = sqrt(ncol(X_train)-2))
  
  # Step 3: Make predictions on the test data
  y_pred <- predict(rf_classifier, X_test)
  
  # Step 4: Evaluate the model's performance
  
  # Calculate F1 score
  confusion_matrix <- confusionMatrix(y_pred, y_test,positive = '1')
  f1_scores[i] <- confusion_matrix$byClass[["F1"]]
  accuracy[i] <- confusion_matrix[["overall"]][["Accuracy"]]
  # Get predicted probabilities for the positive class
  # Calculate AUC
  auc_val[i] <- roc(y_test, as.numeric(y_pred))$auc}

mean(accuracy[accuracy>0.8])

sd(accuracy)
mean(f1_scores)
sd(f1_scores)
mean(auc_val)
sd(auc_val)



