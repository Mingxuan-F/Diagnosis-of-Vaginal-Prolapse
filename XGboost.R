
# Load required libraries
library(xgboost)
library(pROC)

# Function to perform classification using XGBoost
classify_with_xgboost <- function(train_features, train_label, test_features, test_label, params) {
  dtrain <- xgb.DMatrix(data = as.matrix(train_features), label = train_label)
  dtest <- xgb.DMatrix(data = as.matrix(test_features), label = test_label)
  
  model <- xgboost(params, data = dtrain, nrounds = params$nrounds)
  predictions <- predict(model, dtest)
  
  # Convert probabilities to binary predictions (0 or 1)
  binary_predictions <- ifelse(predictions > 0.5, 1, 0)
  
  # Calculate accuracy
  accuracy <- mean(binary_predictions == test_label)
  
  # Calculate F1 score
  true_positive <- sum(binary_predictions == 1 & test_label == 1)
  false_positive <- sum(binary_predictions == 1 & test_label == 0)
  false_negative <- sum(binary_predictions == 0 & test_label == 1)
  
  precision <- true_positive / (true_positive + false_positive)
  recall <- true_positive / (true_positive + false_negative)
  
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Calculate AUC
  auc <- roc(test_label, predictions)$auc
  
  # Return accuracy, F1 score, and AUC
  return(list(accuracy = accuracy, f1_score = f1_score, auc = auc))
}
data <- read.csv('/Users/fanmingxuan/Desktop/GAIN_BAR_selected.csv',header=T)
#训练集、测试集划分
# Set a random seed for reproducibility
set.seed(42)
# Proportion of data to be used for testing (e.g., 0.2 for 20%)
test_size <- 0.2
# Set the number of iterations
num_iterations <- 100

# Vectors to store metrics
accuracies <- numeric(num_iterations)
f1_scores <- numeric(num_iterations)
auc_values <- numeric(num_iterations)

# Set XGBoost parameters
params <- list(
  objective = "binary:logistic",  # For binary classification
  eval_metric = "logloss",        # Logarithmic loss metric
  eta = 0.1,                      # Learning rate
  max_depth = 8,                  # Maximum depth of a tree
  subsample = 0.8,                # Subsample ratio of the training data
  colsample_bytree = 0.8,         # Subsample ratio of columns when constructing each tree
  nrounds = 100                   # Number of boosting rounds (iterations)
)

# Repeat the classification process 1000 times
for (i in 1:num_iterations) {
  # Split data into training and testing sets (replace this with your own data splitting method)
  # train_features, train_label, test_features, and test_label should be prepared before this loop
  # ...
  train_sub = sample(nrow(data),8/10*nrow(data))
  train = data[train_sub,]
  test =data[-train_sub,]
  
  train_features <- train[,1:(ncol(data)-1)]
  test_features <- test[,1:(ncol(data)-1)]
  train_label <- train[,ncol(data)]
  test_label <- test[,ncol(data)]
  # Perform classification and store metrics
  metrics <- classify_with_xgboost(train_features, train_label, test_features, test_label, params)
  accuracies[i] <- metrics$accuracy
  f1_scores[i] <- metrics$f1_score
  auc_values[i] <- metrics$auc
}

# Calculate the mean and standard deviation of metrics
mean_accuracy <- mean(accuracies)
mean_f1_score <- mean(f1_scores)
mean_auc <- mean(auc_values)

sd_accuracy <- sd(accuracies)
sd_f1_score <- sd(f1_scores)
sd_auc <- sd(auc_values)

# Print the results
cat("Mean accuracy:", mean_accuracy, "\n")
cat("Mean F1 score:", mean_f1_score, "\n")
cat("Mean AUC:", mean_auc, "\n")

cat("Standard deviation of accuracies:", sd_accuracy, "\n")
cat("Standard deviation of F1 scores:", sd_f1_score, "\n")
cat("Standard deviation of AUC values:", sd_auc, "\n")

