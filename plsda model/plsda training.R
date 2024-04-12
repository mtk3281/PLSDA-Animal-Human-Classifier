library(caret)
library(pls)

train_and_test <- function(cv_number) {
  
  train_data <- read.csv("trim train.csv")
  
  # Map the 'NAME' column to the 'CLASS' column
  train_data$CLASS <- as.factor(train_data$NAME)
  
  # Remove unnecessary columns
  train_data <- train_data[, !(names(train_data) %in% c("SAMPLE", "NAME"))]
  
  # Define cross-validation folds for training
  
  ctrl <- trainControl(method = "cv", number = cv_number)
  
  tuneGrid <- expand.grid(ncomp = c(10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60))
  
  # Train the PLS-DA model
  model <- train(CLASS ~ ., data = train_data, method = "pls", trControl = ctrl, tuneGrid = tuneGrid, preProcess = "scale")
  
  # Save the trained model
  saveRDS(model, "plsda_model.rds")
  
  # Print the best model
  print(model)
  
  # Load the test data
  test_data <- read.csv("trim test.csv")
  
  # Map the 'NAME' column to the 'CLASS' column
  test_data$CLASS <- as.factor(test_data$NAME)
  
  # Remove unnecessary columns
  test_data <- test_data[, !(names(test_data) %in% c("SAMPLE", "NAME"))]
  
  # Predict using the trained model
  predictions <- predict(model, newdata = test_data)
  
  # Generate confusion matrix
  conf_matrix <- confusionMatrix(predictions, test_data$CLASS)
  
  # Print confusion matrix and statistics
  print(conf_matrix)
}

# Define the cross-validation number
cv_number <- 10

# Call the function with the specified cross-validation number
train_and_test(cv_number)
