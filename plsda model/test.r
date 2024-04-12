# Load required libraries
library(caret)
library(pls)

# Load the saved PLS-DA model
model <- readRDS("plsda_model.rds")

# Load the test dataset
test_data <- read.csv("trim test.csv")

# Map the 'NAME' column to the 'CLASS' column
test_data$CLASS <- as.factor(test_data$NAME)

# Remove unnecessary columns
test_data <- test_data[, !(names(test_data) %in% c("SAMPLE", "NAME"))]

# Predict using the loaded model
predictions <- predict(model, newdata = test_data)

# Generate confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$CLASS)

# Print confusion matrix and statistics
print(conf_matrix)
