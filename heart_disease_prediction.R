# ==============================================
# HEART DISEASE ML COMPARISON WITH GRAPHS
# ==============================================

# Set working directory
setwd("C:/Users/Admin/Desktop/Surya mini project/R program")

# Load required packages
packages <- c("tidyverse","caret","randomForest","e1071","pROC","corrplot")
for(p in packages){
  if(!require(p, character.only = TRUE)) install.packages(p, dependencies=TRUE)
  library(p, character.only = TRUE)
}

# Load dataset
if(!file.exists("heart.csv")) stop("heart.csv not found!")
heart_data <- read.csv("heart.csv")

# Convert target to factor
if(!is.factor(heart_data$target)) heart_data$target <- as.factor(heart_data$target)

cat("✅ Dataset loaded successfully! Records:", nrow(heart_data), "Columns:", ncol(heart_data), "\n")

# ============================
# Split data
# ============================
set.seed(123)
train_index <- createDataPartition(heart_data$target, p=0.8, list=FALSE)
train_data <- heart_data[train_index, ]
test_data <- heart_data[-train_index, ]

# ============================
# 1. Logistic Regression
# ============================
lr_model <- glm(target ~ ., data=train_data, family=binomial)
lr_prob <- predict(lr_model, newdata=test_data, type="response")
lr_pred <- ifelse(lr_prob>0.5,1,0) %>% as.factor()
lr_conf <- confusionMatrix(lr_pred, test_data$target, positive="1")

# ============================
# 2. Random Forest
# ============================
rf_model <- randomForest(target~., data=train_data, ntree=100)
rf_pred <- predict(rf_model, newdata=test_data)
rf_conf <- confusionMatrix(rf_pred, test_data$target, positive="1")

# ============================
# 3. Support Vector Machine
# ============================
svm_model <- svm(target~., data=train_data, probability=TRUE)
svm_pred <- predict(svm_model, newdata=test_data)
svm_conf <- confusionMatrix(svm_pred, test_data$target, positive="1")

# ============================
# Accuracy Output Page
# ============================
accuracy_df <- data.frame(
  Model = c("Logistic Regression","Random Forest","SVM"),
  Accuracy = c(as.numeric(lr_conf$overall["Accuracy"]),
               as.numeric(rf_conf$overall["Accuracy"]),
               as.numeric(svm_conf$overall["Accuracy"]))
)
print("✅ Model Accuracy Comparison:")
print(accuracy_df)

# ============================
# Accuracy Comparison Bar Graph
# ============================
ggplot(accuracy_df, aes(x=Model, y=Accuracy, fill=Model)) +
  geom_bar(stat="identity") +
  ylim(0,1) +
  geom_text(aes(label=round(Accuracy,3)), vjust=-0.5) +
  labs(title="Model Accuracy Comparison") +
  theme_minimal()

# ============================
# ROC Curve Comparison
# ============================
lr_roc <- roc(test_data$target, lr_prob)
rf_prob <- predict(rf_model, newdata=test_data, type="prob")[,2]
rf_roc <- roc(test_data$target, rf_prob)
svm_prob <- attr(predict(svm_model,newdata=test_data,probability=TRUE),"probabilities")[,2]
svm_roc <- roc(test_data$target, svm_prob)

plot(lr_roc, col="blue", lwd=2, main="ROC Curve Comparison")
lines(rf_roc, col="green", lwd=2)
lines(svm_roc, col="red", lwd=2)
legend("bottomright", legend=c("Logistic Regression","Random Forest","SVM"),
       col=c("blue","green","red"), lwd=2)

cat("\nAUC Values:\n")
cat("Logistic Regression:", round(auc(lr_roc),3), "\n")
cat("Random Forest:", round(auc(rf_roc),3), "\n")
cat("SVM:", round(auc(svm_roc),3), "\n")

# ============================
# Feature Importance (Random Forest)
# ============================
importance_df <- data.frame(
  Feature = rownames(rf_model$importance),
  Importance = rf_model$importance[,1]
)

ggplot(importance_df, aes(x=reorder(Feature, Importance), y=Importance, fill=Importance)) +
  geom_bar(stat="identity") +
  coord_flip() +
  labs(title="Feature Importance - Random Forest") +
  theme_minimal()

# ============================
# New Patient Prediction Example
# ============================
new_data <- data.frame(
  age=55, sex=1, cp=2, trestbps=140, chol=230, fbs=0, restecg=1,
  thalach=165, exang=0, oldpeak=1.2, slope=2, ca=0, thal=2
)

lr_pred_new <- ifelse(predict(lr_model,new_data,type="response")>0.5,"💔 Heart Disease","❤️ No Heart Disease")
rf_pred_new <- ifelse(predict(rf_model,new_data)=="1","💔 Heart Disease","❤️ No Heart Disease")
svm_pred_new <- ifelse(predict(svm_model,new_data)=="1","💔 Heart Disease","❤️ No Heart Disease")

cat("\n--- New Patient Prediction ---\n")
cat("Logistic Regression:", lr_pred_new, "\n")
cat("Random Forest:", rf_pred_new, "\n")
cat("SVM:", svm_pred_new, "\n")
