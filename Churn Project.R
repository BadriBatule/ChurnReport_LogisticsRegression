# ================================================================
# Step 1: Load Required Libraries and Dataset
# ================================================================
library(readr)
library(tidyverse)
library(caret)
library(pROC)
library(e1071)

# Load the dataset
WA_Fn_UseC_Telco_Customer_Churn <- read_csv("Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Inspect structure
str(WA_Fn_UseC_Telco_Customer_Churn)

# ================================================================
# Step 2: Data Preprocessing
# ================================================================

# Drop customerID and convert character columns to factors
df <- WA_Fn_UseC_Telco_Customer_Churn %>%
  select(-customerID) %>%
  mutate(
    Churn = factor(Churn, levels = c("No", "Yes")),  # Target variable
    across(where(is.character), as.factor)           # Convert character vars to factors
  )

# Confirm structure
str(df)

# ================================================================
# Step 3: Split Dataset into Training and Test Sets (70/30)
# ================================================================

set.seed(123)  # For reproducibility

# Create partition indices
train_index <- createDataPartition(df$Churn, p = 0.7, list = FALSE)

# Subset the data
train <- df[train_index, ]
test <- df[-train_index, ]

# Convert to base R data.frames for compatibility
train <- as.data.frame(train)
test <- as.data.frame(test)

# ================================================================
# Step 4: Train Logistic Regression Model
# ================================================================

logit_model <- glm(Churn ~ ., data = train, family = binomial)

# View model summary
summary(logit_model)

# ================================================================
# Step 5: Predict on Test Data and Evaluate
# ================================================================

# Predict probabilities
probabilities <- predict(logit_model, newdata = test, type = "response")

# Convert to binary predictions
predicted_classes <- ifelse(probabilities > 0.5, "Yes", "No")
predicted_classes <- factor(predicted_classes, levels = c("No", "Yes"))  # Match factor levels

# Confusion matrix
conf_matrix <- confusionMatrix(predicted_classes, test$Churn, positive = "Yes")
print(conf_matrix)

# ================================================================
# Step 6: ROC Curve and AUC
# ================================================================

roc_obj <- roc(test$Churn, probabilities)
plot(roc_obj, col = "blue", main = "ROC Curve - Logistic Regression")
auc(roc_obj)  # Area Under Curve

# ================================================================
# Step 7: Business-Focused Visualizations
# ================================================================

# 7.1 Churn Rate by Contract Type
ggplot(df, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn Rate by Contract Type", y = "Proportion", x = "Contract Type") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal()

# 7.2 Churn Rate by Tenure Group
df$TenureGroup <- cut(df$tenure,
                      breaks = c(0, 12, 24, 48, 72),
                      labels = c("0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"),
                      right = TRUE)

ggplot(df, aes(x = TenureGroup, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn Rate by Tenure Group", x = "Tenure Group", y = "Proportion") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal()

# 7.3 Churn Distribution by Monthly Charges
ggplot(df, aes(x = MonthlyCharges, fill = Churn)) +
  geom_density(alpha = 0.5) +
  labs(title = "Monthly Charges Distribution by Churn", x = "Monthly Charges") +
  theme_minimal()

# 7.4 Churn by Payment Method
ggplot(df, aes(x = PaymentMethod, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn by Payment Method", y = "Proportion", x = "Payment Method") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 7.5 Churn by Paperless Billing
ggplot(df, aes(x = PaperlessBilling, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn by Paperless Billing", y = "Proportion", x = "Paperless Billing") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal()
