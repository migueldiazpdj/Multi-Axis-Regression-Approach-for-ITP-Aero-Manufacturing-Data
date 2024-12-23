# Supervised Part 1: ITP AERO by Miguel Diaz and Dante Schrantz
# Using the class codes to try to solve the ITP Aero company problem. The database was matched by the company's representatives together with the other companies as one of the 7 challenges during the championship days. We asked them to share their challenge with us with the purpose of using the problem for our Machine Learning final project.

rm(list=ls())
carga <- function(x){
  for( i in x ){
    if( ! require( i , character.only = TRUE ) ){
      install.packages( i , dependencies = TRUE )
      require( i , character.only = TRUE )
    }
  }
}
carga("caret")
library(caret)

# DATASET ITP: This is a Regression Problem
install.packages("openxlsx")
library(openxlsx)
ITP <- read.xlsx("C:/Users/Miguel/Downloads/IndesIAhack_ITPAero_dataset 1.xlsx", sheet = "data")
View(ITP)
str(ITP)

#----------------------------------------#
# PREPROCESSING:
#----------------------------------------#
library(dplyr)
ITP <- ITP %>%
  mutate(
    Brocha = as.character(Brocha),  
    BrochaSN = as.character(BrochaSN),
    OrdenFabricacion = factor(OrdenFabricacion),  
    PartNumber = factor(PartNumber), 
    Maquina = as.character(Maquina),
    TpoIndexador = as.character(TpoIndexador),
    Utillaje = as.character(Utillaje),
    
    NBrochasHSS = as.integer(NBrochasHSS),
    NDiscos = as.integer(NDiscos),
    NUsos = as.integer(NUsos),
    USDutchman = as.integer(USDutchman),
    NUso = as.integer(NUso),
    NDisco = as.integer(NDisco),
    
    DUMMY = as.logical(DUMMY),
    Dutchman = as.logical(Dutchman),
    
    FBrochado = as.POSIXct(FBrochado, format = "%Y-%m-%d %H:%M:%S", tz = "CET"),
    
    XC = as.numeric(XC),
    ZC = as.numeric(ZC),
    BC = as.numeric(BC),
    CC = as.numeric(CC),
    XCMM = as.numeric(XCMM),
    ZCMM = as.numeric(ZCMM),
    BCMM = as.numeric(BCMM),
    CCMM = as.numeric(CCMM)
  )
str(ITP)

# Na´s processing
preProcValues <- preProcess(ITP, method = "medianImpute")
ITP <- predict(preProcValues, ITP)

colSums(is.na(ITP)) 

#----------------------------------------#
# SPLITING the data in train and test set:
#----------------------------------------#
set.seed(333)

# (75% train, 25% test)
spl <- createDataPartition(ITP$XCMM, p = 0.75, list = FALSE)  # For deciding which target variable are we going to use for the partition we will review their distributions
# ---
library(ggplot2)
library(gridExtra)
windows()

plot_xc <- ggplot(ITP, aes(x = XCMM)) + 
  geom_histogram(bins = 30, fill = "blue", color = "black") + 
  ggtitle("Distribución de XCMM")

plot_zc <- ggplot(ITP, aes(x = ZCMM)) + 
  geom_histogram(bins = 30, fill = "green", color = "black") + 
  ggtitle("Distribución de ZCMM")

plot_bc <- ggplot(ITP, aes(x = BCMM)) + 
  geom_histogram(bins = 30, fill = "red", color = "black") + 
  ggtitle("Distribución de BCMM")

plot_cc <- ggplot(ITP, aes(x = CCMM)) + 
  geom_histogram(bins = 30, fill = "purple", color = "black") + 
  ggtitle("Distribución de CCMM")

grid.arrange(plot_xc, plot_zc, plot_bc, plot_cc, nrow = 2, ncol = 2)
# ---
train_set <- ITP[spl, ]
test_set <- ITP[-spl, ]

# ... (To be continued in a few lines)
#----------------------------------------#
# FEATURE ENGINEERING (during this process, changes are applied to the variable formats, to later run the models it is essential to re-run the initial formatting of ITP)
#----------------------------------------#
# Correlation Matrix
carga("corrplot")
carga("Hmisc")
library(corrplot)
library(Hmisc)
View(ITP)

numeric_features <- sapply(ITP, is.numeric)
ITP_numeric <- ITP[, numeric_features]

# Corr matrix
cor <- rcorr(as.matrix(ITP_numeric)) 
M <- cor$r
p_mat <- cor$P

# Ploting
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(M, method = "color", col = col(200),  
         type = "upper", order = "hclust", 
         addCoef.col = "black", # Añadir coeficiente de correlación
         tl.col = "darkblue", tl.srt = 45, # Color y rotación de las etiquetas
         p.mat = p_mat, sig.level = 0.05,  # Combinar con nivel de significancia
         diag = FALSE # Ocultar la diagonal principal
)

# High correlation values
w <- which(abs(M) > 0.8 & row(M) < t(row(M)), arr.ind = TRUE)
high_cor_var <- matrix(colnames(M)[w], ncol = 2)
print("Variables with high correlation (greater than 0.8):")
print(high_cor_var)


## LASSO REGRESSION
library(glmnet)
ITP <- ITP %>%
  mutate(
    Brocha = as.numeric(as.factor(Brocha)),
    BrochaSN = as.numeric(as.factor(BrochaSN)),
    OrdenFabricacion = as.numeric(as.factor(OrdenFabricacion)),
    PartNumber = as.numeric(as.factor(PartNumber)),
    Maquina = as.numeric(as.factor(Maquina)),
    TpoIndexador = as.numeric(as.factor(TpoIndexador)),
    Utillaje = as.numeric(as.factor(Utillaje))
  )

ITP <- ITP %>%
  mutate(
    FBrochado = as.numeric(difftime(FBrochado, min(FBrochado, na.rm = TRUE), units = "days")),
    DUMMY = as.numeric(DUMMY),
    Dutchman = as.numeric(Dutchman)
  )

predictor_vars <- c("Brocha", "BrochaSN", "OrdenFabricacion", "PartNumber",
                    "Maquina", "TpoIndexador", "Utillaje", "NBrochasHSS",
                    "NDiscos", "NUsos", "USDutchman", "NUso", "NDisco", 
                    "FBrochado", "DUMMY", "Dutchman", "XC", "ZC", "BC", "CC")

target_vars <- c("XCMM", "ZCMM", "BCMM", "CCMM")

# Loop to perform LASSO for each target variable
for (target in target_vars) {
  cat("\nPerforming LASSO Regression for target variable:", target, "\n")
  
  y <- as.matrix(ITP[[target]]) 
  x <- as.matrix(scale(ITP[, predictor_vars]))  
  
  set.seed(123)
  cv_model <- cv.glmnet(x, y, alpha = 1)
  
  best_lambda <- cv_model$lambda.min
  cat("Best lambda for", target, ":", best_lambda, "\n")
  cat("Test MSE at best lambda:", cv_model$cvm[which(cv_model$lambda == best_lambda)], "\n")
  
  best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
  
  cat("Coefficients for", target, ":\n")
  print(coef(best_model))
  
  plot(cv_model, main = paste("LASSO Cross-validation for", target))
}



# RANDOM FOREST VARIABLE IMPORTANCE
library(randomForest)
set.seed(76)

ITP <- ITP %>%
  mutate(
    OrdenFabricacion = as.character(OrdenFabricacion),
    PartNumber = as.character(PartNumber)
  )

target_variables <- c("XCMM", "ZCMM", "BCMM", "CCMM")

# Loop through each target variable and fit Random Forest model
for (target in target_variables) {
  
  features <- setdiff(names(ITP), target_variables)
  
  ITP_features <- ITP[, features]
  ITP_target <- ITP[[target]]
  
  formula <- as.formula(paste("ITP_target ~ ."))
  
  rf_model <- randomForest(
    formula,
    data = cbind(ITP_features, ITP_target),
    importance = TRUE,
    ntree = 100
  )
  
  print(paste("Random Forest Model for:", target))
  print(rf_model)
  
  print(paste("Variable Importance for:", target))
  importance_values <- importance(rf_model)
  print(importance_values)
  
  varImpPlot(rf_model, main = paste("Variable Importance for:", target))
}


# !! Apply the fist ITP variable formatting !!


# ... (spliting the data process)
# So here are our variables based on the previous RF analysis
features <- c("Brocha", "PartNumber", "OrdenFabricacion", "FBrochado", 
              "Utillaje", "Maquina", "TpoIndexador", "NUso", "NDisco")
labels <- c('XCMM', 'ZCMM', 'BCMM', 'CCMM')


train_set$PartNumber <- factor(train_set$PartNumber)
head(train_set$PartNumber)
train_set$OrdenFabricacion <- factor(train_set$OrdenFabricacion)
head(train_set$OrdenFabricacion)

X_train <- train_set[, features]
y_train <- train_set[, labels]
X_test <- train_set[, features]
y_test <- train_set[, labels]

# Set PartNumber and OrderManufacturing levels in the test set
X_test$PartNumber <- factor(X_test$PartNumber, levels = levels(X_train$PartNumber))
X_test$OrdenFabricacion <- factor(X_test$OrdenFabricacion, levels = levels(X_train$OrdenFabricacion))

cat("Niveles no coincidentes en PartNumber:", 
    setdiff(levels(X_test$PartNumber), levels(X_train$PartNumber)), "\n")
cat("Niveles no coincidentes en OrdenFabricacion:", 
    setdiff(levels(X_test$OrdenFabricacion), levels(X_train$OrdenFabricacion)), "\n")

# Verify the dimensions of the data sets
cat("Dimensiones de X_train:", dim(X_train), "\n")
cat("Dimensiones de y_train:", dim(y_train), "\n")
cat("Dimensiones de X_test:", dim(X_test), "\n")
cat("Dimensiones de y_test:", dim(y_test), "\n")

# Check for NA values in datasets
cat("Número de NA en X_train:", sum(is.na(X_train)), "\n")
cat("Número de NA en y_train:", sum(is.na(y_train)), "\n")
cat("Número de NA en X_test:", sum(is.na(X_test)), "\n")
cat("Número de NA en y_test:", sum(is.na(y_test)), "\n")

str(X_train)
str(y_train)
str(X_test)
str(X_train)

#----------------------------------------#
# TUNING+RESAMPLING
#----------------------------------------#
# LINEAR REGRESSION
#----------------------------------------#

library(doParallel)

# Parallelization configuration (this is in order to be faster during the train stage, it is optional)
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

formulas <- lapply(labels, function(label) {
  as.formula(paste(label, "~", paste(features, collapse = "+")))
}) # (another method)

# Resampling
control <- trainControl(method = "repeatedcv", number = 5, repeats = 3, verboseIter = TRUE, allowParallel = TRUE)

# Results
results_linear <- list()

# Train
set.seed(123)
for (label in labels) {
  # Create a dataframe for each target variable
  train_data <- cbind(X_train, target = y_train[[label]])
  
  formula <- as.formula(paste("target ~", paste(features, collapse = "+")))
  
  # Train
  model <- train(
    formula,
    data = train_data,
    method = "lm",
    trControl = control,
    tuneLength = 10
  )
  
  results_linear[[label]] <- model
}

# Results
cat("\n--- Linear Regression Model Results ---\n")
for (label in labels) {
  test_data <- cbind(X_test, target = y_test[[label]])
  
  predictions <- predict(results_linear[[label]], newdata = test_data)
  
  resample_results <- postResample(predictions, test_data$target)
  
  max_error <- max(abs(predictions - test_data$target), na.rm = TRUE)
  
  cat("\nResultados de predicción para la variable objetivo:", label, "\n")
  cat("RMSE:", resample_results[1], "\n")
  cat("R-squared:", resample_results[2], "\n")
  cat("MAE:", resample_results[3], "\n")
  cat("Error Máximo:", max_error, "\n")
}

# Stop the cluster
stopCluster(cl)
registerDoSEQ()


#----------------------------------------#
# KNN
#----------------------------------------#
library(doParallel)

# Parallelization configuration
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Resampling
control <- trainControl(method = "repeatedcv", number = 5, repeats = 3, verboseIter = TRUE, allowParallel = TRUE)

# Results
results_knn <- list()

# Entrenar un modelo KNN para cada una de las variables objetivo
set.seed(123)
cat("\n--- Entrenando Modelos KNN ---\n")
for (label in labels) {
  train_data <- cbind(X_train, target = y_train[[label]])
 
  formula <- as.formula(paste("target ~", paste(features, collapse = "+")))
  
  model <- train(
    formula,
    data = train_data,
    method = "knn",
    trControl = control,
    tuneLength = 5
  )
  
  results_knn[[label]] <- model
  
  # Progress printing (in case is too long)
  cat("\nModelo KNN entrenado para la variable objetivo:", label, "\n")
}

cat("\n--- KNN Model Results ---\n")
for (label in labels) {
  test_data <- cbind(X_test, target = y_test[[label]])
  
  predictions <- predict(results_knn[[label]], newdata = test_data)
  
  resample_results <- postResample(predictions, test_data$target)
  
  max_error <- max(abs(predictions - test_data$target), na.rm = TRUE)
  
  cat("\nResultados de predicción para la variable objetivo:", label, "\n")
  cat("RMSE:", resample_results[1], "\n")
  cat("R-squared:", resample_results[2], "\n")
  cat("MAE:", resample_results[3], "\n")
  cat("Error Máximo:", max_error, "\n")
}

# Stop cluster
stopCluster(cl)
registerDoSEQ()

#----------------------------------------#
# RANDOM FOREST (with some tunning)
#----------------------------------------#
library(ranger)

num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Group infrequent levels
group_rare_levels <- function(data, column, threshold = 10) {
  rare_levels <- names(table(data[[column]]))[table(data[[column]]) < threshold]
  data[[column]] <- as.character(data[[column]])
  data[[column]][data[[column]] %in% rare_levels] <- "Otros"
  data[[column]] <- as.factor(data[[column]])
  return(data)
}

# Grouping rare levels into categorical variables
X_train <- group_rare_levels(X_train, "PartNumber")
X_train <- group_rare_levels(X_train, "Utillaje")
X_train <- group_rare_levels(X_train, "Brocha")
X_test <- group_rare_levels(X_test, "PartNumber")
X_test <- group_rare_levels(X_test, "Utillaje")
X_test <- group_rare_levels(X_test, "Brocha")

# Adjust the hyperparameter grid
tunegrid <- expand.grid(
  .mtry = c(3, 5, 7),
  .splitrule = c("variance"),
  .min.node.size = c(1, 3)
)

# Resampling
control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Results
results_rf <- list()

# Train
set.seed(123)
cat("\n--- Training Random Fores Models ---\n")
for (label in labels) {
  cat("\nTraining for:", label, "\n")
  
  train_data <- cbind(X_train, target = y_train[[label]])
  formula <- as.formula(paste("target ~", paste(features, collapse = "+")))
  
  model <- train(
    formula,
    data = train_data,
    method = "ranger",
    trControl = control,
    tuneGrid = tunegrid,
    num.trees = 2000,
    importance = "permutation"
  )
  
  results_rf[[label]] <- model
}

# Results
cat("\n--- Final Results ---\n")
for (label in labels) {
  test_data <- cbind(X_test, target = y_test[[label]])
  predictions <- predict(results_rf[[label]], newdata = test_data)
  
  resample_results <- postResample(predictions, test_data$target)
  max_error <- max(abs(predictions - test_data$target), na.rm = TRUE)
  
  cat("\nVariable:", label, "\n")
  cat("RMSE:", resample_results[1], "\n")
  cat("R-squared:", resample_results[2], "\n")
  cat("MAE:", resample_results[3], "\n")
  cat("Error Máximo:", max_error, "\n")
}

# Stop cluster
stopCluster(cl)
registerDoSEQ()


#----------------------------------------#
# XGBOOST
#----------------------------------------#
# For this we need some changes in our datasets in order to be train the model
library(xgboost)

num_cores <- detectCores() - 1  
cl <- makeCluster(num_cores)
registerDoParallel(cl)

convert_to_numeric <- function(df) {
  df <- df %>%
    mutate(
      Utillaje = as.numeric(as.factor(Utillaje)),
      Maquina = as.numeric(as.factor(Maquina)),
      Brocha = as.numeric(as.factor(Brocha)),
      TpoIndexador = as.numeric(as.factor(TpoIndexador)),
      OrdenFabricacion = as.numeric(OrdenFabricacion),  
      PartNumber = as.numeric(PartNumber)
    )
  return(df)
}

X_train <- convert_to_numeric(X_train)
X_test <- convert_to_numeric(X_test)

X_train$FBrochado <- as.numeric(difftime(X_train$FBrochado, min(X_train$FBrochado), units = "days"))
X_test$FBrochado <- as.numeric(difftime(X_test$FBrochado, min(X_train$FBrochado), units = "days"))

cat("NA en X_train:", sum(is.na(X_train)), "\n")
cat("NA en X_test:", sum(is.na(X_test)), "\n")

# Resampling
Control <- trainControl(
  method = "repeatedcv",
  number = 3,  # Número de folds
  repeats = 2,  # Repeticiones
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Grid XGBoost
tune_grid <- expand.grid(
  nrounds = seq(100, 300, by = 100),
  eta = c(0.01, 0.1),
  max_depth = c(4, 6),
  gamma = c(0, 1),
  colsample_bytree = c(0.8, 1.0),
  min_child_weight = c(1, 3),
  subsample = c(0.9, 1.0)
)

results_xgb <- list()


# TRAINING
# ---------------------------------------- #

set.seed(123)
cat("\n--- Entrenando Modelos XGBoost ---\n")
for (label in labels) {
  # Preparar datos de entrada para XGBoost
  input_x <- as.matrix(X_train)
  input_y <- y_train[[label]]
  
  # Entrenar el modelo XGBoost
  model <- train(
    x = input_x, 
    y = input_y,
    method = "xgbTree",
    trControl = Control,
    tuneGrid = tune_grid,
    verbose = TRUE,
    verbosity = 0
  )
  
  # Guardar el modelo en la lista de resultados
  results_xgb[[label]] <- model
  
  # Progreso
  cat("\nModelo XGBoost entrenado para la variable objetivo:", label, "\n")
}

# EVALUATION
# ---------------------------------------- #

cat("\n--- XGBoost Model Results ---\n")
for (label in labels) {
  test_x <- as.matrix(X_test)
  test_y <- y_test[[label]]
  
  predictions <- predict(results_xgb[[label]], newdata = test_x)
  
  resample_results <- postResample(predictions, test_y)
  
  max_error <- max(abs(predictions - test_y), na.rm = TRUE)
  
  cat("\nResultados de predicción para la variable objetivo:", label, "\n")
  cat("RMSE:", resample_results[1], "\n")
  cat("R-squared:", resample_results[2], "\n")
  cat("MAE:", resample_results[3], "\n")
  cat("Error Máximo:", max_error, "\n")
}

# Stop cluster
stopCluster(cl)
registerDoSEQ() 


# *****************IMPORTANT*****************
# In this project, we decided to proceed with Random Forest & XGBOOST as our chosen algorithm. However, the basic model used in class was insufficient to achieve our performance objectives, particularly in meeting the strict requirements for maximum error and predictive accuracy. 
# Therefore, we opted to implement a more advanced and fine-tuned.
# While maintaining the preprocessing pipeline, feature engineering, and evaluation metrics, (with minor adjustments to variable formatting).
# the main focus of our effort was to enhance the model complexity and parameter optimization.
# The updated and more robust code for the model is in the RandomForestXGB.
# *******************************************

