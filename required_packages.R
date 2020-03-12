checkpackages<-function(package){
  # Checking the Availability of packages 
  # Installs them.  
  # example usage: checkpackages("gtools")
  if (!package %in% installed.packages()){
    install.packages(package)
  }
  library(package, character.only =T)
}
# https://www.kaggle.com/kailex/r-eda-for-gstore-glm-keras-xgb

# Required Packages
lapply(c("MASS", "nnet", "glmnet", "e1071", "klaR", "car", "class",
         "sm", "rpart.plot", "randomForest", "gbm", "caret", "rpart",
         "mice", "scales", "dplyr", 
         #'ElemStatLearn', 
         "h2o",
         "RColorBrewer", "rattle","data.table",
         "jsonlite", "zoo", "stringr",
         #"lightgbm",
         "Matrix"), checkpackages)

lapply(c("Rmisc",
         "countrycode",
         "highcharter",
         "keras",
         "forecast",
         "tidyverse",
         "magrittr",
         "lubridate",
         "xgboost",
         "ggalluvial",
         "onehot"),checkpackages)

cm_accuracy <- function(actual.y, pred.y){
  temp_table <- table(Actual=actual.y, Prediction=pred.y)
  return(1-sum(diag(temp_table))/sum(temp_table))
}
MSE_calc <- function(actual.y, pred.y){
  return(mean((actual.y-pred.y)^2))
}

lossFunction <- function(class.type){
  if (class.type=="regression"){
    return(MSE_calc)
  }
  else if (class.type=="classification"){
    return(cm_accuracy)
  }
  else{
    print("regression/classification")
  }
}

changeToFactor <- function(class.type, y){
  if (class.type=="regression"){
    y <- as.numeric(y)
  }
  else if (class.type=="classification"){
    y <- as.factor(y)
  }
  else {
    print("regression/classification")
    break
  }
  return(y)
}


