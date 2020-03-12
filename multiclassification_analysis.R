#####################
# empty the workspace
#####################
rm(list=ls())

#######################
# directory & file name
#######################
### data
data_loc <- 'C:/Users/JinCheol Choi/Desktop/R/Functions/CAL_pipeline/'
code_loc <- 'C:/Users/JinCheol Choi/Desktop/R/Functions/CAL_pipeline/'

# data name
data_tr <- 'training_data.csv'
data_te <- 'test_data.csv'

#############
# pre-setting
#############
res_pos <- 18                     # column index of response variable
numOfCV <- 3                      # numOfCV = k of k-fold cross validation
R <- 5                            # number of runs of the most outer loop
class.type <- "classification"    # analysis type
#class.type <- "regression"
Methods <- c(                     # methods to implement
  # "LASSOMIN", "LASSO1SE",
  "RF",
  "XGB"
  # "SVM/Linear",
  # "SVM_Radial"
  # "SVM_Sigmoid"
)

################
# load functions
################
source(paste0(code_loc, 'required_packages.R'))
source(paste0(code_loc, 'RF_functions_pipeline.R'))
source(paste0(code_loc, 'XGB_functions_pipeline.R'))

##########################
# Read data & modification
##########################
mtrain <- read.csv(paste0(data_loc, data_tr))
# mtest <- read.csv(paste0(data_loc, data_te))
mtrain <- mtrain[1:100, 2:19]
mtrain$Claim.Class <- mtrain$Claim.Class - 1

################################
# initialize MSEP and MSE matirx
################################
evalMetrics <- lossFunction(class.type)
lasso.family.type <- ifelse(class.type=="classification", "multinomial", "gaussian")
predict.response.type <- ifelse(class.type=="classification", "class", "response")
insample.error <- matrix(NA, nrow=R, ncol=length(Methods))
colnames(insample.error) <- Methods
outsample.error <- matrix(NA, nrow=R, ncol=length(Methods))
colnames(outsample.error) <- colnames(insample.error)

#############################
# Random Forest base settings
#############################
rf.best.params <- matrix(NA, nrow=R, ncol=4)
colnames(rf.best.params) <- c("R#", "ntree", "mtry", "nodesize")

p=ncol(mtrain)-1                               # The number of variables
ntree = c(50, 250, 500, 800, 1000)
mtry = seq(1, p, by = 5)                     # The number of variables at each split
nodesize = c(5, seq(10, 150, by = 30))       # Minimum size of terminal nodes
rf.parameter_choices <- expand.grid(ntree=ntree,
                                 mtry=mtry,
                                 nodesize=nodesize)

#######################
# XGB parameter choices
#######################
### create matrix to keep track the best parameters for each round of R
xgb.best.params <- matrix(NA, nrow=R, ncol=14) %>% as.data.frame()
colnames(xgb.best.params) <- c("R#", "booster", "objective", "eval_metric", 
                               "eta", "gamma", "alpha", "lambda", 
                               "max_depth", "min_child_weight", "subsample",
                               "colsample_bytree", "nrounds", "silent")

eta_values              <- c(0.1, 0.3)         # defalut: 0.3
gamma_values            <- c(0, 100)   
alpha_values            <- c(0, 30)            # defalut: 0
lambda_values           <- c(0, 30)            # defalut: 0
max_depth_values        <- c(6, 10)            # defalut: 6
min_child_weight_values <- c(1, 5)             # defalut: 1
subsample_values        <- c(0.5, 1)           # defalut: 1
colsample_bytree_values <- c(0.5, 1)           # defalut: 1

# sets of parameters for grid search
# parameter_choices <- expand.grid(eta = eta_values,
#                                  gamma = gamma_values,
#                                  alpha = alpha_values,
#                                  lambda = lambda_values,
#                                  max_depth = max_depth_values,
#                                  min_child_weight = min_child_weight_values,
#                                  subsample = subsample_values,
#                                  colsample_bytree = colsample_bytree_values
# )
parameter_choices <- expand.grid(eta = 0.3,
                                 gamma = gamma_values[1],
                                 alpha = 0,
                                 lambda = 0,
                                 max_depth = 6,
                                 min_child_weight = 1,
                                 subsample = 1,
                                 colsample_bytree = 1,
                                 nrounds = 10
)

#######################
# SVM parameter choices
#######################
### create matrix to keep track the best parameters for each round of R
svm.radial.best.params <- matrix(NA, nrow=R, ncol=3)
colnames(svm.radial.best.params) <- c("R#", "gamma", "cost")
svm.sigmoid.best.params <- matrix(NA, nrow=R, ncol=4)
colnames(svm.sigmoid.best.params) <- c("R#", "gamma", "cost", "coef0")

svm.sampling <- "cross"      
# gammm = 1/(2*sigma^2)
svm.gamma.values <- c(10^(seq(-6,-4,1)))
svm.cost.values <- c(10^seq(2,5,1))
svm.coef0.values <- c(10^seq(-1,5,5))
svm.gamma.values <- c(0.01,1)
svm.cost.values <- c(1,100)
svm.coef0.values <- c(0.1,1)
svm.tune.control <- list(sampling=svm.sampling, cross=numOfCV)
svm.pc <- list(gamma=svm.gamma.values,
               cost=svm.cost.values,
               coef0=svm.coef0.values) 

################
# main algorithm
################
for(r in 1:R){
  new <- ifelse(runif(n=nrow(mtrain))<=.75, yes=1, no=2)
  train <- mtrain[which(new==1),]               # training set
  test <- mtrain[which(new==2),]                # test set
  
  mtrain.dummies <- predict(onehot(mtrain, max_levels=1000, stringsAsFactors=TRUE), mtrain)
  colnames(mtrain.dummies)=gsub('[( -=)]', '_', colnames(mtrain.dummies))
  
  # test set with dummy vars
  train.X.dummies <- mtrain.dummies[which(new==1), -length(colnames(mtrain.dummies))]
  test.X.dummies <- mtrain.dummies[which(new==2), -length(colnames(mtrain.dummies))]
  train.X.dummies.scaled <- scale(train.X.dummies)
  test.X.dummies.scaled <- scale(test.X.dummies)
  test.X.dummies.scaled[is.na(test.X.dummies.scaled)] <- 0
  
  # divide x and y
  train.y <- train[,res_pos]                          # training set y
  train.X <- train[,-res_pos]                         # training set x
  test.y <- test[,res_pos]                            # test set y
  test.X <- test[,-res_pos]                           # test set x
  
  ##################
  # LASSO / LASSOMIN
  ##################
  if(length(setdiff(c("LASSOMIN", "LASSO1SE"),colnames(outsample.error)))==0){
    cv.lasso.1 <- cv.glmnet(y=changeToFactor(class.type, train.y), 
                            x=train.X.dummies.scaled, 
                            nfolds=numOfCV,
                            family=lasso.family.type)
    predict.tr <- predict(cv.lasso.1, s=cv.lasso.1$lambda.min, newx=train.X.dummies.scaled,
                          type=predict.response.type) %>% as.vector() %>% as.numeric() # predict x1 using x1 lasso outcome
    predict.te <- predict(cv.lasso.1, s=cv.lasso.1$lambda.min, newx=test.X.dummies.scaled,
                          type=predict.response.type) %>% as.vector() %>% as.numeric() # predict x2 using x1 lasso outcome
    insample.error[r, which(Methods=="LASSOMIN")] <- evalMetrics(train.y, predict.tr)
    outsample.error[r, which(Methods=="LASSOMIN")] <- evalMetrics(test.y, predict.te)
    # interim process message
    print(paste0("<LASSOMIN - Done> R:", r))
    
    ##################
    # LASSO / LASSO1SE
    ##################
    predict.tr <- predict(cv.lasso.1, s=cv.lasso.1$lambda.1se, newx=train.X.dummies.scaled,
                          type=predict.response.type) %>% as.vector() %>% as.numeric() # predict x1 using x1 lasso outcome
    predict.te <- predict(cv.lasso.1, s=cv.lasso.1$lambda.1se, newx=test.X.dummies.scaled,
                          type=predict.response.type)  %>% as.numeric() # predict x2 using x1 lasso outcome
    insample.error[r, which(Methods=="LASSO1SE")] <- evalMetrics(train.y, predict.tr)
    outsample.error[r, which(Methods=="LASSO1SE")] <- evalMetrics(test.y, predict.te)
    # interim process message
    print(paste0("<LASSO1SE - Done> R:", r))
  }
  
  ###############
  # Random Forest
  ###############
  if(length(setdiff(c("RF"),colnames(outsample.error)))==0){
    rf.output=randomForestBest(x=train.X, y=changeToFactor(class.type, train.y),
                               pc=rf.parameter_choices,
                               class.type=class.type,
                               nfolds=numOfCV)
    rf.best=rf.output$model
    rf.best.params[r,1] = r
    rf.best.params[r,2:4] = c(rf.output$best_ntree,
                              rf.output$best_mtry,
                              rf.output$best_nodesize)
    predict.tr <- predict(rf.best, newdata=train.X) %>% as.vector() %>% as.numeric()
    predict.te <- predict(rf.best, newdata=test.X) %>% as.vector() %>% as.numeric()
    
    insample.error[r, which(Methods=="RF")] <- evalMetrics(train.y, predict.tr) #sMSE
    outsample.error[r, which(Methods=="RF")] <- evalMetrics(test.y, predict.te) #MSPE
    # interim process message
    print(paste0("<RF - Done> R:", r))
  }
  
  #########
  # XGBoost
  #########
  if(length(setdiff(c("XGB"),colnames(outsample.error)))==0){
    xgb.best <- xgb_GridToBest(train_X=train.X,
                               train_y=train.y, 
                               parameter_choices=parameter_choices, 
                               class.type=class.type,
                               nclass=length(unique(train.y)), 
                               nfolds=numOfCV)
    xgb.best.params[r,1] <- r
    xgb.temp.params <- xgb.best$params %>% as.data.frame()
    col_diff <- setdiff( colnames(xgb.temp.params), colnames(xgb.best.params) )
    rm(xgb.temp.params)
    xgb.best.params[r,2:14] <-
      (xgb.best$params) %>% 
      as.data.frame() %>% 
      select(-col_diff) %>% 
      as.matrix()  %>% 
      t()
    
    predict.tr <- predict(xgb.best, newdata=toXgbMatrix(train.X, NULL ,class.type))
    predict.te <- predict(xgb.best, newdata=toXgbMatrix(test.X, NULL ,class.type))
    
    outsample.error[r, which(Methods=="XGB")] <- evalMetrics(train.y, predict.tr)
    insample.error[r, which(Methods=="XGB")] <- evalMetrics(test.y, predict.te)
    # interim process message
    print(paste0("<XGB - Done> R:", r))
  }
  
  ############
  # SVM_Linear
  ############
  if(length(setdiff(c("SVM_Linear"),colnames(outsample.error)))==0){
    caret.svmLinear.tuning <- train(x=train.X.dummies.scaled,
                                    y=changeToFactor(class.type, train.y),
                                    method="svmLinear",
                                    tuneLength=10,
                                    trControl = trainControl(method = "cv", number=numOfCV))
  }
  ############
  # SVM_Radial
  ############
  if(length(setdiff(c("SVM_Radial"),colnames(outsample.error)))==0){
    caret.svmRadial.tuning <- train(x=train.X.dummies.scaled,
                                    y=changeToFactor(class.type, train.y),
                                    method="svmRadialSigma",
                                    # preProcess=NULL,
                                    scale=FALSE,
                                    tuneLength=10,
                                    trControl = trainControl(method = "cv", number=numOfCV))
    caret.gamma <- 1/(2*(caret.svmRadial.tuning$bestTune$sigma)^2)
    caret.cost <- caret.svmRadial.tuning$bestTune$C
    
    caret.best.svm <- svm(y=changeToFactor(class.type, train.y), 
                          x=train.X.dummies.scaled, scale=FALSE,
                          kernal="radial", 
                          gamma = caret.gamma, 
                          cost = caret.cost) 
    svm.radial.best.params[r, 1] <- r
    svm.radial.best.params[r, 2:3] <- c(gamma=caret.gamma, cost=caret.cost) %>% as.matrix()
    predict.tr <- 
      predict(caret.best.svm, newdata=train.X.dummies.scaled) %>% as.vector() %>% as.numeric()
    predict.te <- 
      predict(caret.best.svm, newdata=test.X.dummies.scaled) %>% as.vector() %>% as.numeric()
    insample.error[r, which(Methods=="SVM_Radial")] <- evalMetrics(train.y, predict.tr)
    outsample.error[r, which(Methods=="SVM_Radial")] <- evalMetrics(test.y, predict.te)
    # interim process message
    print(paste0("<SVM_Raidal - Done> R:", r))
  }
  
  #############
  # SVM_Sigmoid
  #############
  if(length(setdiff(c("SVM_Sigmoid"),colnames(outsample.error)))==0){
    svm.sigmoid <-  tune.svm(y=changeToFactor(class.type, train.y),
                             x=train.X.dummies.scaled, 
                             scale=FALSE,
                             kernal="sigmoid", 
                             gamma = svm.pc$gamma, 
                             cost = svm.pc$cost, 
                             coef0 = svm.pc$coef0,
                             tunecontrol = tune.control(sampling=svm.sampling, cross=numOfCV)) 
    svm.sigmoid.best.params[r, 1] <- r
    svm.sigmoid.best.params[r, 2:4] <- svm.sigmoid$best.parameters %>% as.matrix()
    predict.tr <- 
      predict(svm.sigmoid$best.model, newdata=train.X.dummies.scaled) %>% as.vector() %>% as.numeric()
    predict.te <- 
      predict(svm.sigmoid$best.model, newdata=test.X.dummies.scaled)  %>% as.vector() %>% as.numeric()
    insample.error[r, which(Methods=="SVM_Sigmoid")] <- evalMetrics(train.y, predict.tr)
    outsample.error[r, which(Methods=="SVM_Sigmoid")] <- evalMetrics(test.y, predict.te)
    # interim process message
    print(paste0("<SVM_Sigmoid - Done> R:", r))
  }
}


######################
# Combine the matrices
######################
#################################
# All methods sqrt(MSPE) Box plot
#################################
par(mfrow=c(1,1))
boxplot(sqrt(insample.error), las=2, main="Test Error (sqrt(MSPE))")
boxplot(sqrt(outsample.error), las=2, main="Test Error (sqrt(MSPE))")

##########################################
# All methods rescaled sqrt(MSPE) Box plot
##########################################
apply(X=outsample.error, MARGIN=1, FUN=min)
# Divide all errors for a given split by this minimum
outsample.error.scaled=outsample.error/apply(X=outsample.error, MARGIN=1, FUN=min)
# Box Plot
par(mfrow=c(1,1))
# boxplot(sqrt(outsample.error), las=2, main="Misclassification error \n (sqrt(1 - Correct rate))")
boxplot(sqrt(outsample.error.scaled), las=2, main="Misclassfication error \n re-scaled(sqrt(1 - Correct rate))")


