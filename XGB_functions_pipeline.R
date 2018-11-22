####################################################
#   Convert to xgb matrix form
####################################################
toXgbMatrix.class <- function( Xmat, yval ){
  if ( is.null(yval) ){
    return( xgb.DMatrix(data = data.matrix(Xmat)) )
  }
  else{
    return( xgb.DMatrix(data = data.matrix(Xmat), label = (yval)) )
  }
}
toXgbMatrix.reg <- function( Xmat, yval ){
  if ( is.null(yval) ){
    return( xgb.DMatrix(data = data.matrix(Xmat)) )
  }
  else{
    return( xgb.DMatrix(data = data.matrix(Xmat), label = log1p(yval)) )
  }
}
toXgbMatrix <- function( Xmat, yval, class.type="na" ){
  if ( class.type == "regression" ){
    return(toXgbMatrix.reg( Xmat, yval ))
  }
  else if( class.type == "classification"){
    return(toXgbMatrix.class( Xmat, yval ))
  }
  else{
    print( "regression / classification" )
    break
  }
}

####################################################
#   Gridsearch Function using CV
####################################################
# grid search function
xgb_gridsearch <- function(pc, xgbMatrix, class.type, nclass, nfolds){
  ####################################################
  # XGB cv gridsearch
  # parameters: early stopping = 200
  ####################################################
  params.reg <- list(booster = "gbtree",
                     objective = "reg:linear",
                     eval_metric = "rmse",
                     eta = pc$eta,
                     gamma = pc$gamma,
                     alpha = pc$alpha,
                     lambda = pc$lambda,
                     max_depth = pc$max_depth,
                     min_child_weight = pc$min_child_weight,
                     subsample = pc$subsample,
                     colsample_bytree = pc$colsample_bytree,
                     nrounds = pc$nrounds)
  
  params.class <- list(booster = "gbtree",
                       objective = "multi:softmax",
                       eval_metric = "mlogloss",
                       num_class = nclass,
                       eta = pc$eta,
                       gamma = pc$gamma,
                       alpha = pc$alpha,
                       lambda = pc$lambda,
                       max_depth = pc$max_depth,
                       min_child_weight = pc$min_child_weight,
                       subsample = pc$subsample,
                       colsample_bytree = pc$colsample_bytree,
                       nrounds = pc$nrounds)
  
  if ( class.type == "regression" ){
    params <- params.reg
  }
  else if( class.type == "classification"){
    params <- params.class
  }
  else{
    print( "regression / classification" )
    break
  }
  cv <- xgb.cv(params = params,
               data = xgbMatrix,
               nfold = nfolds,
               nrounds = params$nrounds,
               early_stopping_rounds = 200,
               print_every_n = 2000)
  return(cv)
}

xgb_GridToBest <- function(train_X, train_y, parameter_choices, 
                           class.type, nclass, nfolds){
  ### Split train and validation
  train_index <- sample(x=c(1:nrow(train_X)), 0.8*nrow(train_X))
  dtrain_X <- train_X[train_index, ]
  dtrain_y <- train_y[train_index]
  
  dval_X <- train_X[-train_index, ]
  dval_y <- train_y[-train_index]
  
  train.xgb <- toXgbMatrix(train_X, train_y, class.type)
  dtrain.xgb <- toXgbMatrix(dtrain_X, dtrain_y, class.type) 
  dval.xgb <- toXgbMatrix(dval_X, dval_y, class.type)
  cols <- colnames(dtrain_X)
  
  print("*****************************************************")
  print(paste0("<<<<<<   ", class.type, "   >>>>>>"))
  print(paste0("TOTAL number of parameter choices: ", nrow(parameter_choices)))
  cv_rmse_mat <- matrix(nrow=nrow(parameter_choices), ncol=6)
  colnames(cv_rmse_mat) <- c('pc_i',
                             'best_iter',
                             'train_rmse_mean',
                             'train_rmse_std',
                             'test_rmse_mean',
                             'test_rmse_std')
  
  # Excution
  for ( i in (1:nrow(parameter_choices)) ){
    pc <- parameter_choices[i,]
    # print("*****************************************************")
    # cat(sprintf("---   PARAMS: choice %d\n", i))
    # print(pc)
    xgb_cv <- xgb_gridsearch(pc, dtrain.xgb, class.type, nclass, nfolds)
    
    
    eval_log <- as.matrix(xgb_cv$evaluation_log[xgb_cv$best_iteration,])
    cv_rmse_mat[i,1] <- i
    cv_rmse_mat[i,2:6] <- eval_log
    # save_loc <- "C:/Users/User/Downloads/Kaggle/google_analytics/xgb_gridsearch/"
    # saveRDS(list(params = xgb_cv$params, evaluation_log = xgb_cv$evaluation_log),
    #         file = sprintf(paste0(save_loc,"xgb_gridSearch_cv-%d.rds"), i))
    rm(xgb_cv, eval_log)
  }
  
  cv_rmse_df <- as.data.frame(cv_rmse_mat)
  best_pos <- cv_rmse_df[which( cv_rmse_df$test_rmse_mean == min(cv_rmse_df$test_rmse_mean) ), ]$pc_i
  best_pc <- parameter_choices[best_pos,]
  
  # print("*****************************************************")
  # print(paste("< Best Parameter Choice: >"))
  # print(best_pc)
  # print("*****************************************************")
  ####################################################
  # XGB model
  # parameters: nrounds = 2000
  #             early stopping = 200
  ####################################################
  p.reg <- list(booster = "gbtree",
            objective = "reg:linear",
            eval_metric = "rmse",
            eta = best_pc$eta,
            gamma = best_pc$gamma,
            alpha = best_pc$alpha,
            lambda = best_pc$lambda,
            max_depth = best_pc$max_depth,
            min_child_weight = best_pc$min_child_weight,
            subsample = best_pc$subsample,
            colsample_bytree = best_pc$colsample_bytree,
            nrounds=best_pc$nrounds)
  
  p.class <- list(booster = "gbtree",
            objective = "multi:softmax",
            eval_metric = "mlogloss",
            num_class = nclass,
            eta = best_pc$eta,
            gamma = best_pc$gamma,
            alpha = best_pc$alpha,
            lambda = best_pc$lambda,
            max_depth = best_pc$max_depth,
            min_child_weight = best_pc$min_child_weight,
            subsample = best_pc$subsample,
            colsample_bytree = best_pc$colsample_bytree,
            nrounds=best_pc$nrounds)
  if ( class.type == "regression" ){
    p <- p.reg
  }
  else if( class.type == "classification"){
    p <- p.class
  }
  else{
    print( "regression / classification" )
    break
  }
  best_xgb <- xgb.train(params = p,
                        data = dtrain.xgb, 
                        nrounds = p$nrounds, 
                        watchlist = list(val = dval.xgb), 
                        print_every_n = 2000, 
                        early_stopping_rounds = 200)
  
  # importance plot
  # xgb.importance(cols, model = best_xgb) %>%
  #   xgb.plot.importance(top_n = 25)
  return(best_xgb) 
}
