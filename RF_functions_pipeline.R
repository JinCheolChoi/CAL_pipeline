##########################################################################
# random forest with best tunning parameters
##########################################################################
randomForestBest = function(x, y, pc, class.type="regression", nfolds=5){
  #pc=rf.parameter_choices
  # x=train.X
  # y=changeToFactor(class.type, train.y)
  
  # Set up matrices to hold results. First two columns are parameter values.
  # Each column after that is a rep.
  RF_Best=c()                                          # Save the index of the best model at each split
  if ( class.type == "regression" ){
    y <- y
  }else if( class.type == "classification" ){
    y <- as.factor(y)
  }else{
    print( "regression / classification" )
    break
  }
  
  ##################
  # cross validation
  ##################
  fold.ind=createFolds(y, k = nfolds)
  
  # create cv mspse matrix that
  CV.MSPE=cbind(pc, matrix(NA, nrow=nrow(pc), ncol=nfolds))
  colnames(CV.MSPE)=c(colnames(pc), paste0("cv.mspe.", 1:nfolds))
  
  for(cv.ind in 1:nfolds){
    # create cv.x and cv.y
    cv.train.X=x[-fold.ind[[cv.ind]],]
    cv.train.y=y[-fold.ind[[cv.ind]]]
    cv.test.X=x[fold.ind[[cv.ind]],]
    cv.test.y=y[fold.ind[[cv.ind]]]
    
    ### Find the best tunnings
    for(i in 1:nrow(pc)){
      rf.best=randomForest(x=cv.train.X, y=cv.train.y,
                           keep.forest=TRUE, importance=TRUE, type = class.type,
                           ntree=pc[i,"ntree"], mtry=pc[i,"mtry"], nodesize=pc[i,"nodesize"])
      
      predict.tr <- predict(rf.best, newdata=cv.train.X) %>% as.vector() %>% as.numeric()
      predict.te <- predict(rf.best, newdata=cv.test.X) %>% as.vector() %>% as.numeric()
      
      CV.MSPE[i, paste0("cv.mspe.", cv.ind)]=evalMetrics(cv.test.y, predict.te) #MSPE
      
      print(paste0("<RF - Tuning Parameter Search> R:", r, " cv:", cv.ind," ntree:", pc[i,"ntree"], 
                   ", mtry:", pc[i,"mtry"],", nodesize:", pc[i,"nodesize"]))
    }
  }
  
  # average cv mspe of tuning parameter combinations
  avg.cv.mspe=apply(CV.MSPE[,paste0("cv.mspe.", 1:nfolds)], 1, mean)
  
  ### Select the best tunnings based on the minimum MSE
  # Save the index of the best model at each split
  best.ind=sample(which(avg.cv.mspe==min(avg.cv.mspe)),1)
  best_ntree=CV.MSPE[best.ind,"ntree"]
  best_mtry=CV.MSPE[best.ind,"mtry"]
  best_nodesize=CV.MSPE[best.ind,"nodesize"]
  
  ########################################
  ## model with the best tuning parameters
  ########################################
  rf.best=randomForest(x=x, y=y,
                       keep.forest=TRUE, importance=TRUE,
                       ntree=best_ntree, mtry=best_mtry, nodesize=best_nodesize)
  
  # save output
  output=list(
    model=rf.best,
    best_ntree=best_ntree,
    best_mtry=best_mtry,
    best_nodesize=best_nodesize
  )
  
  return(output)
}
