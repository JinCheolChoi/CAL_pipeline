####################################################
#   Data preprocessing
#   convert factor to integer -> numeric 
#   ( data.matrix () )
####################################################
tr_te_ann <- tr_te %>% 
  mutate_if(is.factor, as.integer) %>%
  mutate_if(is.integer, as.numeric) %>%
  glimpse()
# rm(tr_te); invisible(gc()) 
glimpse(tr_te)
glimpse(tr_te_ann)

# Feature Scaling
tr_te_ann <- as.data.frame(scale(tr_te_ann))

dtrain1 <- tr_te_ann[tri,]
dte <- tr_te_ann[-tri,]
y <- y[tri]

# revenue > 0
na_pos <- which(is.na(y))
y <- y[-na_pos]
dtrain1 <- dtrain1[-na_pos,]

# split the training and validation dataset
set.seed(2020)
tr_index <- sample(1:nrow(dtrain1), 0.7*nrow(dtrain1))
dtr1 <- dtrain1[tr_index,]
dval1 <- dtrain1[-tr_index,]
ytr <- y[tr_index]
yval <- y[-tr_index]

library(h2o)
h2o.init(nthreads = -1)
tr.h2o <- as.h2o(cbind(dtrain1,y=y))
te.h2o <- as.h2o(cbind(dte))
# dval.h2o <- as.h2o(cbind(dval=dval1, y=yval))
# dtr.h2o <- as.h2o(cbind(dtr=dtr1,y=ytr))

splits <- h2o.splitFrame(tr.h2o, ratios = 0.7, seed = 1)
dtr.h2o <- splits[[1]]
dval.h2o <- splits[[2]]


####################################################
#   GRID SEARCH:
#   Choose candidates for tuning parameters
####################################################
# reference: https://shiring.github.io/machine_learning/2017/03/07/grid_search
#            https://htmlpreview.github.io/?https://github.com/ledell/sldm4-h2o/blob/master/sldm4-deeplearning-h2o.html

# hidden: how many hidden layers and how many nodes per hidden layer the model should learn
# l1      : lets only strong weights survive
# l2      : prevents any single weight from getting too big.
# rho     : similar to prior weight updates
# epsilon : prevents getting stuck in local optima
hyper_params <- list(
  activation = c("Rectifier", 
                 "Maxout",
                 "Tanh"),
                 # "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout"), 
  hidden = list( c(10, 10), c(20, 20), c(50, 50) ),
  epochs = c(50, 100, 200),
  l1 = c(0, 0.00001, 0.0001),
  l2 = c(0, 0.00001, 0.0001)
  # rate = c(0, 01, 0.005),
  # rate_annealing = c(1e-8, 1e-7, 1e-6),
  # rho = c(0.9, 0.95, 0.99),
  # epsilon = c(1e-10, 1e-6, 1e-4),
  # momentum_start = c(0, 0.5),
  # momentum_stable = c(0.99, 0.5, 0),
  # input_dropout_ratio = c(0, 0.1, 0.2)
  # max_w2 = c(10, 100, 1000 )
)
search_criteria <- list(strategy = "RandomDiscrete", 
                        max_models = 100,
                        max_runtime_secs = 900,
                        stopping_tolerance = 0.001,
                        stopping_rounds = 15,
                        seed = 42)

####################################################
#   Excute the grid search
####################################################
ann_grid <- h2o.grid(algorithm = "deeplearning",        # ann 
                    y = 'y',                            # response variable
                    # weights_column = weights,         # weights
                    grid_id = "ann_grid",               # id_name
                    training_frame = dtr.h2o,           # training set
                    validation_frame = dval.h2o,        # validation set
                    nfolds = 5,                         # number of folds                           
                    # fold_assignment = "Stratified",
                    hyper_params = hyper_params,        # tuning parameters 
                    search_criteria = search_criteria,  # seraching criteria
                    seed = 42
)

####################################################
# Performance metrics where smaller is better
# Chossing Best Model
####################################################
sort_options_1 <- c("mean_residual_deviance", "rmse" )
for (sort_by_1 in sort_options_1) {
  grid <- h2o.getGrid("ann_grid", sort_by = sort_by_1, decreasing = FALSE)
  model_ids <- grid@model_ids
  best_model <- h2o.getModel(model_ids[[1]])
  print(("========================================================="))
  print(paste0("best_model_", sort_by_1,": ",grid@model_ids[[1]]))
  print(("========================================================="))
  assign(paste0("best_model_", sort_by_1), best_model)
}  

# h2o.shutdown(prompt = FALSE)

####################################################
# ANN model
####################################################


ann_model <- h2o.deeplearning(y = 'y',
                         training_frame = tr.h2o,
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 50,
                         train_samples_per_iteration = -2)

y_pred <- h2o.predict(model, newdata = as.h2o(te.h2o))
y_pred <- as.data.frame(y_pred) 
mean((y_pred[,1]-yte)^2)


n_ae <- 4
m_ae <- h2o.deeplearning(training_frame = tr.h2o,
                         x = 1:ncol(tr.h2o),
                         autoencoder = T,
                         activation="Rectifier",
                         reproducible = TRUE,
                         seed = 0,
                         sparse = T,
                         standardize = TRUE,
                         hidden = c(32, n_ae, 32),
                         max_w2 = 5,
                         epochs = 25)
tr_ae <- h2o.deepfeatures(m_ae, tr.h2o, layer = 2) %>% as_tibble
te_ae <- h2o.deepfeatures(m_ae, te.h2o, layer = 2) %>% as_tibble

h2o.shutdown(prompt = FALSE)


