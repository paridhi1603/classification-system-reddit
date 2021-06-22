rm(list = ls())
library(data.table)
library(caret)
library(Metrics)
library(xgboost)
library(gtools)
library(ClusterR)
library(dplyr)
library(Rtsne)


#Reads the raw test and train
test <- fread("project/volume/data/raw/test_file.csv")
train <- fread("project/volume/data/raw/train_data.csv")

#reads the embedded test and train
new_train<-fread ('./project/volume/data/interim/train_emb.csv')
new_test<-fread ('./project/volume/data/interim/test_emb.csv')

#melt the train
dat_train <- subset(train, select = -c(2))
dat_train <- melt(dat_train, id.vars =1, variable.name="label")
dat_train <- subset(dat_train, value == 1)
#make num_id
newID <-  as.numeric(gsub(".*?([0-9]+).*", "\\1", dat_train$id)) 
dat_train$num_id<- newID
dat_train$label<-as.numeric(dat_train$label)-1
dat_train<-dat_train[order(num_id),]
dat_train<-subset(dat_train, select=-c(3,4))

#combine melt_train and embedded train
dt<-cbind(dat_train,new_train)
dt<-data.table(dt)

train_label <- dt[,"label"]
train_label<-data.table(train_label)
y.train<-as.numeric(train_label$label)
y.train<-data.table(y.train)
y.train<-sapply(y.train,as.numeric)


dt<-dt[,!"id"]
dt<-dt[,!"label"]
x.train<-data.table(dt)
x.train<-sapply(x.train,as.numeric)


dtrain<-xgb.DMatrix(as.matrix(x.train),label=y.train)

test<-test[,!"text"]
myvar <- c('label')
test[,c(myvar):=0]


new_test<-cbind(test,new_test)
new_test<-data.table(new_test)

test_variable <- new_test$variable
test_variable<-data.table(test_variable)
y.test<-as.numeric(test_variable$test_variable)
y.test<-data.table(y.test)
new_test<-new_test[,!"id"]
new_test<-new_test[,!"label"]
x.test<-data.table(new_test)
x.test<-sapply(x.test,as.numeric)

dtest <- xgb.DMatrix(as.matrix(x.test),missing = NA)

#Run grid search to find the best parameter
searchGridSubCol <- expand.grid(subsample = c(0.75,0.8, 1), 
                                colsample_bytree = c(0.7, 0.8, 1),
                                eta=c(0.1,0.2),
                                
                                gamma=c(0,0.01),
                                max_depth=c(5,6,7,8),
                                min_child_weight=c(1,2,3))
ntrees <- 10000
hyper_param_tune<-NULL
mloglossErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  
  
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentetaRate <- parameterList[["eta"]]
  currentgammaRate <- parameterList[["gamma"]]
  currentmaxDepthRate <- parameterList[["max_depth"]]
  currentMinChildWeightRate <- parameterList[["min_child_weight"]]
  xgb_params<-list(objective = "multi:softprob",
                   eval_metric = "mlogloss",
                   num_class = 10,
                   gamma = currentgammaRate,
                   booster = "gbtree",
                   eta = currentetaRate,
                   max_depth=currentmaxDepthRate,
                   min_child_weight=currentMinChildWeightRate,
                   subsample=currentSubsampleRate,
                   colsample_bytree=currentColsampleRate)
  
  cv_model <- xgb.cv(data =  dtrain, params =xgb_params,nrounds = ntrees, nfold = 7, showsd = TRUE, 
                     verbose = TRUE,early_stopping_rounds = 25)
  
  
  
  best_tree_n<-unclass(cv_model)$best_iteration
  new_row<-data.table(t(xgb_params))
  new_row$best_tree_n<-best_tree_n
  test_error<-unclass(cv_model)$evaluation_log[best_tree_n,]$test_mlogloss_mean
  new_row$test_error<-test_error
  hyper_parm_tune<-rbind(new_row,hyper_param_tune)
  return (hyper_parm_tune)
  
})

output <- mloglossErrorsHyperparameters
df <- do.call("rbind", output)
df <- do.call("rbind", output)
min_error<-min(df$test_error)
param<-df[which.min(df$test_error),]
best_tree<-param$best_tree_n

#using the best parameters from the grid search
new_params <- list(objective = "multi:softprob",
                   eval_metric = "mlogloss",
                   num_class = 10,
                   gamma = param$gamma,
                   booster = "gbtree",
                   eta = param$eta,
                   max_depth=param$max_depth,
                   min_child_weight=param$min_child_weight,
                   subsample=param$subsample,
                   colsample_bytree=param$colsample_bytree,
                   tree_method="hist")
watchlist<-list(train=dtrain)
best_model<-xgb.train(params=new_params,
                      nrounds = best_tree,
                      data=dtrain,
                      watchlist=watchlist,
                      missing=NA,
                      print_every_n=1)


pred <- predict(best_model, newdata = dtest)

pred

submit <- pred

df <- data.frame(matrix(unlist(submit), nrow=20555, byrow=TRUE),stringsAsFactors=TRUE)

df$id <- test$id

names(df)[1] <- "subredditcars"
names(df)[2] <- "subredditCooking"
names(df)[3] <- "subredditMachineLearning"
names(df)[4] <- "subredditmagicTCG"
names(df)[5] <- "subredditpolitics"
names(df)[6] <- "subredditReal_Estate"
names(df)[7] <- "subredditscience"
names(df)[8] <- "subredditStockMarket"
names(df)[9] <- "subreddittravel"
names(df)[10] <- "subredditvideogames"
names(df)[11] <- "id"

fwrite(df,"project/volume/data/processed/submission_final.csv")


















