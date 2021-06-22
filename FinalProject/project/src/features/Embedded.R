rm(list = ls())
library(data.table)
library(caret)
library(Metrics)
library(xgboost)
library(gtools)
library(dplyr)
library(Rtsne)

set.seed(9001)

#Reading the files
test_emb <- fread("./project/volume/data/raw/test_emb.csv")
train_emb<- fread("./project/volume/data/raw/train_emb.csv")

#I will now combine both the embeddings
master<-rbind(train_emb,test_emb)


#Now apply pca and tsne on master
#applied 3 tsnes

tsne<-Rtsne(master,perplexity=30,
            check_duplicates=F)
tsne_table1<-data.table(tsne$Y)

tsne2<-Rtsne(master,perplexity=65,
             check_duplicates=F)
tsne_table2<-data.table(tsne2$Y)

tsne3<-Rtsne(master,perplexity=90,
             check_duplicates=F)
tsne_table3<-data.table(tsne3$Y)

pca<-prcomp(master)
summary(pca)
pca_dt<-data.table(unclass(pca)$x)

#Combine all the tsne tables and pca table in data
new<-cbind(tsne_table1,tsne_table2,tsne_table3,pca_dt)

#Splitting test and train
train_emb<-new[1:200,]
test_emb<-new[201:20755,]

fwrite(train_emb,"./project/volume/data/interim/train_emb.csv")
fwrite(test_emb,"./project/volume/data/interim/test_emb.csv")


