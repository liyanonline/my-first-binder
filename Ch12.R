# Clear plots
if(!is.null(dev.list())) dev.off()
# Clean workspace
rm(list=ls())
# Clear console
cat("\014") 
# setwd("E:/SSPU/Courses/Courses/BI/Wiley.Data.Mining.for.Business.Analytics.Concepts.Techniques.and.Applications.in.R.1118879368/codes")


####Table 12.1
# install.packages("DiscriMiner")
library(DiscriMiner)
mowers.df <- read.csv("RidingMowers.csv")
da.reg <- linDA(mowers.df[,1:2], mowers.df[,3])
da.reg$functions



#### Table 12.2

da.reg <- linDA(mowers.df[,1:2], mowers.df[,3])
# compute probabilities manually (below); or, use lda() in package MASS with predict()
propensity.owner <- exp(da.reg$scores[,2])/(exp(da.reg$scores[,1])+exp(da.reg$scores[,2]))
data.frame(Actual=mowers.df$Ownership, 
           da.reg$classification, da.reg$scores, propensity.owner=propensity.owner)




#### Table 12.3

library(DiscriMiner)
library(caret)

accidents.df <-  read.csv("Accidents.csv")
lda.reg <- linDA(accidents.df[,1:10], accidents.df[,11])
lda.reg$functions
confusionMatrix(lda.reg$classification, factor(accidents.df$MAX_SEV) )



#### Table 12.4

propensity <- exp(lda.reg$scores[,1:3])/
  (exp(lda.reg$scores[,1])+exp(lda.reg$scores[,2])+exp(lda.reg$scores[,3]))

res <- data.frame(Classification = lda.reg$classification, 
                  Actual = accidents.df$MAX_SEV,
                  Score = round(da.reg$scores,2), 
                  Propensity = round(propensity,2))
head(res)

