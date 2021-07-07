setwd("~/Downloads")
library(rpart.plot)
#Importing the data
data = read.csv("data.csv")



###############PERFORMING EDA#################

#Interpreting the structure of the data set
str(data)

####Total number 0f 96 variables.
#All variables are numerical


#Calculating the total missing values in the data set
miss.val=sum(sapply(data, function(x) 
  sum(length(which(is.na(x))))))

print("Totlal Number of missing values:"); miss.val

####There are no null values in the dataset


####Top 20 Important variables data-set
top.data = data.frame(data$Bankrupt., data$Net.Income.to.Total.Assets,
                      data$ROA.B..before.interest.and.depreciation.after.tax,
                      data$ROA.B..before.interest.and.depreciation.after.tax,
                      data$ROA.C..before.interest.and.depreciation.before.interest,
                      data$Net.worth.Assets, data$Debt.ratio..,
                      data$Persistent.EPS.in.the.Last.Four.Seasons,
                      data$Retained.Earnings.to.Total.Assets,
                      data$Net.profit.before.tax.Paid.in.capital,
                      data$Per.Share.Net.profit.before.tax..Yuan...,
                      data$Current.Liability.to.Assets,
                      data$Working.Capital.to.Total.Assets,
                      data$Net.Income.to.Stockholder.s.Equity,
                      data$Borrowing.dependency,
                      data$Current.Liability.to.Current.Assets,
                      data$Liability.to.Equity,
                      data$Net.Value.Per.Share..A.,
                      data$Net.Value.Per.Share..B.,
                      data$Net.Value.Per.Share..C.)
##Statistics for top 20 variables
dataStats.top = data.frame(  mean=sapply(top.data, mean, na.rm = TRUE), 
                         median=sapply(top.data, median, na.rm = TRUE), 
                         sd=sapply(top.data, sd, na.rm = TRUE), 
                         min=sapply(top.data, min, na.rm = TRUE), 
                         max=sapply(top.data, max, na.rm = TRUE),
                         '25%' = sapply( top.data , quantile , probs = 0.25 , na.rm = TRUE ),
                         '50%' = sapply( top.data , quantile , probs = 0.50 , na.rm = TRUE ),
                         '75%' = sapply( top.data , quantile , probs = 0.50 , na.rm = TRUE ),
                         miss.val=sapply(top.data, function(x) 
                           sum(length(which(is.na(x)))))) 
dataStats.top

##Correlation matrix for top 20 variables
cor.mat.top <- cor(top.data)
library(corrplot)
corrplot(cor.mat.top, type = "upper", tl.pos = "td",
         method = "color", tl.cex = 0.5, tl.col = 'black',
         order = "hclust", diag = FALSE)

#Calculating the statistics of all the variables:
dataStats = data.frame(  mean=sapply(data, mean, na.rm = TRUE), 
                         median=sapply(data, median, na.rm = TRUE), 
                         sd=sapply(data, sd, na.rm = TRUE), 
                         min=sapply(data, min, na.rm = TRUE), 
                         max=sapply(data, max, na.rm = TRUE),
                         '25%' = sapply( data , quantile , probs = 0.25 , na.rm = TRUE ),
                         '50%' = sapply( data , quantile , probs = 0.50 , na.rm = TRUE ),
                         '75%' = sapply( data , quantile , probs = 0.50 , na.rm = TRUE ),
                         miss.val=sapply(data, function(x) 
                           sum(length(which(is.na(x)))))) 
dataStats

#Checking, if the data is distributed equally for the target variable Bankrupt?
n= nrow(data)
t = table(data$Bankrupt.)

balDf = data.frame("Bankrupt_Companies" = 100*(length(which(data$Bankrupt.==1))/n),
                    "Safe_Companies" = 100*(length(which(data$Bankrupt.==0))/n)) 
row.names(balDf) = "Percentage"
balDf

##Plotting the distribution of Target Variable
barplot(table(data$Bankrupt.),col=rainbow(2),
        main="Frequency of Bankrutcy",
        xlab="Bankruptcy Tag",
        ylab="Number of Companies")

# we can see that data is highly unbalanced.

# plotting the histogram for all the variables
#TODO
library(tidyr)
library(ggplot2)


ggplot(gather(data[,c(1:48)]), aes(value)) + 
  geom_histogram(bins = 10, fill = "light blue", colour = "black") + 
  facet_wrap(~key, scales = 'free_x')

ggplot(gather(data[,c(49:96)]), aes(value)) + 
  geom_histogram(bins = 10, fill = "orange", colour = "black") + 
  facet_wrap(~key, scales = 'free_x') 


   

#Plotting the correlation of each variable with target variable
library(lares)

indVar = data[,c(2:96)]
corr_var(data,Bankrupt.,top=96)

#Plotting boxplots of each variables:
#TODO
par(mfcol = c(1,1))
par(mar=c(4,18,1,1))

boxplot(scale(data[,c(2:48)]), las=2, horizontal = TRUE, 
        col = c('powderblue', 'mistyrose', 'orange'), ylim = range(-5:5))

boxplot(scale(data[,c(49:96)]), las=2, horizontal = TRUE, 
        col = c('indianred2', 'palegreen1', 'light blue'), ylim = range(-5:5))



#Correlation of all the variables
cor.mat <- cor(data)
library(corrplot)
corrplot(cor.mat, method="color")


###############REMOVING OUTLIERS################
#identifying the outliers
outliers <- boxplot(data$Operating.Gross.Margin, plot=FALSE)$out
length(outliers)

outlier_norm <- function(x){
  qntile <- quantile(x, probs=c(.25, .75))
  caps <- quantile(x, probs=c(.05, .95))
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qntile[1] - H)] <- caps[1]
  x[x > (qntile[2] + H)] <- caps[2]
  return(x)
}

testOut = data.frame(data)

for(i in 1:ncol(testOut)) {       
  testOut[,i] = outlier_norm(testOut[,i])
  outliers <- boxplot(testOut[,i], plot=FALSE)$out
  print(length(outliers))    
}

outliers <- boxplot(testOut, plot=FALSE)$out
length(outliers)



#TODO 
#plotting boxplots bankrupt and important variables
#plotting the distribution of important varibales




#Cleaned Boxplots and distributions
#TODO


##################Splitting the data#######################

set.seed(1)
numberOfRows <- nrow(data)
train.index <- sample(numberOfRows, numberOfRows*0.6)
train.df <- data[train.index, ]
temp.df <- data[-train.index, ]
numberOfRows <- nrow(temp.df)
train.index <- sample(numberOfRows, numberOfRows*0.5)
##Training Data Set. 
train.df
##Validation Data Set
valid.df <- temp.df[train.index, ]
valid.df
#Testing Data Set
test.df <- temp.df[-train.index, ]
test.df

##Removing the temporary data set used for partitioning the data.
rm(temp.df)



library(randomForest)
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(gains)
library(caret)
library(ROCR)
library(pROC)


####################Decision Treee##############
cv.ct <- rpart(Bankrupt. ~ ., data = train.df, method = "class",  
               maxdepth = 30, minsplit = 1, cp = 0)

# use printcp() to print the table. 
printcp(cv.ct)
prp(cv.ct, type = 1, extra = 1, split.font = 1, varlen = -10)  

#predicting using validation data
ct.pred <- predict(cv.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(ct.pred, as.factor(valid.df$Bankrupt.))



##Pruning the Decision Tree####
pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

printcp(pruned.ct)

#plot the best fitting tree
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10,
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))  

prune.pred <- predict(pruned.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(prune.pred, as.factor(valid.df$Bankrupt.))


##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = prune.pred, "Label" = as.factor(valid.df$Bankrupt.))
t.df

pred <- prediction(as.numeric(t.df$Predicted), as.numeric(t.df$Label))
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)


##Plotting ROC and AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")

plot(sens.ci, type="bars")
d<-coords(roc_step,"best","threshold",transpose=T)
d

##Important Variables
v = varImp(cv.ct)

##TODO
plot(varImp(cv.ct))






##############Logistic Regression###############
gen.logit.reg <- glm(Bankrupt. ~., data = train.df, family = "binomial") 
options(scipen=999)
summary(gen.logit.reg)
gen.logit.reg.pred <- predict(gen.logit.reg, newdata = valid.df, type="response")
confusionMatrix(table(predict(gen.logit.reg, newdata = valid.df, 
                              type="response") >= 0.1, valid.df$Bankrupt.==1))

##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = gen.logit.reg.pred, "Label" = as.factor(valid.df$Bankrupt.))
t.df

pred <- prediction(t.df$Predicted, t.df$Label)
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)


##Plotting ROC and AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="tomato2")

plot(sens.ci, type="bars")
d<-coords(roc_step,"best","threshold",transpose=T)
d

##Important Variables 
v = varImp(logit.reg)
plot(v,scale = TRUE)





###############Random Forest###############
rf <- randomForest(as.factor(Bankrupt.) ~ ., data = train.df, 
                   ntree = 750, mtry = 50, nodesize = 1, importance = TRUE, type = 'classification') 

#plot the variables by order of importance
varImpPlot(rf, type = 1)

#create a confusion matrix
valid.df$Bankrupt. <- factor(valid.df$Bankrupt.)
rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, valid.df$Bankrupt.)


##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = rf.pred, "Label" = as.factor(valid.df$Bankrupt.))
t.df

pred <- prediction(as.numeric(t.df$Predicted), as.numeric(t.df$Label))
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)

##Plotting ROC & AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="yellowgreen")

plot(sens.ci, type="bars")
d<-coords(roc_step,"best","threshold",transpose=T)
d

##Important Variables 
v = varImp(rf)
varImpPlot(rf, type = 1)





#######################SMOTE ANALYSIS###########################

##Let's see if using SMOTE as oversampling technique
#can help us improving the performance.

library(DMwR)
train.df$Bankrupt. = as.factor(train.df$Bankrupt.)
balanced.data <- SMOTE(Bankrupt. ~., 
                       train.df, perc.over = 1000, perc.under = 110)


barplot(table(balanced.data$Bankrupt.),col=rainbow(2),
        main="Frequency of Bankrutcy",
        xlab="Bankruptcy Tag",
        ylab="Number of Companies")





#############Decision Tree########
cv.ct <- rpart(Bankrupt. ~ ., data = balanced.data, method = "class", cp = 0.001,  
               maxdepth = 30, minsplit = 1)

# use printcp() to print the table. 
printcp(cv.ct)

##Pruning the Decision Tree####
pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

printcp(pruned.ct)

#plot the best fitting tree
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10,
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))  

prune.pred <- predict(pruned.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(prune.pred, as.factor(valid.df$Bankrupt.))


##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = prune.pred, "Label" = as.factor(valid.df$Bankrupt.))
t.df

pred <- prediction(as.numeric(t.df$Predicted), as.numeric(t.df$Label))
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)


##Plotting ROC and AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")

plot(sens.ci, type="bars")
d<-coords(pROC_obj,"best","threshold",transpose=T)
d







##############Logistic Regression###############
logit.reg <- glm(Bankrupt. ~., data = balanced.data, family = "binomial") 
options(scipen=999)
summary(logit.reg)
logit.reg.pred <- predict(logit.reg, newdata = valid.df, type="response")
confusionMatrix(table(predict(logit.reg, newdata = valid.df, 
                              type="response") >= 0.1, valid.df$Bankrupt.==1))

##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = logit.reg.pred, "Label" = as.factor(valid.df$Bankrupt.))
t.df

pred <- prediction(t.df$Predicted, t.df$Label)
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)


##Plotting ROC and AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="tomato2")

plot(sens.ci, type="bars")
d<-coords(pROC_obj,"best","threshold",transpose=T)
d





############Random Forest#################

###############Random Forest###############
smote.rf <- randomForest(as.factor(Bankrupt.) ~ ., data = balanced.data, 
                   ntree = 750, mtry = 50, nodesize = 1, importance = TRUE, type = 'classification') 

#plot the variables by order of importance
varImpPlot(smote.rf, type = 1)

#create a confusion matrix
valid.df$Bankrupt. <- factor(valid.df$Bankrupt.)
smote.rf.pred <- predict(smote.rf, valid.df)
confusionMatrix(smote.rf.pred, valid.df$Bankrupt.)


##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = smote.rf.pred, "Label" = as.factor(valid.df$Bankrupt.))
t.df

pred <- prediction(as.numeric(t.df$Predicted), as.numeric(t.df$Label))
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)

##Plotting ROC & AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="yellowgreen")

plot(sens.ci, type="bars")
d<-coords(pROC_obj,"best","threshold",transpose=T)
d



##############TRANSFORMING/SCALING DATA##########


scale.train.df = train.df
scale.train.df[,c(2,96)] = scale(scale.train.df[,c(2,96)])
scale.valid.df = valid.df
scale.valid.df[,c(2,96)] = scale(scale.valid.df[,c(2,96)])



##############Logistic Regression###############
logit.reg <- glm(Bankrupt. ~., data = scale.train.df, family = "binomial") 
options(scipen=999)
summary(logit.reg)
logit.reg.pred <- predict(logit.reg, newdata = scale.valid.df, type="response")
confusionMatrix(table(predict(logit.reg, newdata = scale.valid.df, 
                              type="response") >= 0.1, scale.valid.df$Bankrupt.==1))

##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = logit.reg.pred, "Label" = as.factor(scale.valid.df.Bankrupt.))
t.df

pred <- prediction(t.df$Predicted, t.df$Label)
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)


##Plotting ROC and AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="tomato2")

plot(sens.ci, type="bars")
d<-coords(pROC_obj,"best","threshold",transpose=T)
d


#############Decision Tree########
scale.cv.ct <- rpart(Bankrupt. ~ ., data = scale.train.df, method = "class", cp = 0.001,  
               maxdepth = 30, minsplit = 1)

# use printcp() to print the table. 
printcp(scale.cv.ct)

##Pruning the Decision Tree####
scale.pruned.ct <- prune(scale.cv.ct, 
                   cp = scale.cv.ct$cptable[which.min(scale.cv.ct$cptable[,"xerror"]),"CP"])

printcp(scale.pruned.ct)

#plot the best fitting tree
prp(scale.pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10,
    box.col=ifelse(scale.pruned.ct$frame$var == "<leaf>", 'gray', 'white'))  

scale.prune.pred <- predict(scale.pruned.ct, scale.valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(scale.prune.pred, as.factor(scale.valid.df$Bankrupt.))


##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = scale.prune.pred, "Label" = as.factor(scale.valid.df$Bankrupt.))
t.df

pred <- prediction(as.numeric(t.df$Predicted), as.numeric(t.df$Label))
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)


##Plotting ROC and AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")

plot(sens.ci, type="bars")
d<-coords(pROC_obj,"best","threshold",transpose=T)
d


############Random Forest#################

###############Random Forest###############
scale.rf <- randomForest(as.factor(Bankrupt.) ~ ., data = scale.train.df, 
                   ntree = 750, mtry = 50, nodesize = 1, importance = TRUE, type = 'classification') 

#plot the variables by order of importance
varImpPlot(scale.rf, type = 1)

#create a confusion matrix
scale.valid.df$Bankrupt. <- factor(scale.valid.df$Bankrupt.)
scale.rf.pred <- predict(scale.rf, scale.valid.df)
confusionMatrix(scale.rf.pred, scale.valid.df$Bankrupt.)


##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = scale.rf.pred, "Label" = as.factor(scale.valid.df$Bankrupt.))
t.df

pred <- prediction(as.numeric(t.df$Predicted), as.numeric(t.df$Label))
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)

##Plotting ROC & AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="yellowgreen")

plot(sens.ci, type="bars")
d<-coords(pROC_obj,"best","threshold",transpose=T)
d

#########COMPARING 3 BEST MODELS ############

####DECISION TREE

t.df.dec <- data.frame("Predicted" = scale.prune.pred, "Label" = as.factor(scale.valid.df$Bankrupt.))
t.df.dec

pred.decision <- prediction(as.numeric(t.df.dec$Predicted), as.numeric(t.df.dec$Label))
perf.dec <- performance( pred.decision, "tpr", "fpr" )

#####Random Forest

t.df.rand <- data.frame("Predicted" = smote.rf.pred, "Label" = as.factor(valid.df$Bankrupt.))
t.df.rand

pred.random <- prediction(as.numeric(t.df.rand$Predicted), as.numeric(t.df.rand$Label))
perf.rand <- performance( pred.random, "tpr", "fpr" )


#######Logistic Regression

t.df.log <- data.frame("Predicted" = gen.logit.reg.pred, "Label" = as.factor(valid.df$Bankrupt.))
t.df.log

pred.log <- prediction(t.df.log$Predicted, t.df.log$Label)
perf.log <- performance( pred.log, "tpr", "fpr" )

plot( perf.dec, lty = 3, lwd = 3, col = "orangered4")
plot( perf.rand, add = T, lty = 4, lwd = 3, col = "darkslategray4")
plot( perf.log, add = T, lty = 1, lwd = 3, col = "blue4")



legend("bottomright", 
       legend = c("Decision Tree (Scaled)", "Random Forest (SMOTE)", "Logistic Regression"), 
       col = c( "orangered4", "darkslategray4", "blue4"),
       lty = c(3, 4, 1, 4, 1),
       lwd = c(3, 3, 3, 8, 1))


######TESTING THE FINAL MODEL##########

###Logistic Regression
test.gen.logit.reg.pred <- predict(gen.logit.reg, newdata = test.df, type="response")
confusionMatrix(table(predict(gen.logit.reg, newdata = test.df, 
                              type="response") >= 0.1, test.df$Bankrupt.==1))

##Plotting the Performance(True & False Positive Rate)
t.df <- data.frame("Predicted" = test.gen.logit.reg.pred, "Label" = as.factor(test.df$Bankrupt.))
t.df

pred <- prediction(t.df$Predicted, t.df$Label)
perf <- performance( pred, "tpr", "fpr" )
plot( perf, colorize = T)


##Plotting ROC and AUC
pROC_obj <- roc(as.numeric(t.df$Label), as.numeric(t.df$Predicted),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="steelblue4")

plot(sens.ci, type="bars")

##Important Variables 
V = varImp(gen.logit.reg)
ggplot2::ggplot(v, aes(x=reorder(rownames(V),Overall), y=Overall)) +
  geom_point( color="slateblue4", size=3, alpha=0.8)+
  geom_segment( aes(x=rownames(V), xend=rownames(V), y=0, yend=Overall), 
                color='turquoise4') +
  xlab('Variable')+
  ylab('Overall Importance')+
  theme_light() +
  coord_flip()







