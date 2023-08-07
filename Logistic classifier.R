library(lattice)
library(ggplot2)
library(caret)
library(e1071)
library(foreign)
library(survival)
library(MASS)
library(nnet)
library(epiDisplay)
library(pROC)

rm(list = ls()) 
setwd('/Users/fanmingxuan/Desktop')
data<-read.csv('DAE_SCAD_selected.csv',header=T)
# 共线性诊断
#XX<-cor(data[,-42])
#kappa(XX,exact = TRUE)
n_repeat=100
auc_val=vector()
f1_scores=vector()
accuracy=vector()
for (i in 1:n_repeat){
# 划分训练集与测试集
  train_sub = sample(nrow(data),8/10*nrow(data))
  train_data = data[train_sub,]
  test_data =data[-train_sub,]
# 模型构建
  model<-glm(label~.,data=train_data,family = binomial)
  summary(model)
# 模型预测
  pre_logistic<-as.numeric(predict(model,newdata = data,type = "response")>0.5)
# 模型检验
  conMat<-confusionMatrix(factor(pre_logistic),factor(data$label),mode = "everything",positive="1")
  accuracy[i]=conMat[["overall"]][["Accuracy"]]
#logistic.display(model) #输出OR值

#roc1<-roc(test_data$Prolapse,pre_logistic,plot=TRUE, print.thres=TRUE, print.auc=TRUE,levels = c(0,1),direction = "<")

  auc=roc(data$label, pre_logistic)$auc
  auc_val[i]=auc
  
  f1_score=conMat[["byClass"]][["F1"]]
  f1_scores[i]=f1_score
}
mean(accuracy)
sd(accuracy)
mean(f1_scores)
sd(f1_scores)
mean(auc_val)
sd(auc_val)

