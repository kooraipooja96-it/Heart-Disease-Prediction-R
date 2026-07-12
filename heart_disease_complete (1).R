#########################################################
# HEART DISEASE ANALYSIS AND PREDICTION USING R
#########################################################

install.packages("ggplot2")
install.packages("dplyr")
install.packages("corrplot")
install.packages("caret")
install.packages("pROC")
install.packages("randomForest")

library(ggplot2)
library(dplyr)
library(corrplot)
library(caret)
library(pROC)
library(randomForest)

heart <- read.csv("heart.csv")

head(heart)
tail(heart)
str(heart)
summary(heart)
dim(heart)
names(heart)

sum(is.na(heart))
sum(duplicated(heart))

heart$sex <- factor(heart$sex, labels=c("Female","Male"))
heart$target <- factor(heart$target, labels=c("No Disease","Disease"))

mean(heart$age)
median(heart$age)
sd(heart$age)
var(heart$age)
IQR(heart$age)

ggplot(heart,aes(age))+
geom_histogram(fill="skyblue",bins=15)+
labs(title="Age Distribution")

ggplot(heart,aes(sex))+
geom_bar(fill="orange")+
labs(title="Gender Distribution")

ggplot(heart,aes(target))+
geom_bar(fill="green")+
labs(title="Heart Disease Distribution")

ggplot(heart,aes(chol))+
geom_histogram(fill="red",bins=20)+
labs(title="Cholesterol Distribution")

ggplot(heart,aes(trestbps))+
geom_histogram(fill="purple",bins=15)+
labs(title="Blood Pressure Distribution")

ggplot(heart,aes(age,chol,color=target))+
geom_point(size=3)+
labs(title="Age vs Cholesterol")

ggplot(heart,aes(target,chol,fill=target))+
geom_boxplot()+
labs(title="Cholesterol vs Heart Disease")

num_data <- heart
num_data$sex <- as.numeric(num_data$sex)
num_data$target <- as.numeric(num_data$target)

cor_matrix <- cor(num_data)
corrplot(cor_matrix,method="color",tl.cex=0.7)

continuous <- c("age","trestbps","chol","thalach","oldpeak")

sapply(heart[continuous], function(x) shapiro.test(x)$p.value)

t.test(age~target,data=heart)
t.test(chol~target,data=heart)
t.test(trestbps~target,data=heart)

chisq.test(table(heart$sex,heart$target))

detect_outliers <- function(x){
Q1 <- quantile(x,0.25)
Q3 <- quantile(x,0.75)
IQR1 <- IQR(x)
lower <- Q1-1.5*IQR1
upper <- Q3+1.5*IQR1
sum(x<lower | x>upper)
}

sapply(heart[continuous],detect_outliers)

write.csv(heart,"clean_heart.csv",row.names=FALSE)

set.seed(123)
split <- createDataPartition(heart$target,p=0.80,list=FALSE)
train <- heart[split,]
test <- heart[-split,]

log_model <- train(target~.,data=train,method="glm",family="binomial")
rf_model <- randomForest(target~.,data=train)

prediction <- predict(log_model,test)
confusionMatrix(prediction,test$target)

prob <- predict(log_model,test,type="prob")
roc_curve <- roc(test$target,prob$Disease)
plot(roc_curve,col="blue",main="ROC Curve")
auc(roc_curve)

importance <- varImp(log_model)
plot(importance)

pred <- data.frame(Actual=test$target,Predicted=prediction)
write.csv(pred,"predictions.csv",row.names=FALSE)

cat("Heart Disease Analysis Completed Successfully")
