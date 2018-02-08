#Getting Data
NoShowData = read.csv(file.choose())
head(NoShowData)
table(NoShowData)

#Remove PatientID, AppointmentID columns
NoShowDataUpdated = NoShowData[,c(-1,-2)]
NoShowDataUpdated = NoShowDataUpdated[NoShowDataUpdated$Age > 0,]

#Convert columns to factors
NoShowDataUpdated$Scholarship = as.factor(NoShowDataUpdated$Scholarship)
NoShowDataUpdated$Hipertension = as.factor(NoShowDataUpdated$Hipertension)
NoShowDataUpdated$Diabetes = as.factor(NoShowDataUpdated$Diabetes)
NoShowDataUpdated$Alcoholism = as.factor(NoShowDataUpdated$Alcoholism)
NoShowDataUpdated$Handcap = as.factor(NoShowDataUpdated$Handcap)
NoShowDataUpdated$SMS_received = as.factor(NoShowDataUpdated$SMS_received)

#convert date columns into character
NoShowDataUpdated$ScheduledDay = as.character(NoShowDataUpdated$ScheduledDay)
NoShowDataUpdated$AppointmentDay = as.character(NoShowDataUpdated$AppointmentDay)

#Cleaning up date columns
NoShowDataUpdated$ScheduledDay = gsub("(.*)T.*","\\1",NoShowDataUpdated$ScheduledDay)
NoShowDataUpdated$AppointmentDay = gsub("(.*)T.*","\\1",NoShowDataUpdated$AppointmentDay)

library(lubridate)
#Extracting date month and year from appointment and schedule day columns
NoShowDataUpdated$AppointmentMonth = month(as.POSIXlt(NoShowDataUpdated$AppointmentDay, format="%Y-%m-%d"))
NoShowDataUpdated$AppointmentMonth = as.factor(NoShowDataUpdated$AppointmentMonth)
NoShowDataUpdated$AppointmentYear = year(as.POSIXlt(NoShowDataUpdated$AppointmentDay, format="%Y-%m-%d"))
NoShowDataUpdated$AppointmentYear = as.factor(NoShowDataUpdated$AppointmentYear)
NoShowDataUpdated$AppointmentDate = day(as.POSIXlt(NoShowDataUpdated$AppointmentDay, format="%Y-%m-%d"))
NoShowDataUpdated$AppointmentDate = as.factor(NoShowDataUpdated$AppointmentDate)

NoShowDataUpdated$ScheduledMonth = month(as.POSIXlt(NoShowDataUpdated$ScheduledDay, format="%Y-%m-%d"))
NoShowDataUpdated$ScheduledMonth = as.factor(NoShowDataUpdated$ScheduledMonth)
NoShowDataUpdated$ScheduledYear = year(as.POSIXlt(NoShowDataUpdated$ScheduledDay, format="%Y-%m-%d"))
NoShowDataUpdated$ScheduledYear = as.factor(NoShowDataUpdated$ScheduledYear)
NoShowDataUpdated$ScheduledDate = day(as.POSIXlt(NoShowDataUpdated$ScheduledDay, format="%Y-%m-%d"))
NoShowDataUpdated$ScheduledDate = as.factor(NoShowDataUpdated$ScheduledDate)

#Calculate difference between Scheduled and Appointment dates
NoShowDataUpdated$ScheduledDay = date(as.POSIXlt(NoShowDataUpdated$ScheduledDay, format="%Y-%m-%d"))
NoShowDataUpdated$AppointmentDay = date(as.POSIXlt(NoShowDataUpdated$AppointmentDay, format="%Y-%m-%d"))

NoShowDataUpdated$DateDifference = difftime(NoShowDataUpdated$AppointmentDay, NoShowDataUpdated$ScheduledDay, units = "days")
NoShowDataUpdated$DateDifference = as.character(NoShowDataUpdated$DateDifference)
NoShowDataUpdated$DateDifference = gsub(" days","",NoShowDataUpdated$DateDifference)
head(NoShowDataUpdated)

#Creating BalancedData
NoData = NoShowDataUpdated[NoShowDataUpdated$No.show == "No",]
RandNoData = NoData[sample(nrow(NoData), 23000),]
YesData = NoShowDataUpdated[NoShowDataUpdated$No.show == "Yes",]

BalancedData = rbind(YesData,RandNoData)

#removing negative values and values >1
BalancedData = BalancedData[BalancedData$Handcap %in% c(0,1),]
BalancedData = BalancedData[BalancedData$DateDifference >=0,]

ShuffledBalancedData = BalancedData[sample(nrow(BalancedData)),]

####################### Naive Bayes and Logistic ################################
library(caret)
library(e1071)

indexes = sample(1:nrow(ShuffledBalancedData), size=0.5*nrow(ShuffledBalancedData))
trainData = ShuffledBalancedData[-indexes,]
testData = ShuffledBalancedData[indexes,]
x = testData[,-12]

fitNB = naiveBayes(No.show ~ Age + Hipertension + SMS_received + Scholarship, data = trainData)
summary(fitNB)

predictNB = predict(fitNB, x)
confusionMatrix(testData$No.show, predictNB)

#Logistic
NS.LR = glm(No.show ~ Age + Handcap + SMS_received + Scholarship, data = trainData, family = binomial(link = "logit"))
summary(NS.LR)
predictNS.LR = predict(NS.LR, x, type = "response")
table(testData$No.show, predictNS.LR > 0.5)

####################### LASSO ##################################
library(lars)
library(glmnet)

#Cleaning part 2
BalancedDataLasso = BalancedData

#Gender to 0(F) and 1(M)
BalancedDataLasso$Gender = as.character(BalancedDataLasso$Gender)
BalancedDataLasso$Gender[BalancedDataLasso$Gender == "M"] = 1
BalancedDataLasso$Gender[BalancedDataLasso$Gender == "F"] = 0
BalancedDataLasso$Gender = as.factor(BalancedDataLasso$Gender)

#No show to 0(NO) and 1(Yes)
BalancedDataLasso$No.show = as.character(BalancedDataLasso$No.show)
BalancedDataLasso$No.show[BalancedDataLasso$No.show == "Yes"] = 1
BalancedDataLasso$No.show[BalancedDataLasso$No.show == "No"] = 0
BalancedDataLasso$No.show = as.factor(BalancedDataLasso$No.show)

BalancedDataLasso = BalancedDataLasso[,-c(2,3,5,13,14,15,16,17,18)]

BalancedDataLasso$Gender = as.numeric(BalancedDataLasso$Gender)
BalancedDataLasso$Age = as.numeric(BalancedDataLasso$Age)
BalancedDataLasso$Scholarship = as.numeric(BalancedDataLasso$Scholarship)
BalancedDataLasso$Hipertension = as.numeric(BalancedDataLasso$Diabetes)
BalancedDataLasso$Alcoholism = as.numeric(BalancedDataLasso$Alcoholism)
BalancedDataLasso$Handcap = as.numeric(BalancedDataLasso$Handcap)
BalancedDataLasso$SMS_received = as.numeric(BalancedDataLasso$SMS_received)
BalancedDataLasso$No.show = as.numeric(BalancedDataLasso$No.show)
BalancedDataLasso$Diabetes = as.numeric(BalancedDataLasso$Diabetes)
BalancedDataLasso$DateDifference = as.numeric(BalancedDataLasso$DateDifference)

BalancedDataLasso$Gender[BalancedDataLasso$Gender == 1] = 0
BalancedDataLasso$Gender[BalancedDataLasso$Gender == 2] = 1

BalancedDataLasso$Age[BalancedDataLasso$Age == 1] = 0
BalancedDataLasso$Age[BalancedDataLasso$Age == 2] = 1

BalancedDataLasso$Scholarship[BalancedDataLasso$Scholarship == 1] = 0
BalancedDataLasso$Scholarship[BalancedDataLasso$Scholarship == 2] = 1

BalancedDataLasso$Hipertension[BalancedDataLasso$Hipertension == 1] = 0
BalancedDataLasso$Hipertension[BalancedDataLasso$Hipertension == 2] = 1

BalancedDataLasso$Diabetes[BalancedDataLasso$Diabetes == 1] = 0
BalancedDataLasso$Diabetes[BalancedDataLasso$Diabetes == 2] = 1

BalancedDataLasso$Alcoholism[BalancedDataLasso$Alcoholism == 1] = 0
BalancedDataLasso$Alcoholism[BalancedDataLasso$Alcoholism == 2] = 1

BalancedDataLasso$Handcap[BalancedDataLasso$Handcap == 1] = 0
BalancedDataLasso$Handcap[BalancedDataLasso$Handcap == 2] = 1

BalancedDataLasso$SMS_received[BalancedDataLasso$SMS_received == 1] = 0
BalancedDataLasso$SMS_received[BalancedDataLasso$SMS_received == 2] = 1

BalancedDataLasso$No.show[BalancedDataLasso$No.show == 1] = 0
BalancedDataLasso$No.show[BalancedDataLasso$No.show == 2] = 1

ShuffledBalancedDataLasso = BalancedDataLasso[sample(nrow(BalancedDataLasso)),]

x1 = model.matrix(No.show~., ShuffledBalancedDataLasso)[,-1]
y = ShuffledBalancedDataLasso$No.show

#Lasso
lasso.mod <- glmnet(x1, y, alpha=1, nlambda=100, lambda.min.ratio=0.0001)
lasso.mod$lambda[25]
coef(lasso.mod)[,25]

set.seed(1)
cv.out <- cv.glmnet(x1, y, alpha=1, nlambda=100, lambda.min.ratio=0.0001)
plot(cv.out)

best.lambda <- cv.out$lambda.min
best.lambda

predict(lasso.mod, s=best.lambda, type="coefficients")[1:10, ]
#Age,Scholarship,Hipertension,Alcoholism,SMS(without datediff)
#Scholarship,Hipertension,Alcoholism,sms / gender,age,handcap, datediff

#Ridge
ridge.mod <- glmnet(x1, y, alpha=0, nlambda=100, lambda.min.ratio=0.0001)
ridge.mod$lambda[25]
coef(ridge.mod)[,25]

set.seed(1)
cv.out <- cv.glmnet(x1, y, alpha=0, nlambda=100, lambda.min.ratio=0.0001)
plot(cv.out)

best.lambda <- cv.out$lambda.min
best.lambda

predict(ridge.mod, s=best.lambda, type="coefficients")[1:9, ]

######## Regression after Lasso/ Ridge ###############
p = ShuffledBalancedDataLasso[,-9]
NS.LR = glm(No.show ~ . , data = ShuffledBalancedDataLasso, family = binomial(link = "logit"))
summary(NS.LR)
predictNS.LR = predict(NS.LR, p, type = "response")
table(BalancedDataLasso$No.show, predictNS.LR > 0.5)

NS.LR = lm(No.show~.,data = ShuffledBalancedDataLasso)
summary(NS.LR)

cor(ShuffledBalancedDataLasso)

############ PCA ##################################

NoShoWDataPCA = ShuffledBalancedDataLasso

df = NoShoWDataPCA[,c(1:8,10)]
pc = prcomp(df, scale. = TRUE)
pc$sdev
head(pc$rotation)
head(pc$x)

library(FactoMineR)
library(factoextra)
# apply PCA
pca3 = PCA(df, graph = FALSE)

# matrix with eigenvalues
pca3$eig
head(pca3$ind$coord)

#plot pca
fviz_eig(pc)

#Regression using PCA
dfPCA = as.data.frame(pc$x)
head(dfPCA)
head(NoShoWDataPCA)
dfPCA$No.show = NoShoWDataPCA[,9]

LR.PCA = lm(No.show ~ PC1+ PC2+ PC3 + PC4, data = dfPCA)
summary(LR.PCA)


MF_Age <- ggplot(BalancedData,aes(x=Age)) + geom_bar(color="blue", fill="white")+facet_grid(~Gender)+ theme_minimal()
MF_Age
