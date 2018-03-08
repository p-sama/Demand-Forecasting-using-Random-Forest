#################### 442 - Advance Statistics - Final Project - Demand Forecasting #####################

# Set the working directory
setwd("~/Desktop/Classes/442-Advance Statistics/FinalProject")

# Importing the required libraries
library(dplyr)
library(ggplot2)
library(randomForest)
library(party)
library(reshape2)
library(repr)

# To replicate the results
set.seed(1)

# Reading the datasets
train <- read.csv("train.csv")
store <- read.csv("store.csv")

# Merge to get the store info into datasets
train <- merge(train,store)

# Let's get a quick glimpse on the dataset
glimpse(train)
summary(train)

# There are some NAs in the competition distance column, so let us impute that with the median
train$CompetitionDistance[is.na(train$CompetitionDistance)] <- median(train$CompetitionDistance, na.rm=TRUE)
summary(train$CompetitionDistance)

# Removing the stores that were closed
train <- train[which(train$Open == '1'),]

# Considering only the obs with sales > 0
train <- train[which(train$Sales > 0),]

# Extracting the elements of the date column
train$month <- as.integer(format(as.Date(train$Date), "%m"))
train$year <- as.integer(format(as.Date(train$Date), "%Y"))
train$day <- as.integer(format(as.Date(train$Date), "%d"))

# Converting the categorical integer data to factor 
train$Store <- as.factor(train$Store)
train$DayOfWeek <- as.factor(train$DayOfWeek)
train$Open <- as.factor(train$Open)
train$Promo <- as.factor(train$Promo)
train$StateHoliday <- as.factor(ifelse(train$StateHoliday == '0', 0, 1))
train$SchoolHoliday <- as.factor(train$SchoolHoliday)
train$month <- as.factor(train$month)
train$year <- as.factor(train$year)

# Exploratory Data Analyses & Visualization
options(repr.plot.width=4, repr.plot.height=3)
# Distribution of Sales
ggplot(data=train, aes(train$Sales)) + 
  geom_histogram(bins=50, col="black", fill="green", alpha = .5) + 
  labs(title="Sales Distribution") +
  labs(x="Sales", y="Count")

options(repr.plot.width=4, repr.plot.height=3)
# Distribution of Customers
ggplot(data=train, aes(train$Customers)) + 
  geom_histogram(bins=50, col="black", fill="blue", alpha = .5) + 
  labs(title="#Customers Distribution") +
  labs(x="Customers", y="Count")

options(repr.plot.width=6, repr.plot.height=3)
# Line chart to show sales over time
ggplot(train, aes(x = as.Date(Date), y = Sales, color = factor(year))) + 
  geom_smooth() +
  labs(title="Sales over time") +
  labs(x="", y="Sales") +
  guides(color=FALSE)

options(repr.plot.width=6, repr.plot.height=3)
# Sales over days of a month
ggplot(train, aes(x = day, y = Sales)) + 
  geom_point(col="salmon") +
  labs(title="Sales over days of a month") +
  labs(x="", y="Sales")

temp <- train %>%
  group_by(Store) %>%
  summarise(avgSales = mean(Sales)) %>%
  as.data.frame
options(repr.plot.width=10, repr.plot.height=4)
# Distribution of Sales by Store ID
ggplot(data=temp, aes(x=temp$Store, y=temp$avgSales)) + 
  geom_bar(stat="identity",col="black", fill="red", alpha = .3) + 
  labs(title="Avg Sales by Stores") +
  labs(x="Store", y="Avg Sales")

options(repr.plot.width=5, repr.plot.height=3)
# Distribution by day of the week
ggplot(train, aes(x = factor(DayOfWeek), y = Sales, fill=DayOfWeek)) +
  geom_boxplot(color = "black") +
  labs(title="Sales by Day Of Week") +
  labs(x="Day of week", y="Sales") +
  guides(fill=FALSE)

options(repr.plot.width=4, repr.plot.height=3)
# Distribution by Store type
ggplot(train, aes(x = factor(StoreType), y = Sales, fill=StoreType)) +
  geom_boxplot(color = "black") +
  labs(title="Sales by Store Type") +
  labs(x="Store Type", y="Sales") +
  guides(fill=FALSE)

options(repr.plot.width=4, repr.plot.height=3)
# Distribution by Assortment
ggplot(train, aes(x = factor(Assortment), y = Sales, fill=Assortment)) +
  geom_boxplot(color = "black") +
  labs(title="Sales by Assortment") +
  labs(x="Assortment", y="Sales") +
  guides(fill=FALSE)

options(repr.plot.width=5, repr.plot.height=3)
# Sales vs. Customers
ggplot(train, aes(x = Customers, y = Sales)) + 
  geom_point(col="seagreen") +
  labs(title="Sales vs. #Customers") +
  labs(x="#Customers", y="Sales")

options(repr.plot.width=5, repr.plot.height=3)
# Sales vs. Competition Distance
ggplot(train, aes(x = CompetitionDistance, y = Sales)) + 
  geom_point(col="plum1") +
  labs(title="Sales vs. Competition Distance") +
  labs(x="Competition Distance", y="Sales")

options(repr.plot.width=4, repr.plot.height=3)
# Distribution of sales by Promotion
ggplot(train, aes(x = factor(Promo), y = Sales)) + 
  geom_jitter(alpha = 0.1) +
  geom_boxplot(color = "seashell", outlier.colour = NA, fill = NA) +
  labs(title="Sales Distribution by Promotion") +
  labs(x="", y="Sales")

options(repr.plot.width=4, repr.plot.height=3)
# Distribution of sales by State Holiday
ggplot(train, aes(x = factor(StateHoliday), y = Sales)) + 
  geom_jitter(alpha = 0.1) +
  geom_boxplot(color = "orange", outlier.colour = NA, fill = NA) +
  labs(title="Sales Distribution by State Holiday") +
  labs(x="", y="Sales")

options(repr.plot.width=4, repr.plot.height=3)
# Distribution of sales by School Holiday
ggplot(train, aes(x = factor(SchoolHoliday), y = Sales)) + 
  geom_jitter(alpha = 0.1) +
  geom_boxplot(color = "turquoise", outlier.colour = NA, fill = NA) +
  labs(title="Sales Distribution by School Holiday") +
  labs(x="", y="Sales")

options(repr.plot.width=7, repr.plot.height=3)
# Creating a correlation heat map for the numeric vars
nums <- sapply(train, is.numeric)
cormat <- round(cor(train[,nums]),2)
melted_cormat <- melt(cormat)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()

# Removing the date column as required elements have been extracted
train <- train[,-c(3)]

# Some feature engineering
# Even though sales will vary from store to store, let's use average sales per store as another variable
storeAvgSales <- train %>%
  group_by(Store, month) %>%
  summarise(store_avg_salespm = mean(Sales))
# No. of customers itself can be a response variable to predict, so will use avg cust per mth per store
storeAvgCust <- train %>%
  group_by(Store, month) %>%
  summarise(store_avg_custpm = mean(Customers))

# Merging them back to the main dataset
train <- merge(train, storeAvgSales, by=c("Store","month"))
train <- merge(train, storeAvgCust, by=c("Store","month"))

head(train)

# Sampling a small subset to train due to resource limitations
n = nrow(train)
trainIndex = sample(1:n, size = 50000, replace=FALSE)
train = train[trainIndex ,]
summary(train) 

# Let's split the dataset into train and test - 70-30
n = nrow(train)
trainIndex = sample(1:n, size = round(0.7*n), replace=FALSE)
train = train[trainIndex ,]
test = train[-trainIndex ,]

# Extracting the candidate variables to build the model
feature.names <- names(train)[c(2:3,7:12,14:16)]
feature.names

# Building the Random Forest model
model_rf <- randomForest(train[,feature.names], 
                         train$Sales,
                         mtry=4,
                         ntree=20,
                         do.trace=TRUE)

# Assessing the model performance
print("model stats:");model_rf
print("Training RMSE:");print(sqrt(mean(model_rf$mse)))
options(repr.plot.width=6, repr.plot.height=5)
varImpPlot(model_rf, main="Relative Variable Importance")
plot(model_rf, main="MSE vs. Number for trees")
pred_tr <- predict(model_rf, train[,feature.names])
print("Training RMSE:");print(sqrt(mean((train$Sales-pred_tr)^2)))
print("Training MAPE:");print(mean(abs((train$Sales-pred_tr)/train$Sales) * 100))

options(repr.plot.width=10, repr.plot.height=6)
# Plotting a simple conditional tree to demonstrate the Random Forest internal splitting
demoT <- ctree(Sales ~ Promo + DayOfWeek, data=train)                                    
plot(demoT, type="simple")

# Prediction using Test data
summary(test)
pred <- predict(model_rf, test[,feature.names])
print("Test RMSE:");print(sqrt(mean((test$Sales-pred)^2)))
print("Test MAPE:");print(mean(abs((test$Sales-pred)/test$Sales) * 100))



