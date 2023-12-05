# Project start

###############################################################################################################################################################

# Table of contents:
    # 0. Setup
    # 1. Pre-processing
    # 2. Randomly selecting 100,000 observations
    # 3. Sentiment analysis
    # 4. Additional data manipulation and cleaning
    # 5. Split into test and training
    # 6. Exploratory and descriptive analysis
    # 7. Modelling

###############################################################################################################################################################
# 0 Setup
###############################################################################################################################################################

#setwd("C:/Users/sofiy/OneDrive/Рабочий стол/Uni/Study/Y3/EC349/Assignment 349")
library(tidyverse)
library(jsonlite)
library(caret)
library(httr)
library(ggplot2)
library(stringr)
library(tidytext)
library(textdata)
library(caTools)
library(syuzhet)
library(tree)
library(rpart)
library(rpart.plot)
library(MASS)
library(randomForest)
library(patchwork)
library(kableExtra)
library(scales)

rm(list=ls())
set.seed(100)

###############################################################################################################################################################
# 1 Pre-processing
###############################################################################################################################################################

# 1.1 Load ##############################################################################################

business <- stream_in(file("yelp_academic_dataset_business.json"))
checkin <- stream_in(file("yelp_academic_dataset_checkin.json"))
tips <- stream_in(file("yelp_academic_dataset_tip.json"))

load(file="yelp_review_small.Rda") #review_data_small
assign('reviews', review_data_small)
rm(review_data_small)

load(file="yelp_user_small.Rda") #user_data_small
assign('users', user_data_small)
rm(user_data_small)


# 1.2 Examine data #######################################################################################

# Checking structure:
for (i in list(business, checkin, tips, reviews, users)) {
  str(i)
} 

# Viewing data:
view(business[1:50,])
view(checkin[1:50,])
view(tips[1:50,])
view(reviews[1:50,])
view(users[1:50,])

# 1.3 Selecting out 'candidate' explanatory variables: ####################################################

# Removing ata sets I will not use:
rm(checkin)
rm(tips)

# Changing variable names in data sets I will use to avoid issses in merged data:

# reviews
reviews <- reviews %>%
  rename(
    r_stars = stars
  )

#users
users <- users[c("user_id", 
                 "review_count", 
                 "average_stars")]
users <- users %>%
  rename(
    u_average_stars = average_stars,
    u_review_count = review_count
    
  )

#business
business <- business[c("business_id",
                       "stars",
                       "review_count")]
business <- business %>%
  rename(
    b_review_count = review_count,
    b_stars = stars
  )


# 1.4 Merging data ########################################################################################

master <- merge(x=reviews, y=users, by="user_id", all.x=TRUE)
master <- merge(x=master, y=business, by="business_id", all.x=TRUE)
master <- master %>% relocate(review_id, .before=business_id)

# Examining merged data:
dim(reviews)
dim(users)
dim(business)

dim(master)
view(master[1:50,])
str(master)

# 1.5 Selecting observations where u_average_stars is not missing ###########################################

# Assessing number of missing values:
master$r_stars <- factor(master$r_stars)
sum(!is.na(master$u_average_stars))/dim(master)[1]*100

# Comparing review star distribution for:
          # observations with missing u_average_stars (master_A)
          # observations with present u_average_stars (master_B)
master_A <- master %>%
  filter(is.na(master$u_average_stars))
master_B <- master %>%
  filter(!is.na(master$u_average_stars))

A <- ggplot(data=master_A, aes(x=r_stars)) + 
  geom_bar() + 
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  scale_y_continuous(labels = label_number()) +
  ggtitle("Missing")

B <- ggplot(data=master_B, aes(x=r_stars)) + 
  geom_bar() + 
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  scale_y_continuous(labels = label_number()) +
  ggtitle("Not missing")

A+B

# Judging from A+B plot, r_star distributions look similar => proceed with master_B as data in use
master_in_use <- master_B

# Examining data in use:
str(master_in_use)
for (colname in colnames(master_in_use)) {
  print(colname)
  print(sum(is.na(master_in_use[[colname]]))/dim(master_in_use)[1]*100)
}

###############################################################################################################################################################
# 2 Randomly selecting 100 000 observations to proceed with (due to computing power issues)
###############################################################################################################################################################

master_sample <- master_in_use[sample(nrow(master_in_use), size = 100000), ]

###############################################################################################################################################################
# 3 Sentiment Analysis
###############################################################################################################################################################

master_sample <- master_sample %>% mutate(line=row_number())

# Fetching sentiment dictionaries:
bing <- get_sentiments("bing")
afinn <- get_sentiments("afinn")

# Transforming text column into a tibble for easier manipulation:
txtib <- tibble(
  line = 1:100000,
  text = master_sample$text
)

# Defining custom stopwords (mainly negations not contained in stop_words):
stop_words$word
custom_stopwords <- stop_words$word[!str_detect(stop_words$word, "no|not|don't|won't|doesn't|isn't")]

# Creating BING sentiment score:
bing_data <- txtib %>% 
  unnest_tokens(bigram, text, token = "ngrams", n=2) %>%
  filter(!is.na(bigram))

bing_data <- bing_data %>% 
  separate(bigram, c("w1","w2"), sep=" ") %>%
  filter(!w1 %in% custom_stopwords) %>%
  filter(!w2 %in% stop_words$word)

bing_data <- bing_data %>% 
  rename(word = w2) %>%
  inner_join(bing) %>%
  mutate(sentiment = if_else(str_detect(w1, "no|not|don't|won't|doesn't|isn't") & sentiment == "positive", "negative", sentiment)) %>%
  mutate(sentiment = if_else(str_detect(w1, "no|not|don't|won't|doesn't|isn't") & sentiment == "negative", "positive", sentiment)) %>%
  count(line, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill=0) %>%
  mutate(bing_sentiment = positive-negative)

# Merging BING score with master:
master_sample <- merge(x=master_sample, y=bing_data, by="line", all.x=TRUE)
sum(is.na(master_sample$bing_sentiment))/dim(master_sample)[1]*100

#Creating AFINN sentiment score:
afinn_data <- txtib %>% 
  unnest_tokens(bigram, text, token = "ngrams", n=2) %>%
  filter(!is.na(bigram))

afinn_data <- afinn_data %>% 
  separate(bigram, c("w1","w2"), sep=" ") %>%
  filter(!w1 %in% custom_stopwords) %>%
  filter(!w2 %in% stop_words$word)

afinn_data <- afinn_data %>% 
  rename(word = w2) %>%
  inner_join(afinn) %>%
  mutate(value = if_else(str_detect(w1, "no|not|don't|won't|doesn't|isn't"), -value, value)) %>%
  group_by(line) %>%
  summarise(afinn_sentiment = sum(value))

# Merging AFINN score with master:
master_sample <- merge(x=master_sample, y=afinn_data, by="line", all.x=TRUE)
sum(!is.na(master_sample$bing_sentiment) | !is.na(master_sample$afinn_sentiment))/dim(master_sample)[1]*100

# Normalising sentiment scores by review length:
master_sample$revlength <- str_count(master_sample$text, " ") + 1
master_sample <- master_sample %>%
  mutate(bing_sentiment = bing_sentiment/revlength) %>%
  mutate(afinn_sentiment = afinn_sentiment/revlength)

###############################################################################################################################################################
# 4 Additional data manipulation and cleaning
###############################################################################################################################################################

# Creating new variable: exclam 
master_sample$exclam <- str_count(master_sample$text, "!")/master_sample$revlength

# Transforming b_stars into factor (as takes values 1, 1.5, 2 ... 5):
str(master_sample)
master_sample <- master_sample %>%
  mutate(b_stars = factor(b_stars))

# View dimensions:
dim(master_sample)

# Dealing with missing values generated for AFINN and BING scores:

master_sample <- master_sample %>%
  mutate(bing_NA = if_else(is.na(bing_sentiment), TRUE, FALSE)) %>%
  mutate(afinn_NA = if_else(is.na(afinn_sentiment), TRUE, FALSE))

master_sample_A <- master_sample %>%
  filter(afinn_NA==TRUE)

master_sample_B <- master_sample %>%
  filter(afinn_NA==FALSE)

AA <- ggplot(data=master_sample_A, aes(x=r_stars)) + 
  geom_bar() + 
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  scale_y_continuous(labels = label_number()) +
  ggtitle("Missing")

BB <- ggplot(data=master_sample_B, aes(x=r_stars)) + 
  geom_bar() + 
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  scale_y_continuous(labels = label_number()) +
  ggtitle("Not missing")

AA+BB

# r_star distributions look similar in both plots => Can proceed with deleting observations with missing variables
master_sample <- master_sample %>%
  filter(afinn_NA==FALSE) %>%
  filter(bing_NA==FALSE)

dim(master_sample)
str(master_sample)


###############################################################################################################################################################
# 5 Split into test and training
###############################################################################################################################################################

master_sample <- na.omit(master_sample)
selector <- sample.split(Y = master_sample$review_id, SplitRatio=0.7)
master_train <- master_sample[selector,]
master_test <- master_sample[!selector,]
nrow(master_train) #37661
nrow(master_test) #16141

###############################################################################################################################################################
# 6 Exploratory analysis for master_sample
###############################################################################################################################################################

# 6.1 Correlation coefficients ##################################################################################################
master_sample_forcorr <- master_sample %>%
  dplyr::select(r_stars, useful, funny, cool, u_review_count, u_average_stars, b_stars, b_review_count, bing_sentiment, afinn_sentiment, revlength, exclam) %>%
  mutate(r_stars = as.numeric(as.character(r_stars))) %>%
  mutate(b_stars = as.numeric(as.character(b_stars)))
str(master_sample_forcorr)
corr_table <- round(cor(master_sample_forcorr),2)
kable(corr_table)

# 6.2 Visualisations ##################################################################################################

# Review stars 
starsplot <- ggplot(data=master_sample) + 
  geom_bar(aes(x=r_stars)) + 
  geom_text(stat = 'count', aes(x=r_stars, label = ..count..), vjust = -0.5, size = 2) +
  ggtitle("Distribution of Star Reviews") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
ggsave("stars.png", plot = starsplot, width = 3, height = 3)

# Review stars against:

# Average user stars
usr_bar = ggplot(data=master_sample) + 
  geom_bar(stat="summary", aes(x=r_stars, y=u_average_stars)) + 
  ggtitle("Bar: Review Stars / Avg. User Stars") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))

usr_violin = ggplot(data=master_sample) + 
  geom_violin(aes(x=r_stars,y=u_average_stars)) + 
  ggtitle("Violin: Review Stars / Avg. User Stars") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
usr_comb = usr_bar + usr_violin
usr_comb
ggsave("usr_comb.png", plot = usr_comb, width = 5, height = 3)

# Business stars
master_sample$b_stars <- as.numeric(master_sample$b_stars)
bsn_bar = ggplot(data=master_sample) + 
  geom_bar(stat="summary", aes(x=r_stars, y=b_stars)) + 
  ggtitle("Bar: Review Stars / Business Stars") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
master_sample$b_stars <- factor(master_sample$b_stars)

bsn_count = ggplot(data=master_sample) + 
  geom_count(aes(x=r_stars,y=b_stars)) + 
  scale_size_area() +
  ggtitle("Review Stars / Business Stars") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
bsn_comb <- bsn_bar + bsn_count

ggsave("business_stars.png", plot = bsn_comb, width = 5, height = 3)

# Useful
master_sample$useful <- factor(master_sample$useful)
useful_count <- ggplot(data=master_sample) +
  geom_count(aes(x=r_stars,y=useful)) + 
  coord_cartesian(ylim = c(0, 5)) +
  scale_size_area() +
  ggtitle("Review Stars / 'Useful'") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
master_sample$useful <- as.numeric(master_sample$useful)

useful_bar = ggplot(data=master_sample, aes(x=r_stars, y=useful)) + 
  geom_bar(stat="summary") +
  ggtitle("Bar: Review Stars / 'Useful'") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
useful_comb = useful_bar + useful_count
useful_comb
ggsave("useful_comb.png", plot = useful_comb, width = 5, height = 3)

# Funny
funny_bar = ggplot(data=master_sample, aes(x=r_stars, y=funny)) + 
  geom_bar(stat="summary") +
  ggtitle("Violin: Review Stars / 'Funny'") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))

master_sample$funny <- factor(master_sample$funny)
funny_count <- ggplot(data=master_sample) +
  geom_count(aes(x=r_stars,y=funny)) + 
  coord_cartesian(ylim = c(0, 5)) +
  scale_size_area() +
  ggtitle("Review Stars / 'Funny'") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4),
        legend.text = element_text(size=6))
master_sample$funny <- as.numeric(master_sample$funny)

funny_comb = funny_bar + funny_count
funny_comb
ggsave("funny_comb.png", plot = funny_comb, width = 5, height = 3)

# Cool
cool_bar = ggplot(data=master_sample, aes(x=r_stars, y=cool)) + 
  geom_bar(stat="summary") +
  ggtitle("Violin: Review Stars / 'Cool'") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4)) #weird on scale

master_sample$cool <- factor(master_sample$cool)
cool_count <- ggplot(data=master_sample) +
  geom_count(aes(x=r_stars,y=cool)) + 
  coord_cartesian(ylim = c(0, 5)) +
  scale_size_area() +
  ggtitle("Review Stars / 'Cool'") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4),
        legend.text = element_text(size=6))
master_sample$cool <- as.numeric(master_sample$cool)

cool_comb = cool_bar + cool_count
cool_comb
ggsave("cool_comb.png", plot = cool_comb, width = 5, height = 3)

# User review count
revcount_bar = ggplot(data=master_sample, aes(x=r_stars, y=u_review_count)) + 
  geom_bar(stat="summary") +
  ggtitle("Review Stars / Avg. review count of a user") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))

revcount_violin = ggplot(data=master_sample) + 
  geom_violin(aes(x=r_stars, y=u_review_count)) + 
  coord_cartesian(ylim = c(0,50)) +
  ggtitle("Violin: Review Stars / Avg. review count of a user") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))

revcount_comb = revcount_bar + revcount_violin
revcount_comb
ggsave("revcount_comb.png", plot = revcount_comb, width = 7, height = 3)

# Length of review 
len_bar <- ggplot(data=master_sample, aes(x=r_stars, y=revlength)) +
  geom_bar(stat="summary") +
  ggtitle("Bar: Review Stars / Review Length") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))

len_violin = ggplot(data=master_sample) + 
  geom_violin(aes(x=r_stars, y=revlength)) + 
  coord_cartesian(ylim = c(1,400)) +
  ggtitle("Violin: Review Stars / Review Length") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))

len_comb <- len_bar + len_violin
len_comb
ggsave("len_comb.png", plot = len_comb, width = 5, height = 3)

# Exclamation marks
exclam_bar <- ggplot(data=master_sample, aes(x=r_stars, y=exclam)) + 
  geom_bar(stat="summary") +
  ggtitle("Bar: Review Stars / Number of '!'") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
exclam_violin <- ggplot(data=master_sample, aes(x=r_stars, y=exclam)) + 
  geom_violin() + 
  coord_cartesian(ylim=c(0,0.03)) +
  ggtitle("Violin: Review Stars / Number of '!'") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))

exclam_comb = exclam_bar + exclam_violin
exclam_comb
ggsave("exclam_comb.png", plot = exclam_comb, width = 5, height = 3)

# Afinn sentiment score
afinn_bar = ggplot(data=master_sample) + 
  geom_bar(stat="summary", aes(x=r_stars, y=afinn_sentiment)) + 
  ggtitle("Bar: Review Stars / AFINN Sentiment Score") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
afinn_violin = ggplot(data=master_sample) +
  geom_violin(aes(x=r_stars,y=afinn_sentiment)) + 
  coord_cartesian(ylim=c(-0.1,0.2)) +
  ggtitle("Violin: Review Stars / AFINN Sentiment Score") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
afinn_comb = afinn_bar + afinn_violin
afinn_comb
ggsave("afinn_comb.png", plot = afinn_comb, width = 6, height = 4)

# Bing sentiment score
bing_bar = ggplot(data=master_sample) + 
  geom_bar(stat="summary", aes(x=r_stars, y=bing_sentiment)) + 
  ggtitle("Bar: Review Stars / BING Sentiment Score") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
bing_violin = ggplot(data=master_sample) +
  geom_violin(aes(x=r_stars,y=bing_sentiment)) + 
  coord_cartesian(ylim=c(-0.07,0.1)) +
  ggtitle("Violin: Review Stars / BING Sentiment Score") +
  theme(plot.title = element_text(size = 8, hjust = 0.5), 
        axis.title = element_text(size = 6),
        axis.text = element_text(size=4))
bing_comb = bing_bar + bing_violin
bing_comb
ggsave("bing_comb.png", plot = bing_comb, width = 6, height = 4)


###############################################################################################################################################################
# 7 Modelling
###############################################################################################################################################################

# 7.1 Ordered Logit ##############################################################################################

# Specification 1: Full set of covariates
OL1 <- polr(r_stars ~ 
              u_average_stars +
              b_stars +
              afinn_sentiment +
              cool +
              revlength, 
            data = master_train,
            method = "logistic")
OL1

# Eval. on test data
OL1_prediction <- predict(OL1,
                          newdata = master_test,
                          type = "class")
master_test$OL1_p <- OL1_prediction
cm_OL1 <- table(master_test$r_stars, master_test$OL1_p)
cm_OL1
acc_OL1 <- sum(diag(cm_OL1)) / sum(cm_OL1)

# Eval. on training data
OL1_prediction_train <- predict(OL1,
                                newdata = master_train,
                                type = "class")
master_train$OL1_p_train <- OL1_prediction_train
cm_OL1_train <- table(master_train$r_stars, master_train$OL1_p_train)
cm_OL1_train
acc_OL1_train <- sum(diag(cm_OL1_train)) / sum(cm_OL1_train)


# Specification 2: Sentiment analysis only
OL2 <- polr(r_stars ~ 
              afinn_sentiment, 
            data = master_train,
            method = "logistic")
OL2

# Eval. on test data
OL2_prediction <- predict(OL2,
                          newdata = master_test,
                          type = "class")
master_test$OL2_p <- OL2_prediction
cm_OL2 <- table(master_test$r_stars, master_test$OL2_p)
cm_OL2
acc_OL2 <- sum(diag(cm_OL2)) / sum(cm_OL2)

# Eval. on training data 
OL2_prediction_train <- predict(OL2,
                                newdata = master_train,
                                type = "class")
master_train$OL2_p_train <- OL2_prediction_train
cm_OL2_train <- table(master_train$r_stars, master_train$OL2_p_train)
cm_OL2_train
acc_OL2_train <- sum(diag(cm_OL2_train)) / sum(cm_OL2_train)


# Specification 3: Numerical only
OL3 <- polr(r_stars ~ 
              u_average_stars +
              b_stars,
            data = master_train,
            method = "logistic")
OL3

# Eval. on test data
OL3_prediction <- predict(OL3,
                          newdata = master_test,
                          type = "class")
master_test$OL3_p <- OL3_prediction
cm_OL3 <- table(master_test$r_stars, master_test$OL3_p)
cm_OL3
acc_OL3 <- sum(diag(cm_OL3)) / sum(cm_OL3)

# Eval. on training data
OL3_prediction_train <- predict(OL3,
                                newdata = master_train,
                                type = "class")
master_train$OL3_p_train <- OL3_prediction_train
cm_OL3_train <- table(master_train$r_stars, master_train$OL3_p_train)
cm_OL3_train
acc_OL3_train <- sum(diag(cm_OL3_train)) / sum(cm_OL3_train)

# Print accuracy:

# Test data:
print(paste("Spec.1 accuracy:", acc_OL1))
print(paste("Spec.2 accuracy:", acc_OL2))
print(paste("Spec.3 accuracy:", acc_OL3))
# Training data:
print(paste("Spec.1 accuracy:", acc_OL1_train))
print(paste("Spec.2 accuracy:", acc_OL2_train))
print(paste("Spec.3 accuracy:", acc_OL3_train))


# 7.2 Decision Tree ##############################################################################################

# Specification 1: All covariates
DT1 <- rpart(r_stars ~ 
               u_average_stars +
               b_stars +
               afinn_sentiment +
               bing_sentiment +
               u_review_count +
               revlength +
               exclam + 
               useful +
               cool,
             data = master_train,
             method = "class",
             cp=0.0003)

# Eval. on test data
DT1_prediction <-predict(DT1,
                         newdata=master_test,
                         type="class")
rpart.plot(DT1)
confusionMatrix(master_test$r_stars, DT1_prediction)

# Eval. on training data
DT1_prediction_train <-predict(DT1,
                               newdata=master_train,
                               type="class")
confusionMatrix(master_train$r_stars, DT1_prediction_train)


# Specification 2: Sentiment Data Only
DT2 <- rpart(r_stars ~ 
               afinn_sentiment +
               bing_sentiment +
               exclam,
             data = master_train,
             method = "class",
             cp=0.0003)

# Eval. on test data
DT2_prediction <-predict(DT2,
                         newdata=master_test,
                         type="class")

rpart.plot(DT2, box.palette = "Gn")
confusionMatrix(master_test$r_stars, DT2_prediction)

# Eval. on training data
DT2_prediction_train <-predict(DT2,
                               newdata=master_train,
                               type="class")
confusionMatrix(master_train$r_stars, DT2_prediction_train)


# Specification 3: Numerical Only
DT3 <- rpart(r_stars ~ 
               u_average_stars +
               b_stars,
             data = master_train,
             method = "class",
             cp=0.0003)

# Eval. on test data
DT3_prediction <-predict(DT3,
                         newdata=master_test,
                         type="class")

rpart.plot(DT3)
confusionMatrix(master_test$r_stars, DT3_prediction)

# Eval. on training data
DT3_prediction_train <-predict(DT3,
                               newdata=master_train,
                               type="class")
confusionMatrix(master_train$r_stars, DT3_prediction_train)


# 7.3 Random Forest ##############################################################################################

# Specification 1: All covariates
RT1 <- randomForest(r_stars ~ 
                      u_average_stars +
                      b_stars +
                      afinn_sentiment +
                      bing_sentiment +
                      u_review_count +
                      revlength +
                      exclam + 
                      useful +
                      cool,
                    data = master_train,
                    nodesize = 200,
                    ntree=200)

# Eval. on test data
RT1_prediction <-predict(RT1,
                         newdata=master_test,
                         type="class")

actual_stars1 <- master_test$r_stars
acc_RT1 <- sum(RT1_prediction == actual_stars1) / length(actual_stars1) *100
table(master_test$r_stars, RT1_prediction)

# Eval. on training data
RT1_prediction_train <-predict(RT1,
                               newdata=master_train,
                               type="class")

actual_stars1_train <- master_train$r_stars
acc_RT1_train <- sum(RT1_prediction_train == actual_stars1_train) / length(actual_stars1_train) *100
table(master_train$r_stars, RT1_prediction_train)


# Specification 2: Sentiment Only
RT2 <- randomForest(r_stars ~ 
                      afinn_sentiment +
                      bing_sentiment +
                      exclam,
                    data = master_train,
                    nodesize = 200,
                    ntree=200)

# Eval. on test data
RT2_prediction <-predict(RT2,
                         newdata=master_test,
                         type="class")

actual_stars2 <- master_test$r_stars
acc_RT2 <- sum(RT2_prediction == actual_stars2) / length(actual_stars2) *100
table(master_test$r_stars, RT2_prediction)

# Eval. on training data
RT2_prediction_train <-predict(RT2,
                               newdata=master_train,
                               type="class")

actual_stars2_train <- master_train$r_stars
acc_RT2_train <- sum(RT2_prediction_train == actual_stars2_train) / length(actual_stars2_train) *100
table(master_train$r_stars, RT2_prediction_train)


# Specification 3: User and Business Data Only
RT3 <- randomForest(r_stars ~ 
                      u_average_stars +
                      b_stars,
                    data = master_train,
                    nodesize = 200,
                    ntree=200)

# Eval. on test data
RT3_prediction <-predict(RT3,
                         newdata=master_test,
                         type="class")

actual_stars3 <- master_test$r_stars
acc_RT3 <- sum(RT3_prediction == actual_stars3) / length(actual_stars3) *100
table(master_test$r_stars, RT3_prediction)

# Eval. on training data
RT3_prediction_train <-predict(RT3,
                               newdata=master_train,
                               type="class")

actual_stars3_train <- master_train$r_stars
acc_RT3_train <- sum(RT3_prediction_train == actual_stars3_train) / length(actual_stars3_train) *100
table(master_train$r_stars, RT3_prediction_train)

# Print accuracy:

# Test data:
print(paste("Spec. 1 accuracy:", acc_RT1, "%"))
print(paste("Spec. 2 accuracy:", acc_RT2, "%"))
print(paste("Spec. 3 accuracy:", acc_RT3, "%"))

# Training data:
print(paste("Spec. 1 accuracy:", acc_RT1_train, "%"))
print(paste("Spec. 2 accuracy:", acc_RT2_train, "%"))
print(paste("Spec. 3 accuracy:", acc_RT3_train, "%"))


#############################################################################################################################

# Project end