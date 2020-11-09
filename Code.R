# Load packages
library("OneR")
library(C50)
library(gmodels)
library(rpart)
library("rpart.plot")

#####Example C5.0 Decision tree algorithm####
credit <- read.csv("~/Desktop/Portfolio/Credit defaulting/docs/credit.csv", stringsAsFactors=TRUE)

View(credit)
str(credit)
table(credit$checking_balance) ##values are in Deustche Mark because data was obtained from Germany
summary(credit$months_loan_duration)
table(credit$default)

#randomly split data into training, validation and test (80,10,10)
RNGversion("4.0.2"); set.seed(100) ##123 is abritraty
train_sample <- sample(1000, 900) # here we randomly choose 900 numbers from number 1-1000
##get the initial training df
credit_train <- credit[train_sample, ]
#get the test df
credit_test  <- credit[-train_sample, ]

val_sample <- sample(900, 800) # here we randomly choose 900 numbers from number 1-1000
#Get the validation df
credit_val  <- credit_train[-val_sample, ]
##get the training df
credit_train <- credit_train[val_sample, ]



##check proportions of credit for each dataset to make sure not biased
prop.table(table(credit_train$default))
prop.table(table(credit_val$default))
prop.table(table(credit_test$default))

####One Rule learner####
# As a baseline


credit_1R <- OneR(default ~ .,data = credit_train) 
credit_1R
# the training model witha 1 rule learner (based on credit history) has an accuracy of 73.5%


credit_1R_pred <- predict(credit_1R, credit_val)
table(actual = credit_val$default, predicted = credit_1R_pred)
#on the validation data, the 1 rule model has ana accuracy of 65%, which is used as a baseline indicator. 


######running C5.0 DT########
install.packages("C50")
library(C50)
?C5.0Control #gives info on fine-tuning the algorithm


# we need to exlude the outcome variable from the training dataset, that is if a person defaults on their loan or not. and it has to be a factor variable. 
credit_train$default <- as.factor(credit_train$default)
credit_model <- C5.0(credit_train[-17], credit_train$default)
credit_model # we see there are 20 predictors and has 67 branches (it's a big tree). 
summary(credit_model)
# Based on the training data there was an accuracy of 89.4% for correctly predicting credit default (error rate, 10.6%). Indicating the model does not have overfit or underfit for the training data.

####Evaulate model performance on the validation data
credit_pred <- predict(credit_model, credit_val)
library(gmodels)
CrossTable(credit_val$default, credit_pred, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default'))

#We see the decision tree models has a 72% correct classification and 28% incorrect (error rate). Additionally, the model correctly predicts 15 defaults but missed on 17 defaults (this is not good.) Another way to view this, if we simply said all customers won't default we have a 68% accuracy which isn't much worse than our current model of 74%. So let's improve the model. 

# #######improve the model. see chapter 11 for more info:#########
#adaptive boosting is a DT is built the tree votes on the best class for each example. 

# do on the training set first. it is the same but add "trials = 10" 10 is the default and work shows that it typically increases model accuracy by 25% of prior performance. not a raw 25% but 25% from model with no trial

credit_boost10 <- C5.0(credit_train[-17], credit_train$default,
                       trials = 10)

credit_boost10
# we have ran the model 10 times with 20 predictors but the average tree is 58.2 compared to the prior wich was a single 67
summary(credit_boost10)
#so we have 10 DT's, but scroll down and look at confusion matirx. we have 4 errors out of 800, which is a 0.5% error rate which is much smaller than the prior error rate of 10.6% for the training set. But does this overfit the training data?

#run adpatve boosting on test data
credit_boost_pred10 <- predict(credit_boost10, credit_val)
CrossTable(credit_val$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
## our model decreased in accuracy from 72% to 69% because of overfit.


##model with 25 trial
credit_boost25 <- C5.0(credit_train[-17], credit_train$default,
                       trials = 25)

credit_boost25
# we have ran the model 25 times with 20 predictors but the average tree is 51.3 compared to the prior wich was a single 54
summary(credit_boost25)
#so we have 25 DT's, but scroll down and look at confusion matirx. we have 9 errors out of 900, which is a 1.0% error rate (9/900) which is much smaller than the prior error rate of 15% for the training set. 

#run adpatve boosting on test data
credit_boost_pred25 <- predict(credit_boost25, credit_val)
CrossTable(credit_val$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
## even with 25 trials, our model decreased in accuracy from 74% to 66%, but we have a better classifacications of defaults on loans comapred to the prior model

#reasons for poor transloation to test set:
#1:The lack of an even greater improvement may be a function of our relatively small training dataset, or 
#2: it may just be a very difficult problem to solve.

# sme mistakse are more costly than others, giving a loan to somone who will default vs not giving someone a loan who wont default. can assess this with a cost matrix: so we make one
matrix_dimensions <- list(c("1", "2"), c("1", "2"))
names(matrix_dimensions) <- c("predicted", "actual")
##next we fill out the matrix. 0 cost nothing, a 1 is for saying no to someone who wont default, but a 4 is given to saying yes to someone who will default. these are giving the classifications weights
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2,
                     dimnames = matrix_dimensions)
error_cost
##apply to non-boosting model, just adding "costs = "
credit_cost <- C5.0(credit_train[-17], credit_train$default, costs = error_cost)
summary(credit_cost)
#this has 27.3% error rate because of weight
credit_cost_pred <- predict(credit_cost, credit_test)
CrossTable(credit_test$default, credit_cost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
####here we see 59% accuracy but the number of laons given to people who default has greatly decreased down to 7. while we turn down a lot of loans for people who wont default 34). it is a more conservative approach


#### a very conservative approach, give it 8 times the wieght
matrix_dimensions <- list(c("1", "2"), c("1", "2"))
names(matrix_dimensions) <- c("predicted", "actual")
##next we fill out the matrix. 0 cost nothing, a 1 is for saying no to someone who wont default, but a 8 is given to saying yes to someone who will default. these are giving the classifications weights
error_cost <- matrix(c(0, 1, 8, 0), nrow = 2,
                     dimnames = matrix_dimensions)
error_cost
##apply to non-boosting model, just adding "costs = "
credit_cost <- C5.0(credit_train[-17], credit_train$default, costs = error_cost)
summary(credit_cost)
#this has 27.3% error rate because of weight
credit_cost_pred <- predict(credit_cost, credit_test)
CrossTable(credit_test$default, credit_cost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

####here we see 56% accuracy but the number of laons given to people who default has greatly decreased down to 3. while we turn down a lot of loans for people who wont default 41. it is a very very conservative approach


####combining adaptive boositng and cost matrix#######

matrix_dimensions <- list(c("1", "2"), c("1", "2"))
names(matrix_dimensions) <- c("predicted", "actual")
##next we fill out the matrix. 0 cost nothing, a 1 is for saying no to someone who wont default, but a 8 is given to saying yes to someone who will default. these are giving the classifications weights
error_cost <- matrix(c(0, 1, 10, 2), nrow = 2,
                     dimnames = matrix_dimensions)
##apply to non-boosting model, just adding "costs = "
credit_cost_boost <- C5.0(credit_train[-17], credit_train$default, costs = error_cost, trials = 5)
# summary(credit_cost_boost)
#this has 27.3% error rate because of weight
credit_costboost_pred <- predict(credit_cost_boost, credit_test)
CrossTable(credit_test$default, credit_costboost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))
#### heres a model with weights and adaptive boosting. we decreased the boosts to 5 trials, and heavily weighted giving loans to people who will default and the model says no they won't. bt we also have an ethical part of weighting giving loans to people who will default as predicted. overall accuracy is only 66% but it has ethics in the loans(not modeled for risk or maing money, but ethics to avoid a crisis in banking)



###############Rule learners##################
# These use if-then statements
# used in indentfitying changes that may happen in a system(mechanical failure, stock market change)
####differs from DT because even easier to understand because its a bunch of if-then logic#####

#####pros: very useful for identifying rare events!!!!!!!!!!!!!!!!!!!!!! good with nominal data

## instead of divide and conquer they call it separate and conquer. it is like classificying a bunch of random animals into mammals and non-mammals see pg. 149 for animal example. 
##these can also be called covering alogrithms. 

####a type of rule learner algorithm (the 1R algorithm)#####
#a zeroR would be a learner who learns nothing but just consderes the highest probability option (its like a card counter on the first hand they play, they see the odds of all the cards out there but havent learned anything yet). the 1R selects a single rule and follows that. 

#pros: 1. creates a single, easy to follow rule, easy to understand. 2. performs surprising well
#####pro 3: 1R algorithms are often the bench used for more complex algorithms####
#cons: uses a single feature, overly simple
#works by choosing a rule for each feature, then the rule with the lowest error rate is chosen. see pg. 152 for animal example
#####sum of the 1R= find the single most important rule and stop there####



#####regression tree####
credit_train1 <- credit_train
credit_train1$default <- as.numeric(as.character(credit_train1$default))
m.rpart <- rpart(default ~ ., data = credit_train1) ##looking at all feature predicting quality
m.rpart# all wines begin at root, 3674 of which 2300 have alc less than 10.85 and 1391 has alc more than 10.85. level of alc was the first split so its the most important. the first section is wines under alc 10.85 and second section is wines over 10.85 alc. NODES ending with * are terminal nodes. the number at the end is the quality rating
#### we can visualize a regression tree #####
rpart.plot(m.rpart, digits = 3)
#the numbers on top are the quality ratings and the % is the number of wines from the total 
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101) # like this plot better
#fallen leaves moves all nods to bottom, type and extra are styling formats. 

#evaluate the model
p.rpart <- predict(m.rpart, credit_train1)
summary(p.rpart)
summary(credit_train1$default)
# we see the model is doing well btw Q1 and Q3 but not great at the extremes
cor(credit_train1$default, p.rpart)
#acceptable, but not a great correlation of .60
#so lets check the model with mean absolute error (MAE)
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))
}
MAE(credit_train1$default, p.rpart)
#so the mean error is 0.26 for the model in determing the default 
mean(credit_train1$default)
MAE(1.29, credit_val$default)#the mean for the training set is 1.29, so when we look at mean absolute error of this mean to the test set we get an MAE of .42 which compare to our model is .26 which is worse but not great. Using a neural network model we've gotten down to .58 and with a support vector machine its been down to .45


#######notes:######
# Decision trees and rule learners are known as greedy learners because they use data on a first-come, first-served basis. Both the divide and conquer heuristic used by decision trees and the separate and conquer heuristic used by rule learners attempt to make partitions one at a time, finding the most homogeneous partition first, followed by the next best, and so on, until all examples have been classified.
# The downside to the greedy approach is that greedy algorithms are not guaranteed to generate the optimal, most accurate, or smallest number of rules for a particular dataset. By taking the low-hanging fruit early, a greedy learner may quickly find a single rule that is accurate for one subset of data; however, in doing so, the learner may miss the opportunity to develop a more nuanced set of rules with better overall accuracy on the entire set of data. However, without using the greedy approach to rule learning, it is likely that for all but the smallest of datasets, rule learning would be computationally infeasible.

######good note########
#: also for a DT once a branch  has been made that set of data can't be included in other decisions, it stays on that branch and that branch alone, so we lose overall information when we branch. but gain info for that specific branch. but for rule learners once separate and conquer finds a rule, any examples not covered by all of the rule's conditions may be re-conquered.
#see pg 156 for example of difference. 


# 
# Note The evolution of classification rule learners didn't stop with RIPPER. New rule learning algorithms are being proposed rapidly. A survey of literature shows algorithms called IREP++, SLIPPER, TRIPPER, among many others.
# 
# Lantz, Brett. Machine Learning with R: Expert techniques for predictive modeling, 3rd Edition (pp. 154-155). Packt Publishing. Kindle Edition. 