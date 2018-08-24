# Logistic Regression on Amazon Reviews (Part I)#

## Amazon Fine Food Review Dataset ##

Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews

The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon. <br/>
Number of reviews                   : 568,454  <br/>
Number of users                     : 256,059  <br/>
Number of products                  : 74,258  <br/>
Timespan: Oct 1999                  : Oct 2012  <br/>
Number of Attributes/Columns in data: 10 <br/>

### Attribute Information ###
1. Id <br/>
2. ProductId - unique identifier for the product <br/>
3. UserId - unqiue identifier for the user <br/>
4. ProfileName <br/>
5. HelpfulnessNumerator - number of users who found the review helpful <br/>
6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not <br/>
7. Score - rating between 1 and 5 <br/>
8. Time - timestamp for the review <br/>
9. Summary - brief summary of the review <br/>
10. Text - text of the review <br/>

## Objective ##

The code below would **clean the review text from html tags and punctuations and write it as a new column in the database and write it to disk**. This is further taken up in Part 2 to find accuracy of 10-fold cross validation KNN on vectorized input data, for each of the 4 featurizations, namely BoW, tf-IDF, W2V, tf-IDF weighted W2V.

## Significant Points ##

1. **Duplication of reviews** are found with same userid and timestamp (Cleaned).
2. Found discrepancy issues with HelpfulnessDenominator (Cleaned).
3. final.sqlite db is to be **used for further processing** such as Text to Vector operations.
4. The preprocessing step is one time effort but the training & visualization steps require multiple runs. Hence, it is prudent to make reprocessing step independant, to avoid multiple runs.

# Logistic Regression on Amazon Reviews Dataset (Part II) #

## Data Source ##

The preprocessing step has produced final.sqlite file after doing the data preparation & cleaning. The review text is now devoid of punctuations, HTML markups and stop words.

## Objective ##

**To find optimal lambda using GridSearchCV & RandomSearchCV** on standardized feature vectors obtained from BoW, tf-idf, W2V and tf-idf weighted W2V featurizations. To study the impact on sparsity upon increasing lambda.

**Find Precision, Recall, F1 Score, Confusion Matrix, Accuracy of 10-fold cross validation with GridSearch and RandomSearch with optimal Logistic Regression regression model on vectorized input data, for BoW, tf-idf, W2V and tf-idf weighted W2V featurizations**. TPR, TNR,
FPR and FNR is calculated for all.

After finding the optimal model, **do Perturbation test** to remove multicollinear features. **Find top n words** using the weight vector, w.

## At a glance ##

Random Sampling is done to reduce input data size and time based slicing to split into training and testing data. The optimal lambda is found out using GridSearchCV & RandomSearchCV with a range of lamda values to search (for GridSearch) and an uniform distribution (for RandomSearchCV.

The Precision, Recall, F1 Score, Confusion Matrix, Accuracy metrics are found out for all 4 featurizations. A normal distribution noise is added for perturbnatino test and the identified multicollinear features are removed. Then the top ’n’ words are found out after removal of multicollinear features based on highest values of |w|.

## Custom Defined Functions ##

5 user defined functions are written to
1. Perform GridSearchCV & RandomSearchCV for Optimal Alpha Estimation.
2. Compute Logistic Regression Classifier Performance Metrics.
3. Find Most Frequent Words.
4. Analyze Sparsity for increasing Lambda.
5. Perturbation Test with a Normal Distributed Noise. <br/> Sparsity of input vector is preserved for BoW and tf-idf featurizations. For W2V and tf-idf W2V the features are dense.

## BoW ##

BoW will result in a sparse matrix with huge number of features as it creates a feature for each unique word in the review.

For Binary BoW feature representation, CountVectorizer is declared as float, as the values can take non-integer values on further processing. Top n words are found out after checking for multicollinearity.

i1

i2

## Sparsity vs F1 score Plot ##

The variation of sparsity corresponding to varying values of lambda is plotted and the lambda with the highest accuracy is identified. The optimal model can be found out using the sparsity vs f1 score plot also.

spar

score

## tf-IDF ##

Sparse matrix generated from tf-IDF is fed in to GridSearch and RandomSearch Logistic Regression Cross Validator to find the optimal lambda value. Performance metrics of optimal LR with tf-idf featurization is found.

The optimal value of lambda using RandomizedSearchCV is 0.002245.

**Metric Analysis of Logistic Classifier for Optimal Lamdba**
Accuracy = 86.033333
Precision = 91.228756
Recall = 92.310733
F1 Score = 91.766555

**Confusion Matrix**
True Negatives = 492
True Positives = 4670
False Negatives = 389
False Positives = 449

Total Actual Positives = 5059
Total Actual Negatives = 941
True Positive Rate(TPR) = 0.92
True Negative Rate(TNR) = 0.52
False Positive Rate(FPR) = 0.48
False Negative Rate(FNR) = 0.08

**GridSearchCV: Best C: 0.001**

The optimal value of lambda using GridSearchCV is 1000.000000.

**Metric Analysis of Logistic Classifier for Optimal Lamdba**
Accuracy = 89.283333
Precision = 90.000000
Recall = 98.201226
F1 Score = 93.921921

**Confusion Matrix**
True Negatives = 389
True Positives = 4968
False Negatives = 91
False Positives = 552

Total Actual Positives = 5059
Total Actual Negatives = 941
True Positive Rate(TPR) = 0.98
True Negative Rate(TNR) = 0.41
False Positive Rate(FPR) = 0.59
False Negative Rate(FNR) = 0.02

**Length of Weight Vector (Before Removing Collinearity): 15114**
Distance between Weight vectors before & after Perturbation = 2.93
Multicollinear Features = 7557

**Length of Weight Vector (After Removing Collinearity): 7557**

rv2

gv2

## Word2Vec ##

Dense matrix generated from Word2Vec is fed in to GridSearch and RandomSearch Logistic Regression Cross Validator to find the optimal lambda value.

Performance metrics of optimal LR with W2V featurization is found. But we cannot find the top ’n’ words when we use Word2Vec based featurization, because the feature doesnt correspond to a word in the vocabulary.

**The optimal value of lambda using RandomizedSearchCV is 103.277877.**

**Metric Analysis of Logistic Classifier for Optimal Lamdba**
Accuracy = 84.250000
Precision = 84.386493
Recall = 99.782566
F1 Score = 91.440993

**Confusion Matrix**
True Negatives = 7
True Positives = 5048
False Negatives = 11
False Positives = 934

Total Actual Positives = 5059
Total Actual Negatives = 941
True Positive Rate(TPR) = 1.0
True Negative Rate(TNR) = 0.01
False Positive Rate(FPR) = 0.99
False Negative Rate(FNR) = 0.0

**GridSearchCV: Best C: 100**

The optimal value of lambda using GridSearchCV is 0.010000.

**Metric Analysis of Logistic Classifier for Optimal Lamdba**
Accuracy = 52.150000
Precision = 86.515354
Recall = 51.235422
F1 Score = 64.357542

**Confusion Matrix**
True Negatives = 537
True Positives = 2592
False Negatives = 2467
False Positives = 404

Total Actual Positives = 5059
Total Actual Negatives = 941

True Positive Rate(TPR) = 0.51
True Negative Rate(TNR) = 0.57
False Positive Rate(FPR) = 0.43
False Negative Rate(FNR) = 0.49

**Length of Weight Vector (Before Removing Collinearity): 300**
Distance between Weight vectors before & after Perturbation = 0.0
Multicollinear Features = 9

**Length of Weight Vector (After Removing Collinearity): 291**

r3

g3

## TF-IDWeighted W2V ##

Grid Search and Random Search CV using Logistic Regression Best Penalty: l1
RandomizedSearchCV: Best C: 0.9648400471483856

The optimal value of lambda using RandomizedSearchCV is 1.036441.

**Metric Analysis of Logistic Classifier for Optimal Lamdba**
Accuracy = 52.566667
Precision = 83.540467
Recall = 54.477169
F1 Score = 65.948792

**Confusion Matrix**
True Negatives = 398
True Positives = 2756
False Negatives = 2303
False Positives = 543

Total Actual Positives = 5059
Total Actual Negatives = 941

True Positive Rate(TPR) = 0.54
True Negative Rate(TNR) = 0.42
False Positive Rate(FPR) = 0.58
False Negative Rate(FNR) = 0.46

GridSearchCV: Best C: 100

**The optimal value of lambda using GridSearchCV is 0.010000.**

**Metric Analysis of Logistic Classifier for Optimal Lamdba**
Accuracy = 50.833333
Precision = 86.424870
Recall = 49.456414
F1 Score = 62.911743

**Confusion Matrix**
True Negatives = 548
True Positives = 2502
False Negatives = 2557
False Positives = 393

Total Actual Positives = 5059
Total Actual Negatives = 941

True Positive Rate(TPR) = 0.49
True Negative Rate(TNR) = 0.58
False Positive Rate(FPR) = 0.42
False Negative Rate(FNR) = 0.51

**Length of Weight Vector (Before Removing Collinearity): 300**
Distance between Weight vectors before & after Perturbation = 12.02
Multicollinear Features = 136

**Length of Weight Vector (After Removing Collinearity): 164**

r4

g4

## Summary Statistics ##

sum

## Observations ##

1. From the Sparsity and F1 Score plot, it can be identified that Performance & Sparsity is the best when Log (Lambda) is between 1 and 2. i.e. Lambda = 10ˆ1 ~ 10ˆ2 = 10 ~ 100. The lambda values obtained via plotting method is almost same as the lambda value found
out by GridSearchCV and RandomSearchCV. (Please note that, Sparsity = # of non-zero elements, in this project).

2. It has also been noticed that, with increasing lambda, the sparsity (# of non-zero elements) has been decreasing steadily. This is an expected behaviour, as L1 regularization is used.

3. The Lambda values found by GridSearchCV and RandomizedSearchCV are near, only when the range of "C" values is set within a narrow range, around optimum. i.e. if the optimal C = 1 (as per GridSearchCV), then by setting C as a uniform distribution between 0 and 4
will yield C = 1 (+/- 0.05) approximately, within say, 100 iterations. But if C value is set as a uniform distribution between 0 and say, 10000, then the error in C value is found to be very high.

4. Alternatively, if the range of C value is wide, to arrive at optimal C, we need to increase the number of iterations significantly. It is seen that, when iterations are increased from 100 to 1000, the C value is converging to optimum. But the time complexity of such an approach would be much higher.
