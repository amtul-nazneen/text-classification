# Text Classification

Implement and evaluate Naive Bayes and Logistic Regression for text classification for the spam/ham dataset.

## Overview
Multinomial Naive Bayes algorithm as per http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf and MCAP Logistic Regression algorithm with L2 regularization has been implemented. Improvement over these algorithms by throwing away the stop words and their impact analysis has been done. 

## Steps to run
1. Naive Bayes
* Run the command `python NaiveBayes.py`
* This will run for *With* and *Without* Stop Words
2. Logistic Regression
* Run the command `python LogisticRegression.py`
* This will run for *With* and *Without* Stop Words
* Parameters such as: lambdaVal, totalIterations, learningRate are configurable (at the beginning of the file)
3. Analysis of both the algorithms with their accuracies have been documented in `src\analysis.pdf`
