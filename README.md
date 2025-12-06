# CS777-termproject
# by Xioaning Zhang, Pearlrose Nwuke, and Boris Godoy


This document explains how to use the file flighy.py

Most important thing to set is the correct path for your file:

path = "/Documents/..../yourfile.csv "

Once you have set up the correct path, then you can run the whole
pipeline, or simply crun it by the 3 main sections and run it as a 
Jupyter notebook

# 1st section 
We read the dataset and extract the rows of interest for the
classifier

All relevant data is in the variable :

selected = rows.map(safe_convert).filter(lambda x: x is not None)


# 2nd section 
We create a dataframe because we use the library embedded for logistic 
regression. 

We then do feature engineering, that is, creating some features from
the original ones. This goes from Create Data Frame to Train/test split

Since logistic regression only works with numerical data, we need to encode
the categorical important data. We do that also in this section


# 3rd section

Here we calculate our logistic regression model and then calculate the
performance metrics for the classifier


# Data set

![image Alt](https://github.com/borgod/CS777-termproject/blob/main/graph01.png?raw=true)





# Results

![image Alt](https://github.com/borgod/CS777-termproject/blob/main/results_project.png?raw=true)

Here, we created a table with the prediction of flights and the value ROC AUC. This value is modest so far, and it might be due to the limited dataset (we are researching the cause).










