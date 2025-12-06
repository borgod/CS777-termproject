# CS777-termproject
# by Xioaning Zhang, Pearlrose Nwuke, and Boris Godoy


# Description of the code

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


# Exploration of the Data set

![image Alt](https://github.com/borgod/CS777-termproject/blob/main/graph01.png?raw=true)

The above image shows data the 20-most delayed routes in the dataset



![image Alt](https://github.com/borgod/CS777-termproject/blob/main/graph02.png?raw=true)

The above image shows data the 20-most delayed origin airports


![image Alt](https://github.com/borgod/CS777-termproject/blob/main/graph03.png?raw=true)
The above image shows JetBlue as the airline with the worst departure delay



![image Alt](https://github.com/borgod/CS777-termproject/blob/main/graph04.png?raw=true)
The above image shows average departure delay by hour of the day

![image Alt](https://github.com/borgod/CS777-termproject/blob/main/graph05.png?raw=true)
The above image shows average departure delay by month of the year


# Preliminary Results

![image Alt](https://github.com/borgod/CS777-termproject/blob/main/results_project.png?raw=true)

Here, we created a table with the prediction of flights and the value ROC AUC. This value is modest so far, and it might be due to the limited dataset (we are researching the cause). We will know when we run the code with the big dataset in the cloud.










