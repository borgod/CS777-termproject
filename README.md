# CS777-termproject
# by Xioaning Zhang, Pearlrose Nwuke, and Boris Godoy


# Description of the files
==========================

We basically have 3 python files, one of each corresponding to a 
different classifier.

We also have 1 file, which is a sample version of the entire dataset.

# Logistic regression classifier - lr_classifier.py
----------------------------------------------------

This files calculates the logistic regression classifier, and also 
it calculates its performance.

The pipeline is self-contained, and the user only needs to run
the file in a terminal :

$python lr_classifier.py 

The performance metrics will print out at the end, as well a model will be saved 
for future use.

# Random Forrest classifier - rf_classifier.py
-----------------------------------------------

This files calculates the random forrest classifier, and also 
it calculates its performance.

The pipeline is self-contained, and the user only needs to run
the file in a terminal :

$python rf_classifier.py

The performance metrics will print out at the end, as well a model will be saved 
for future use.


# Gradient-boosted trees  - gbt_classifier.py
----------------------------------------------

This files calculates the gradient-boosted trees classifier, and also 
it calculates its performance.

The pipeline is self-contained, and the user only needs to run
the file in a terminal :

$python gbt_classifier.py 

The results of the software will print out at the end, as well a model will be saved 
for future use

The user only to make sure to add the correct path at the beginning of each
file.

path = "/Documents/..../your_flight_dataset.csv "

Once you have set up the correct path, then you can run the whole
pipeline, or simply run it by the 3 main sections and run it in a 
Jupyter notebook (you need to copy/past the sections).


# Common aspects common to each of the files

We can say that each of the files are composed of 3 sections, and they can run sequentially

# 1st section 
We read the dataset and extract the rows of interest for the
classifier

All relevant data is in the variable :

selected = rows.map(safe_convert).filter(lambda x: x is not None)

# 2nd section 
We create a dataframe because we use the embedded library  for classifiers.

We then do feature engineering, that is, creating some features from
the original ones. This goes from Create Data Frame to Train/test split

In the logistic regression classifier, an enconder is used to take
the categorical variables because it only can take numerical values.

# 3rd section
Here we calculate our respective classifier (lr, gbt, or rf), and then the
performance metrics are printed out for each of them.


# Environment setup
===================

Here, we describe all dependencies, libraries, versions, and setup steps 
necessary to reproduce the environment.

Python
------
-. Python 3.8 

Java/JVM
--------
PySpark requires a Java Runtime Environment (JRE) or JDK

-. Minimum: Java 8 (1.8)

-. Recommended: OpenJDK 11 or Java 17

-. Make sure JAVA_HOME is set:

$java -version

$export JAVA_HOME=/path/to/java

Apache Spark
------------
-. PySpark 3.x is recommended

-. Spark version should match Hadoop compatibility if using HDFS, but for local mode, any 3.x is fine

-. Apache Spark 3.5.0 (binary pre-built for Hadoop 3.3 or later)

Key Python libraries
---------------------

-. Apache Spark 3.5.0 (binary pre-built for Hadoop 3.3 or later)

-. numpy 1.24+ 

-. pandas 1.6+

-. matplotlib/seaborn latest


Ems

# Exploration of the Data set and results
=========================================

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










