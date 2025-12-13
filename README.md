# CS777-termproject
# by Xioaning Zhang, Pearlrose Nwuke, and Boris Godoy


# Description of the files
==========================

We basically have 1 python file (flight_prediction_v02.py)

We also have 1 file, which is a sample version of the entire dataset.

# Logistic regression classifier - See section 3A
----------------------------------------------------

This part of the code calculates the logistic regression classifier, and also 
it calculates its performance for a small grid of hyperparameter values.

The pipeline is self-contained, and the user only needs to run
the file in a terminal :

$python flight_prediction_v02.py 

The performance metrics will print out at the end.

# Random Forest classifier - See section 3B
-----------------------------------------------

Because the pipeline for the pre-processing and feature
engineering are common to the all the classifiers, then 
rather than making 3 separate files, we have created only
1 file that can obtain the classifier sequentially

In Section 3B, we have the random forest classifier, and also 
it calculates its performance.

The pipeline is self-contained, and the user only needs to run
the file in a terminal :

$python flight_prediction_v02.py 

The performance metrics will print out at the end.


# Gradient-boosted trees  - See section 3C
----------------------------------------------

We run this classifier after the other 2, logistic regression
and randon forest have been obtained.

The pipeline is self-contained, and the user only needs to run
the file in a terminal :

$python flight_prediction_v02.py 

The results of the software will print out at the end.

The user only to make sure to add the correct path at the beginning of each
file.

path = "/Documents/..../your_flight_dataset.csv " or
alternative whatever location you have on the cloud e.g.
path = path = "gs://data-flights/flights_sample_3m.csv"

Once the user has set up the correct path, then it can run the whole
pipeline, or simply run it by secions in Jupyter notebook (user needs 
to copy/past the sections). Notice that section 1 and 2 are common to 
sections 3A, 3B, and 3C. These ones only depend upon sections 1, and 2.

# Aspects about the file

# 1st section 
We read the dataset and extract the rows of interest for the
classifier.

All relevant data is in the variable :

selected = rows.map(safe_convert).filter(lambda x: x is not None)

# 2nd section 
We create a dataframe to use the embedded library  for classifiers.

We then do feature engineering, that is, creating some new features from
the original ones. 


# 3rd section, A,B, and C
Here we calculate our respective classifiers (lr, rf, gbt), and then the
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

-. Minimum: Java 11

-. Recommended: OpenJDK 11 or Java 17

-. Make sure JAVA_HOME is set:

$java -version

$export JAVA_HOME=/path/to/java

Apache Spark
------------
-. PySpark 3.x recommended

-. Spark version should match Hadoop compatibility if using HDFS, but for local mode, any 3.x is fine

-. Apache Spark 3.5.0 (binary pre-built for Hadoop 3.3 or later)

Key Python libraries
---------------------

-. Apache Spark 3.5.0 (binary pre-built for Hadoop 3.3 or later)

-. numpy 1.24+ 

-. pandas 1.6+

-. matplotlib/seaborn latest



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


# Results

We have basically 3 tables to show, and they are produced by sections 3A, 3B, and 3C, 
respectively.


![image Alt]

![image Alt]

![image Alt]

![image Alt](https://github.com/borgod/CS777-termproject/blob/main/results_project.png?raw=true)









