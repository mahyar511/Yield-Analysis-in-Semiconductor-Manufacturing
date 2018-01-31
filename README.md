# Yield Analysis in Semiconductor Manufacturing Process
<br>
<div style="text-align: justify">
A complex modern semiconductor manufacturing process is normally under consistent monitoring of signals/variables collected from sensors and process measurements. However, not all of these signals are equally valuable in a specific monitoring system. The measured signals contain a combination of useful information, irrelevant information as well as noise. It is often the case that useful information is buried in the later two. Engineers typically have a much larger number of signals than are actually required. If we consider each type of signal as a feature, then feature selection may be applied to identify the most relevant signals. The Process Engineers may then use these signals to determine key factors contributing to yield excursions downstream in the process. This will enable an increase in process throughput, decreased time to learning and reduce the per unit production costs.
<br><br>
<p align="center">
<img src="Figures/process.gif"  width="40%">  
</p>
<p align="center">
Figure 1. [Basic Semiconductor Manufacturing Process] (http://blog.associatie.kuleuven.be/danhuayao/introduction-of-the-metallic-contamination/)
</p>

<br>
In this project,  [SECOM] (http://archive.ics.uci.edu/ml/datasets/secom/) data-set is first screened in order to identify effective parameters on semiconductor production yield. Then, more analysis is conducted to bring more insight from the data and recommend optimization potential throughout the process. At the end, machine learning technique is used to develop a data-driven model for yield prediction at final stage of fabrication, based on operation data and sensor measurements gathered throughout the process. This notebook is organized as follows: </div>

# Table of Contents
1. [Data-Set Description](#DSD)
2. [Dimension Reduction](#DR)
3. [Exploratory Data Analysis (EDA)](#EDA)
4. [Statistical Analysis & Hypothesis Testing](#SAHT)
5. [Time Series Analysis](#TSA)
6. [Imbalanced Data](#ID)
7. [Machine Learning Model Development](#MLMD)
8. [XGB Model Optimization](#XGBMO)
9. [Final Note](#FN)



### <a name="DSD"></a>1. Data-Set Description
 
<div style="text-align: justify">
The SECOM data-set comes in 2 separate files. "secom_data", which is consisting of 1567 examples each with 591 features a 1567 x 591 matrix and "secom_labels", which is containing the classification labels and date time stamp for each example.
Each example represents a single production entity with associated measured features and the labels represent a simple pass/fail yield for in house line testing and associated date time stamp. Where –1 corresponds to a pass and 1 corresponds to a fail and the data time stamp is for that specific test point. The SECOM data-set is anonymized, which results in no feature identification. In addition, all categorical data is converted to numerical value. <div> 
<bt>

### <a name="DR"></a> 2. Dimension Reduction
<div style="text-align: justify">
As mentioned earlier, there are 591 features collected for each product, but only fraction of them are really significant in yield analysis and the rest are trivial or correlated. One way to reduce the dimension of unnecessary data is to employ Lasso regularization technique. This technique identifies feature significance based on its variance. Therefore, features with smaller variances (less significant) will vanish over the course of regularization. Using this technique decreases number of features and usually considered as a first step in feature selection.   

<br>
<p align="center">
<img src="Figures/LASSO-01.png"  width="40%"> 
Figure 2. Feature Reduction Via Lasso Regularization
</p>  
<br>

In LASSO regularization technique, by tuning alpha (regularization rate) one can determine how many features to remain in the data-set. It is recommended to examine range of regularization rates to reach the optimal value. As shown above, in current case alpha = 0.2 is chosen as the optimal value, which results in drastic reduction in number of features from 591 to 41! <div>
 
 ### <a name="EDA"></a> 3- Exploratory Data Analysis (EDA)
<div style="text-align: justify">
We start exploring selected features by making correlation coefficient pair plot. As shown below, there are few correlated features in remaining data-set. If our desired machine learning technique is prone to correlated data then this issue needs to be addressed before feeding data to the ML model. But more importantly, this plot shows no significant correlation between any of these features and final label column. This observation can raise concern that the collected data might not be a good descriptor of the final label. <div> 
<br>
<p align="center">
<img src="Figures/pcp.png"  width="60%"> 
</p>  
<div style="text-align: center">
Figure 3. Correlation Coefficient Pair Plot
<div>
<br>
 <div style="text-align: justify">
Understanding data variations and outliers is the next step in exploratory data analysis. Box plot can visually represent both concepts in a concise way. As it is evident, some of these features vary couple order of magnitudes and almost all of them suffer from outliers. These are significant issues which needs to be considered later on. Unfortunately because of the anonymity of data-set, it is very difficult to understand nature of these outliers, in order to address them accordingly. It is very important that in model selection take all these factors into account. <div>
<br>
<p align="center">
<img src="Figures/BoxPlot.png" > 
</p>  
<div style="text-align: center">
Figure 4. Selected Features Box Plot  
<div>
<br>
