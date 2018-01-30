# Yield Analysis in Semiconductor Manufacturing Process
<br>
<div style="text-align: justify">
A complex modern semiconductor manufacturing process is normally under consistent monitoring of signals/variables collected from sensors and process measurements. However, not all of these signals are equally valuable in a specific monitoring system. The measured signals contain a combination of useful information, irrelevant information as well as noise. It is often the case that useful information is buried in the later two. Engineers typically have a much larger number of signals than are actually required. If we consider each type of signal as a feature, then feature selection may be applied to identify the most relevant signals. The Process Engineers may then use these signals to determine key factors contributing to yield excursions downstream in the process. This will enable an increase in process throughput, decreased time to learning and reduce the per unit production costs.
<br><br>
<p align="center">
<img src="Figures/process.gif"  width="40%">  
</p>
<div style="text-align: center">

Figure 1. [Basic Semiconductor Manufacturing Process](http://blog.associatie.kuleuven.be/danhuayao/introduction-of-the-metallic-contamination/)


</div>

<br>
In this project,  [SECOM](http://archive.ics.uci.edu/ml/datasets/secom/) data-set is first screened in order to identify effective parameters on semiconductor production yield. Then, more analysis is conducted to bring more insight from the data and recommend optimization potential throughout the process. At the end, machine learning technique is used to develop a data-driven model for yield prediction at final stage of fabrication, based on operation data and sensor measurements gathered throughout the process. This notebook is organized as follows: </div>

# Table of Contents

<b>
1. [Data-Set Description](#DSD)<br>
2. Dimension Reduction<br>
3. Exploratory Data Analysis (EDA)<br>
4. Statistical Analysis & Hypothesis Testing<br>
5. Time Series Analysis<br>
6. Imbalanced Data<br>
7. Machine Learning Model Development<br>
8. XGB Model Optimization<br>
9. Final Note<br>
</b>



<br>
<a id="DSD"></a>
### 1- Data-Set Description 
<div style="text-align: justify">
The SECOM data-set comes in 2 separate files. "secom_data", which is consisting of 1567 examples each with 591 features a 1567 x 591 matrix and "secom_labels", which is containing the classification labels and date time stamp for each example.
Each example represents a single production entity with associated measured features and the labels represent a simple pass/fail yield for in house line testing and associated date time stamp. Where –1 corresponds to a pass and 1 corresponds to a fail and the data time stamp is for that specific test point. The SECOM data-set is anonymized, which results in no feature identification. In addition, all categorical data is converted to numerical value. <div> 
