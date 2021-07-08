# Bankruptcy-Prediction

## Objective
The objective of the project is to use the features (i.e., financial parameters) given in the dataset to understand their impact in identifying whether a company will face bankruptcy in the future or not.

## Business Understanding

### Firm Description
**CTBC BANK**
* Established in 1966 is one of the largest private banks in Taiwan, headquartered in Taipei.

**Number of Employees**                 
* 11,000 employees

**Geographic Footprint of the Firm**    
* 152 branches in Taiwan
* 116 overseas branches across 14 countries such as the US, Canada, Japan, Indonesia, the Philippines, India, Thailand, Vietnam, Malaysia, Hong Kong, Singapore, China, Myanmar, and Australia.

**Major Product or Service Line**
* Institutional banking, capital markets, and overseas business
* Retail banking

**Revenues tied to each Product or Service line**

  ![image](https://user-images.githubusercontent.com/60916305/124686152-332e4e00-de98-11eb-97a7-d2fb49495943.png)

**Line of business that is the subject of Analysis**
* Retail Banking- Loan services:

  Small and medium-sized enterprises are provided with diverse loan services including land loan, factory loan, loan for purchase of machinery and equipment,  refinancing,     installment loans, policy-based loans, financial planning mortgages, unsecured term loans, and revolving loans, export loan, purchase turnover financing.

* Global Risk Management Group:

  Responsible for regulatory compliance such as Credit Risk Management. 
  This includes maintenance of minimum regulatory capital as per central bank regulations and submitting reports for the same.

## SWOT Analysis
  ![image](https://user-images.githubusercontent.com/60916305/124686524-ea2ac980-de98-11eb-8abd-2e01949b16c1.png)

## Data Description
  Dataset is from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction

  Deron Liang and Chih-Fong Tsai, deronliang'@'gmail.com; cftsai'@’mgt.ncu.edu.tw, National Central University, Taiwan
  Company bankruptcy was defined based on the business regulations of the Taiwan Stock Exchange.

  The data is collected from the Taiwan Economic Journal for the years 1999 to 2009

## Data Analysis and Understanding
  Number of records:                      6819

  Number of features/ variables:          96

  Target variable of interest:            Bankrupt?

  ![image](https://user-images.githubusercontent.com/60916305/124687037-ccaa2f80-de99-11eb-957c-3eb2c16a3e4b.png)
  
  ![image](https://user-images.githubusercontent.com/60916305/124687049-d03db680-de99-11eb-8f05-fb9f2d6df6c8.png)

### Dataset Fields and Description
  ![image](https://user-images.githubusercontent.com/60916305/124687089-e0ee2c80-de99-11eb-968e-37cd4214d6f3.png)
  
  ![image](https://user-images.githubusercontent.com/60916305/124687100-e3e91d00-de99-11eb-9a82-7d9f0bdfa3ab.png)
  
  ![image](https://user-images.githubusercontent.com/60916305/124687113-ec415800-de99-11eb-946b-ce227c53de64.png)
  
  ![image](https://user-images.githubusercontent.com/60916305/124687118-ef3c4880-de99-11eb-85ce-77c8fe813ce8.png)

### Summarize the dataset
**Target Variable Distribution**
* We can observe that our dataset is very imbalanced. 
* The minority class which is the one we're most interested by predicting represents about 3% of total observations.
![image](https://user-images.githubusercontent.com/60916305/124687544-bb155780-de9a-11eb-94a7-04196691f977.png)

![image](https://user-images.githubusercontent.com/60916305/124687554-bea8de80-de9a-11eb-9b60-610e3f672d97.png)

**Target variable correlation**

![image](https://user-images.githubusercontent.com/60916305/124687569-c5375600-de9a-11eb-9ed5-7f735d6ae893.png)

**Correlation Matrix**

* One thing to point out is that there are groups of features that appear highly correlated with each other as well as the label. 

![image](https://user-images.githubusercontent.com/60916305/124687789-3840cc80-de9b-11eb-919f-a0bc2acb4957.png)

**Data Statistics**

![image](https://user-images.githubusercontent.com/60916305/124687810-41ca3480-de9b-11eb-8577-75a9961e42a1.png)

**Histograms**

![image](https://user-images.githubusercontent.com/60916305/124688211-04b27200-de9c-11eb-8573-43f841181c73.png)

![image](https://user-images.githubusercontent.com/60916305/124688219-07ad6280-de9c-11eb-989b-f19d7e6fdd7b.png)

**Boxplots**
* Major values are concentrated around starting ranges yet there are very high valued records.

* Some features show outliers in top 1% values only. Few of such features are:
  Total_debt/Total_net_worth
  Revenue_per_person
  Net_Value_Growth_Rate
  Revenue_Per_Share etc

* There are some features that have significant number of higher values, like:
  Current_Asset_Turnover_Rate
  Cash_Turnover_Rate

![image](https://user-images.githubusercontent.com/60916305/124688322-30355c80-de9c-11eb-8763-688eb5bcae8b.png)

![image](https://user-images.githubusercontent.com/60916305/124688336-33c8e380-de9c-11eb-9a7a-cc505ace87bc.png)

## Data Modeling
### Prediction: 

  To predict if a company will go bankrupt or not.
  
### Hope to Accomplish:

  We hope to build a classification model that will predict the bankruptcy and help lenders, shareholders and investors to make informed decisions.
  
### Target variable Description: Bankrupt? 

  Target variable determines if a company is bankrupt or not.
  1 - bankrupt
  0 - not bankrupt
  
### Classification models:

  Decision Trees 
  
  Random Forest Classifier
  
  Logistic Regression

### Data Transformation: Each model is run with the following.

  Original data
  
  Scaled data
  
  SMOTE
 
### Analysis of metrices such as accuracy, confusion matrix to determine the best model for the prediction.

### Data Split:
  
  ![image](https://user-images.githubusercontent.com/60916305/124689212-b900c800-de9d-11eb-8589-34861b7a1708.png)

### Number of Missing Values : 0

  ![image](https://user-images.githubusercontent.com/60916305/124689275-d6359680-de9d-11eb-8b43-12361c7e2fee.png)

## Logistic Regression Model

**Summary Statistics**

  ![image](https://user-images.githubusercontent.com/60916305/124689552-51974800-de9e-11eb-9207-cec2d9452885.png)

**ROC Curve**

  ![image](https://user-images.githubusercontent.com/60916305/124689619-725f9d80-de9e-11eb-9460-9305e47377a1.png)

## Decision Tree Model (Scaled Data)
  TRANSFORMATION: Scaling
  Max-depth: 30
  
**Decision Tree**

  ![image](https://user-images.githubusercontent.com/60916305/124690077-4690e780-de9f-11eb-8ece-28e85f21e932.png)

**Summary Statistics**

  ![image](https://user-images.githubusercontent.com/60916305/124690125-54466d00-de9f-11eb-9c03-462ad8c9b2f0.png)

**ROC Curve**

  ![image](https://user-images.githubusercontent.com/60916305/124690147-5c9ea800-de9f-11eb-8c3f-30ba91d8de92.png)

## Random Forest Model (SMOTE)
  Transformation: SMOTE algorithm for unbalanced classification problems

  This function handles unbalanced classification problems using the SMOTE method. 

**SMOTE Dataset**

  ![image](https://user-images.githubusercontent.com/60916305/124690257-86f06580-de9f-11eb-9ea4-7755fca04644.png)

**Target Variable Distribution (SMOTE Dataset)**

  ![image](https://user-images.githubusercontent.com/60916305/124690295-91aafa80-de9f-11eb-90f9-73b7b1efbfa5.png)

**Summary Statistics**

  ![image](https://user-images.githubusercontent.com/60916305/124690328-a091ad00-de9f-11eb-866b-4f5c149704f9.png)

**ROC Curve**

  ![image](https://user-images.githubusercontent.com/60916305/124690342-a7202480-de9f-11eb-964a-6b5791279db9.png)

## Model Performance Comparison
**ROC Curves**

  ![image](https://user-images.githubusercontent.com/60916305/124690403-c0c16c00-de9f-11eb-83d3-d1bbcb2ab126.png)

**Model Performance Metrices**

  ![image](https://user-images.githubusercontent.com/60916305/124690424-ccad2e00-de9f-11eb-9e05-be1c3fd26ee8.png)

**Final Model**
  LOGISTIC REGRESSION

## Prediction with Test data using Logistic Regression (Final Model)

**Summary Statistics**

  ![image](https://user-images.githubusercontent.com/60916305/124690563-0847f800-dea0-11eb-9541-b39b7e4bb038.png)

**ROC Curve**

  ![image](https://user-images.githubusercontent.com/60916305/124690606-11d16000-dea0-11eb-8ab2-b4bb96b3edc4.png)

**Variable Importance Plot**

  ![image](https://user-images.githubusercontent.com/60916305/124690650-201f7c00-dea0-11eb-9981-647291cbe638.png)

## Limitations
* Most companies tend to submit flawed statements or are limited by the availability
* Assumption that the models are stable across economic conditions that change over time, such as inflation, interest rates, and credit availability
* The companies which gets predicted as bankrupt by the model but is not actually bankrupt leads to loss of potential business and thereby profits for the bank
* The companies which are classified as not bankrupt by the model but are bankrupt lead to risk of default for the bank

## Improvement Scope
* The data contains outliers in almost all the variables. Corrective measures in each one of them can drive increased performance of the model.
* Other classification algorithms such as K-nearest neighbours, Naive Bayes, SVM etc
* More accurate predictions can be made from separate industry-based models
* Factor the current macroeconomic conditions like growth rates, inflation, and interest rates into the model

## Recommendations
### CTBC Bank can use this model to evaluate a company’s  financial stability before establishing new relationships or engagements with a company.

![image](https://user-images.githubusercontent.com/60916305/124691465-793bdf80-dea1-11eb-9656-9d6437b1383f.png)

* They can decide if the loan requested by a company should be sanctioned or not depending on if it classifies as bankrupt or not by the model. 
* Furthermore, this model can also be used to set interest rates for the loan issued. 
* For example, if a company gets classified as bankrupt by the model and CTBC still wants to sanction the loan, they can quote a higher interest rate to take into account the risk of being bankrupt and hence default of loan.

### CTBC Bank can also use this model to assess the financial distress of companies that they have already lend to.

![image](https://user-images.githubusercontent.com/60916305/124691668-c9b33d00-dea1-11eb-883b-6d631c578253.png)

* The prediction of a company as bankrupt or not can be used to set aside exposure amounts which are subject to default. 
* This model can also be used by CTBC Bank to determine the floating interest rate for the loan which can be increased or decreased based on bankruptcy prediction.

### CTBC Bank can utilize this model to calculate the minimum regulatory capital required to be maintained by every bank as per Central Bank of Republic of China (Taiwan)

![image](https://user-images.githubusercontent.com/60916305/124691743-f10a0a00-dea1-11eb-8dd4-f0e64dacd5b0.png)
































