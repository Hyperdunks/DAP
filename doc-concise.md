# Streaming Data Analysis Project Report

**Version:** Concise  
**Course:** Data Analytics using Python  
**Project Title:** Streaming Data Analysis using Exploratory Data Analysis and Machine Learning  
**Student Name:** [Add Name]  
**Roll Number:** [Add Roll Number]  
**Institution:** [Add College / University Name]  
**Submission Date:** [Add Date]

---

## Abstract

This project analyzes a streaming-domain dataset using Python-based data analytics and machine learning techniques. The selected dataset, the **Netflix Customer Churn Dataset**, is used to study customer engagement, inactivity behavior, and churn patterns. The project includes data understanding, exploratory data analysis, missing value inspection, outlier detection, spread of data analysis, regression modeling, classification, and model evaluation. The overall aim is to connect academic concepts such as EDA, supervised learning, and performance measurement with practical implementation.

---

## 1. Introduction

Streaming platforms generate large volumes of customer behavior data that can be analyzed to improve engagement and retention. This project uses a structured CSV dataset from the streaming domain and applies core data analytics techniques to understand how customer-related attributes influence viewing behavior and churn.

The original task sheet uses retail-style variables such as *Sales*, *Profit*, *Discount*, and *Quantity*. Since the present study uses a streaming dataset, domain-equivalent variables are used:

- `Sales` equivalent -> `watch_hours`
- engagement predictors -> `avg_watch_time_per_day`, `last_login_days`, `monthly_fee`, `number_of_profiles`
- classification target -> `churned`

### 1.1 Objectives

- To understand the structure of a real-world streaming dataset
- To perform univariate, bivariate, and multivariate EDA
- To inspect missing values and outliers
- To analyze data distribution using descriptive statistics
- To build regression and classification models
- To evaluate predictive performance using standard metrics

---

## 2. Dataset Description

- **Dataset Name:** Netflix Customer Churn Dataset
- **Source:** Kaggle
- **Type:** Structured CSV dataset
- **Domain:** Video streaming / OTT analytics
- **File Used:** `netflix_customer_churn.csv`

### 2.1 Key Variables

- `age`
- `subscription_type`
- `watch_hours`
- `last_login_days`
- `monthly_fee`
- `churned`
- `number_of_profiles`
- `avg_watch_time_per_day`
- `favorite_genre`

### 2.2 Add in Report

- [Insert DataFrame: first 5 rows]
- [Insert DataFrame: last 5 rows]
- [Insert Output Screenshot: dataset shape and columns]

---

## 3. Tools and Libraries Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## 4. Methodology

The project follows a structured sequence:

1. Data loading and inspection
2. Data understanding and variable classification
3. Exploratory data analysis
4. Missing value and outlier analysis
5. Statistical spread analysis
6. Regression modeling
7. Classification modeling
8. Model evaluation and visualization

---

## 5. Task 1: Data Understanding

### Objective

To inspect the dataset structure and classify the variables.

### Work Performed

- Loaded the dataset using Pandas
- Displayed the first 5 rows and last 5 rows
- Checked dataset shape and column names
- Examined data types using `info()`
- Generated summary statistics using `describe()`

### Variable Types

**Quantitative (Discrete):** `age`, `last_login_days`, `number_of_profiles`, `churned`  
**Quantitative (Continuous):** `watch_hours`, `monthly_fee`, `avg_watch_time_per_day`  
**Qualitative (Nominal):** `customer_id`, `gender`, `region`, `device`, `payment_method`, `favorite_genre`  
**Qualitative (Ordinal):** `subscription_type`

### Add in Report

- [Insert Code Block: dataset loading and inspection]
- [Insert DataFrame: `head()`]
- [Insert DataFrame: `tail()`]
- [Insert Output Screenshot: `info()`]
- [Insert DataFrame: `describe()`]

---

## 6. Task 2: Exploratory Data Analysis (EDA)

### Objective

To identify patterns, distributions, and relationships among variables.

### Univariate Analysis

The following variables were analyzed individually:

- `watch_hours`
- `monthly_fee`
- `avg_watch_time_per_day`
- `last_login_days`

**Techniques Used:** histogram, boxplot, count plot

### Bivariate Analysis

Relationships examined:

- `watch_hours` vs `avg_watch_time_per_day`
- `watch_hours` vs `last_login_days`
- `watch_hours` vs `monthly_fee`

**Techniques Used:** scatter plots, correlation matrix

### Multivariate Analysis

**Techniques Used:** pair plot, heatmap, grouped boxplots by `subscription_type`, `favorite_genre`, and `region`

### Add in Report

- [Insert Figure: histogram and boxplot of `watch_hours`]
- [Insert Figure: scatter plots]
- [Insert DataFrame: correlation matrix]
- [Insert Figure: correlation heatmap]
- [Insert Figure: pair plot]

---

## 7. Task 3: Handling Missing Data and Outliers

### Objective

To examine missing values and detect outliers in numerical variables.

### Findings

- Missing values were checked using `isnull().sum()`
- No missing values were found in the raw dataset
- Outliers were identified using boxplots and the IQR method

### Interpretation

Although the dataset is complete, some variables such as `watch_hours` and `avg_watch_time_per_day` contain extreme values that may influence statistical summaries and model coefficients.

### Add in Report

- [Insert Code Block: missing value and outlier analysis]
- [Insert DataFrame: missing value summary]
- [Insert DataFrame: outlier summary]
- [Insert Figure: boxplots]

---

## 8. Task 4: Spread of Data

### Objective

To study the distribution and variability of numerical variables.

### Statistical Measures Used

- Mean
- Median
- Standard deviation
- Skewness
- Kurtosis

### Interpretation

The spread analysis showed that some engagement-related variables are positively skewed, indicating that a smaller group of users contributes disproportionately to total watch activity.

### Add in Report

- [Insert Code Block: spread of data statistics]
- [Insert DataFrame: mean, median, standard deviation, skewness, kurtosis]

---

## 9. Task 5: Automating EDA using Python

### Objective

To automate repeated EDA operations using built-in functions and reusable helper functions.

### Functions Used

- `describe()`
- `info()`
- `isnull()`
- `corr()`

### Reusable Functions Created

- histogram and boxplot function
- count plot function
- scatter plot function
- outlier summary function
- distribution statistics function

### Add in Report

- [Insert Code Block: reusable EDA helper functions]
- [Insert Output Screenshot: selected automated outputs]

---

## 10. Task 6: Regression Analysis

### Objective

To identify the dependent and independent variables and study their linear relationship.

### Variables Used

- **Dependent variable:** `watch_hours`
- **Independent variable:** `avg_watch_time_per_day`

### Analysis Performed

- Covariance calculation
- Correlation calculation
- Simple linear regression setup

### Add in Report

- [Insert Code Block: covariance, correlation, and regression setup]
- [Insert Output Screenshot: covariance and correlation values]

---

## 11. Task 7: Supervised Learning - Regression Model

### Objective

To build predictive models for engagement using training, validation, and testing data.

### Models Used

- Simple Linear Regression
- Multiple Linear Regression
- Logistic Regression

### Multiple Regression Features

- `age`
- `last_login_days`
- `monthly_fee`
- `number_of_profiles`
- `avg_watch_time_per_day`

### Add in Report

- [Insert Code Block: data splitting and model training]
- [Insert DataFrame: regression results]
- [Insert Figure: regression fit plot]

---

## 12. Task 8: Overfitting and Underfitting Analysis

### Objective

To compare training and testing errors across different model complexities.

### Analysis Performed

- Polynomial models of different degrees were compared
- Training and testing MSE were analyzed

### Interpretation

This analysis demonstrates the balance required between model simplicity and generalization.

### Add in Report

- [Insert Code Block: model complexity comparison]
- [Insert DataFrame: train vs test MSE]
- [Insert Figure: model complexity vs error]

---

## 13. Task 9: Classification Task

### Objective

To predict customer churn using supervised classification.

### Classification Setup

- **Target variable:** `churned`
- **Model used:** Logistic Regression

### Evaluation Measures

- Accuracy
- Confusion Matrix

### Add in Report

- [Insert Code Block: logistic regression model]
- [Insert DataFrame: classification accuracy]
- [Insert Figure: confusion matrix]

---

## 14. Task 10: Model Evaluation

### Objective

To evaluate regression model performance using standard metrics.

### Metrics Used

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R2 Score)

### Interpretation

The multiple linear regression model produced better results than the simple linear regression model, suggesting that engagement is influenced by multiple features.

### Add in Report

- [Insert Code Block: model evaluation code]
- [Insert DataFrame: final regression metrics]

---

## 15. Task 11: Data Visualization

### Objective

To present visual evidence of distributions, relationships, and model outcomes.

### Visuals Included

- univariate plots
- bivariate plots
- multivariate plots
- correlation heatmap
- pair plot
- confusion matrix
- model complexity plot

### Add in Report

- [Insert Figure: selected univariate plots]
- [Insert Figure: selected bivariate plots]
- [Insert Figure: heatmap / pair plot]
- [Insert Figure: confusion matrix]

---

## 16. Key Findings

- The dataset contains no missing values in raw form
- Engagement variables are strongly related to churn behavior
- Inactivity contributes positively to churn tendency
- Multiple linear regression performs better than simple linear regression
- Logistic regression provides strong churn prediction accuracy

---

## 17. Conclusion

This project demonstrates a complete streaming data analytics workflow using Python. It combines descriptive analysis, statistical interpretation, regression modeling, classification, and evaluation within a single structured study. The results show that streaming engagement and churn behavior can be meaningfully analyzed using customer-level variables. Overall, the project successfully connects theoretical concepts from data analytics with practical implementation on a real-world dataset.

---

## 18. References

- Kaggle: Netflix Customer Churn Dataset
- Python documentation for Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn
- Project requirements from `tasks.md`
