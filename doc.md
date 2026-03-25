# Streaming Data Analysis Project Report

**Version:** Detailed

**Course:** Data Analytics using Python  
**Project Title:** Streaming Data Analysis using Machine Learning and Exploratory Data Analysis  
**Student Name:** [Add Name]  
**Roll Number:** [Add Roll Number]  
**Institution:** [Add College / University Name]  
**Submission Date:** [Add Date]

## Abstract

This project presents a structured data analytics study in the domain of video streaming services using the **Netflix Customer Churn Dataset**. The work applies exploratory data analysis, statistical interpretation, regression modeling, classification, and performance evaluation using Python. The primary goal is to understand customer behavior, engagement patterns, and churn tendencies through a real-world structured CSV dataset. The study also demonstrates how core concepts from data analytics such as missing value analysis, outlier detection, spread of data, supervised learning, and model evaluation can be connected to practical implementation.

The analysis is conducted through a reproducible workflow implemented in Python and documented through tables, visualizations, and model performance metrics. Special care is taken to preserve the raw dataset without artificial modification. The final outcome is an academically structured report that integrates theory, implementation, and interpretation in a manner suitable for university-level submission.

---

## Suggested Preliminary Pages for Word / Google Docs

If your university expects a formal report format, add these pages before the main report:

### Certificate / Declaration Page

- [Add university certificate text if required]
- [Add student declaration if required]

### Acknowledgement

I would like to express my sincere gratitude to my faculty guide, department, and institution for providing the guidance and resources necessary to complete this project. I also acknowledge the open-source data community and Kaggle for making the dataset publicly available for academic use.

### Table of Contents

- [Generate automatically in Google Docs / Word after pasting the report]

---

## 1. Introduction

The streaming industry has become one of the most data-driven sectors in the digital economy. Platforms such as Netflix, Amazon Prime Video, Hulu, JioHotstar, and Disney+ rely on customer behavior data to understand viewing habits, improve recommendations, reduce churn, and optimize pricing or subscription strategies.

This project focuses on analyzing a streaming-domain dataset using Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn. The academic task sheet originally uses retail-style variables such as *Sales*, *Profit*, *Discount*, and *Quantity*. Since this project uses a streaming-service dataset, equivalent streaming variables are used wherever appropriate.

From an academic standpoint, this project is significant because it demonstrates how a general data analytics workflow can be adapted to a domain-specific dataset. It covers descriptive analytics, inferential interpretation, predictive modeling, and performance evaluation in a single integrated study.

### 1.1 Variable Mapping Used in This Project

- `Sales` equivalent -> `watch_hours`
- engagement-related predictors -> `avg_watch_time_per_day`, `last_login_days`, `monthly_fee`, `number_of_profiles`
- classification target -> `churned`

### 1.2 Objectives of the Study

- To understand the structure and properties of a real-world streaming dataset
- To perform univariate, bivariate, and multivariate exploratory data analysis
- To examine missing values and outliers
- To study data distribution using statistical measures
- To automate EDA using reusable Python functions
- To build and evaluate regression and classification models
- To interpret model behavior using standard performance metrics

### 1.3 Problem Statement

Streaming platforms must understand how customer engagement, subscription behavior, and usage patterns influence the likelihood of churn. The central problem addressed in this project is whether customer-level behavioral and subscription variables can be used to explain engagement and predict churn with acceptable analytical confidence.

### 1.4 Scope of the Study

The scope of this project is limited to structured tabular analysis using one publicly available dataset. The study does not include recommendation systems, time-series forecasting, deep learning models, or production deployment. Its primary purpose is educational and analytical.

### 1.5 Limitations of the Study

- The dataset represents one platform-focused customer view rather than a full cross-platform market analysis.
- The dataset is observational and does not establish causal relationships.
- The report relies on available fields in the dataset; therefore, content metadata, marketing spend, and customer support behavior are not considered.
- Model performance is limited by the variables provided in the source dataset.

---

## 2. Dataset Description

### 2.1 Dataset Information

- **Dataset Name:** Netflix Customer Churn Dataset
- **Source:** Kaggle
- **Data Type:** Structured CSV dataset
- **Domain:** Video streaming / OTT customer analytics
- **File Used:** `netflix_customer_churn.csv`

### 2.2 Feature Overview

The dataset contains customer-level records with demographic, behavioral, subscription, and payment-related information. Key columns include:

- `customer_id`
- `age`
- `gender`
- `subscription_type`
- `watch_hours`
- `last_login_days`
- `region`
- `device`
- `monthly_fee`
- `churned`
- `payment_method`
- `number_of_profiles`
- `avg_watch_time_per_day`
- `favorite_genre`

### 2.3 Relevance of the Dataset

This dataset is appropriate for the project because it supports:

- structured data understanding
- exploratory data analysis
- outlier identification
- regression modeling
- classification modeling
- model performance evaluation

### 2.4 Add in Report

- [Insert DataFrame: dataset preview showing the first 5 rows]
- [Insert DataFrame: dataset preview showing the last 5 rows]
- [Insert Output Screenshot: dataset shape and column names]

### 2.5 Dataset Suitability for the Project

The chosen dataset aligns well with the academic requirements because it contains both numerical and categorical variables, includes a clearly defined classification target (`churned`), and offers sufficient observations for exploratory analysis and model training. It is especially suitable for demonstrating regression, classification, and visualization-based interpretation in a single project.

---

## 3. Tools and Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## 4. Methodology

The analysis was performed in the following sequence:

1. Data loading and dataset inspection
2. Variable classification and statistical understanding
3. Exploratory data analysis using visual methods
4. Missing value and outlier detection
5. Distribution analysis using descriptive statistics
6. Regression modeling
7. Classification modeling
8. Overfitting and underfitting analysis
9. Model evaluation and interpretation

### 4.1 Analytical Design

The project adopts a descriptive-to-predictive workflow. It begins with data familiarization, proceeds to exploratory and statistical analysis, and then transitions into predictive modeling. This sequence ensures that modeling decisions are supported by prior understanding of data quality, distributions, and relationships.

### 4.2 Reporting Strategy

For submission purposes, the final report should contain a balanced mix of:

- explanation of concepts
- selected code snippets
- tables of outputs
- screenshots of important graphs
- concise analytical interpretation

The report should not include every line of code. Instead, only the most relevant code cells and outputs should be included in the main body, while additional code can be placed in an appendix if needed.

---

## 5. Task 1: Data Understanding

### 5.1 Objective

The objective of this task is to inspect the dataset structure and classify its variables into quantitative and qualitative categories.

### 5.2 Work Performed

- Loaded the CSV dataset using Pandas
- Displayed the first 5 rows
- Displayed the last 5 rows
- Determined dataset shape
- Listed all column names
- Examined data types using `info()`
- Generated descriptive statistics using `describe()`

### 5.3 Variable Classification

#### 5.3.1 Quantitative Variables

- **Discrete:** `age`, `last_login_days`, `number_of_profiles`, `churned`
- **Continuous:** `watch_hours`, `monthly_fee`, `avg_watch_time_per_day`

#### 5.3.2 Qualitative Variables

- **Nominal:** `customer_id`, `gender`, `region`, `device`, `payment_method`, `favorite_genre`
- **Ordinal:** `subscription_type`

### 5.4 Add in Report

- [Insert Code Block: dataset loading and initial inspection]
- [Insert DataFrame: first 5 rows]
- [Insert DataFrame: last 5 rows]
- [Insert Output Screenshot: dataset shape and column names]
- [Insert Output Screenshot: `df.info()`]
- [Insert DataFrame: `df.describe()` output]

### 5.5 Interpretation

The dataset contains a balanced mixture of numerical and categorical variables, which makes it suitable for descriptive analysis as well as predictive modeling.

### 5.6 Academic Writing Note

When writing this section in the final document, emphasize that data understanding forms the foundation of the full analytical pipeline. This section should establish the reader's confidence that the dataset has been properly inspected before advanced analysis begins.

---

## 6. Task 2: Exploratory Data Analysis (EDA)

### 6.1 Objective

The purpose of EDA is to discover patterns, distributions, and relationships among variables before modeling.

### 6.2 Univariate Analysis

Univariate analysis was performed on major numerical variables such as:

- `watch_hours`
- `monthly_fee`
- `avg_watch_time_per_day`
- `last_login_days`

#### Techniques Used

- Histogram
- Boxplot
- Count plots for categorical variables

### 6.3 Bivariate Analysis

Relationships between important variables were studied using scatter plots and correlation analysis.

#### Variable Combinations Studied

- `watch_hours` vs `avg_watch_time_per_day`
- `watch_hours` vs `last_login_days`
- `watch_hours` vs `monthly_fee`

### 6.4 Multivariate Analysis

Multivariate analysis was used to examine relationships across several variables at once.

#### Techniques Used

- Correlation matrix
- Heatmap
- Pair plot
- Boxplots across categories such as:
  - `subscription_type`
  - `favorite_genre`
  - `region`

### 6.5 Add in Report

- [Insert Code Block: EDA plotting code]
- [Insert Figure: histogram of `watch_hours`]
- [Insert Figure: boxplot of `watch_hours`]
- [Insert Figure: histogram/boxplot of `monthly_fee`]
- [Insert Figure: scatter plot `watch_hours` vs `avg_watch_time_per_day`]
- [Insert Figure: scatter plot `watch_hours` vs `last_login_days`]
- [Insert DataFrame: correlation matrix]
- [Insert Figure: correlation heatmap]
- [Insert Figure: pair plot]
- [Insert Figure: grouped boxplots by subscription type / region / genre]

### 6.6 Interpretation

Exploratory analysis showed that engagement-related variables are closely linked to churn behavior. Highly engaged users tend to differ from churned users in meaningful ways, especially with respect to watch time and inactivity.

### 6.7 Academic Writing Note

This section should not merely describe graphs. Each paragraph should explain what pattern is observed, why that pattern matters, and how it supports later modeling decisions.

---

## 7. Task 3: Handling Missing Data and Outliers

### 7.1 Objective

This task focuses on checking data completeness and identifying unusual values that may influence statistical interpretation or model performance.

### 7.2 Missing Value Analysis

Missing values were examined using `isnull().sum()` for all columns.

### 7.3 Outcome

No missing values were found in the raw dataset. Therefore, no imputation was applied.

### 7.4 Outlier Detection

Outliers were identified using:

- boxplots
- interquartile range (IQR) method

### 7.5 Effect of Outliers

Outliers can:

- distort averages and standard deviations
- increase skewness
- affect regression coefficients
- influence prediction stability

### 7.6 Add in Report

- [Insert Code Block: missing value and outlier analysis code]
- [Insert DataFrame: missing value summary]
- [Insert DataFrame: outlier summary table]
- [Insert Figure: boxplots for important numerical variables]

### 7.7 Interpretation

Although the dataset has no missing values, some variables such as `watch_hours` and `avg_watch_time_per_day` contain extreme values, which should be interpreted carefully during modeling.

### 7.8 Academic Writing Note

Clearly mention that the raw dataset was preserved as-is. This is important because it demonstrates methodological transparency and avoids the impression that values were altered to improve model results artificially.

---

## 8. Task 4: Spread of Data

### 8.1 Objective

This task studies the distribution and variability of numerical variables using descriptive statistics.

### 8.2 Statistical Measures Used

- Mean
- Median
- Standard deviation
- Skewness
- Kurtosis

### 8.3 Interpretation Framework

- A distribution with skewness near zero is relatively symmetric
- Positive skewness indicates a longer right tail
- Negative skewness indicates a longer left tail
- High kurtosis indicates heavier tails and more extreme values

### 8.4 Add in Report

- [Insert Code Block: distribution statistics code]
- [Insert DataFrame: mean, median, standard deviation, skewness, and kurtosis table]

### 8.5 Interpretation

The variables `watch_hours` and `avg_watch_time_per_day` exhibit substantial positive skewness, suggesting that a smaller portion of users consume content much more heavily than the rest of the population.

### 8.6 Academic Writing Note

Use this section to connect statistical measures with real customer behavior. For example, high positive skewness can be interpreted as evidence that a small number of highly active users account for a disproportionate share of engagement.

---

## 9. Task 5: Automating EDA using Python

### 9.1 Objective

This task demonstrates how repetitive EDA operations can be automated using built-in Pandas functions and reusable Python functions.

### 9.2 Built-in Functions Used

- `describe()`
- `info()`
- `isnull()`
- `corr()`

### 9.3 Reusable Functions Developed

- histogram and boxplot function
- count plot function
- scatter plot function
- outlier summary function
- distribution statistics function

### 9.4 Add in Report

- [Insert Code Block: helper functions used for EDA automation]
- [Insert Output Screenshot: `describe()`]
- [Insert Output Screenshot: `info()`]
- [Insert Output Screenshot: `isnull()`]
- [Insert Output Screenshot: `corr()`]

### 9.5 Interpretation

EDA automation improves reproducibility, reduces repetition, and makes the analysis more consistent and easier to extend.

### 9.6 Academic Writing Note

This section demonstrates programming maturity. In the final report, describe how automation reduces manual repetition and makes the notebook or script easier to maintain and reuse.

---

## 10. Task 6: Regression Analysis

### 10.1 Objective

The objective of this task is to identify dependent and independent variables and study their linear relationship.

### 10.2 Variables Selected

- **Dependent variable:** `watch_hours`
- **Independent variable for simple linear regression:** `avg_watch_time_per_day`

### 10.3 Statistical Relationship Examined

- Covariance between `watch_hours` and `avg_watch_time_per_day`
- Correlation between `watch_hours` and `avg_watch_time_per_day`

### 10.4 Add in Report

- [Insert Code Block: covariance, correlation, and regression setup]
- [Insert Output Screenshot: covariance and correlation values]

### 10.5 Interpretation

The positive relationship between average daily watch time and total watch hours supports the use of regression for engagement prediction.

### 10.6 Academic Writing Note

The discussion here should remain concise and technically focused. Mention whether the relationship is weak, moderate, or strong based on the computed correlation value.

---

## 11. Task 7: Supervised Learning - Regression Model

### 11.1 Objective

This task builds predictive models for engagement using supervised learning techniques.

### 11.2 Data Splitting Strategy

The dataset was divided into:

- training set
- validation set
- testing set

### 11.3 Models Developed

- Simple Linear Regression
- Multiple Linear Regression
- Logistic Regression

### 11.4 Features Used in Multiple Linear Regression

- `age`
- `last_login_days`
- `monthly_fee`
- `number_of_profiles`
- `avg_watch_time_per_day`

### 11.5 Add in Report

- [Insert Code Block: dataset split into training, validation, and test sets]
- [Insert Code Block: simple and multiple regression models]
- [Insert DataFrame: regression results table]
- [Insert Figure: simple linear regression fit plot]

### 11.6 Interpretation

The multiple linear regression model performs better than the simple linear regression model because it incorporates more explanatory variables relevant to user engagement.

### 11.7 Academic Writing Note

This section should explicitly connect feature selection to model performance. It should also mention that validation and test sets were used to assess generalization rather than relying only on training performance.

---

## 12. Task 8: Overfitting and Underfitting Analysis

### 12.1 Objective

This task examines how model complexity influences training and testing performance.

### 12.2 Conceptual Explanation

- **Underfitting** occurs when the model is too simple and cannot capture the pattern in the data.
- **Overfitting** occurs when the model becomes too tailored to training data and fails to generalize well to unseen data.

### 12.3 Analytical Method

Polynomial regression with multiple degrees was used to compare training and testing mean squared error.

### 12.4 Add in Report

- [Insert Code Block: model complexity analysis]
- [Insert DataFrame: polynomial degree vs train/test MSE]
- [Insert Figure: model complexity vs error plot]

### 12.5 Interpretation

Comparing training and testing errors across polynomial degrees helps illustrate the trade-off between simplicity and predictive flexibility.

### 12.6 Academic Writing Note

When drafting the final report, explain overfitting and underfitting conceptually first, then support the explanation using the error comparison table and graph.

---

## 13. Task 9: Classification Task

### 13.1 Objective

The aim of this task is to convert the problem into a classification setting and predict whether a customer is likely to churn.

### 13.2 Target Variable

- `churned`
  - `0` = customer retained
  - `1` = customer churned

### 13.3 Classification Model Used

- Logistic Regression

### 13.4 Evaluation Measures

- Accuracy
- Confusion Matrix

### 13.5 Add in Report

- [Insert Code Block: logistic regression training code]
- [Insert DataFrame: classification accuracy on train/validation/test]
- [Insert Figure: confusion matrix]
- [Insert DataFrame: confusion matrix values]

### 13.6 Interpretation

The classification model provides a strong basis for identifying churn risk and supports the practical use of supervised learning in the streaming domain.

### 13.7 Academic Writing Note

Use formal language to explain the confusion matrix. Rather than simply listing the numbers, state what they imply about the model's ability to distinguish churned and retained customers.

---

## 14. Task 10: Model Evaluation

### 14.1 Objective

This task evaluates the performance of regression models using standard error metrics.

### 14.2 Metrics Used

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R2 Score)

### 14.3 Comparative Focus

The regression models were compared on test performance to determine which model better predicts engagement.

### 14.4 Add in Report

- [Insert Code Block: regression evaluation code]
- [Insert DataFrame: final regression metrics table]

### 14.5 Interpretation

The multiple linear regression model achieved better performance than the simple linear regression model, indicating that engagement is influenced by multiple features rather than a single predictor.

### 14.6 Academic Writing Note

This section should compare model results rather than repeating metric definitions alone. Emphasize the relative improvement in predictive quality.

---

## 15. Task 11: Data Visualization

### 15.1 Objective

The objective of this task is to present visual evidence of data behavior, feature relationships, and model performance.

### 15.2 Visual Outputs Included

- univariate analysis plots
- bivariate analysis plots
- multivariate analysis plots
- correlation heatmap
- pair plot
- grouped boxplots
- confusion matrix
- model complexity plot

### 15.3 Add in Report

- [Insert Figure: univariate analysis plots]
- [Insert Figure: bivariate analysis plots]
- [Insert Figure: multivariate analysis plots]
- [Insert Figure: correlation heatmap]
- [Insert Figure: pair plot]
- [Insert Figure: confusion matrix]
- [Insert Figure: model complexity vs error plot]

### 15.4 Interpretation

The visualizations complement the statistical analysis and make the results more interpretable for both academic and practical discussion.

### 15.5 Academic Writing Note

Use only the most relevant figures in the main report body. Avoid inserting every generated plot. Select figures that best support the analytical story of the project.

---

## 16. Results and Discussion

### 16.1 Summary of Statistical Findings

The statistical analysis indicates that user engagement is unevenly distributed. Variables such as `watch_hours` and `avg_watch_time_per_day` show substantial positive skewness, which implies that a smaller group of users contributes disproportionately to total consumption. In contrast, variables such as `age` and `last_login_days` display more balanced distributions.

The correlation analysis shows that churn is negatively associated with engagement and positively associated with inactivity. This suggests that customers who spend less time watching content and remain inactive for longer durations are more likely to churn.

### 16.2 Summary of Modeling Findings

The regression analysis demonstrates that multiple linear regression produces better performance than simple linear regression, although the overall explanatory power remains moderate. This implies that user engagement is influenced by multiple factors and not entirely captured by a single variable.

The logistic regression classifier performs strongly in identifying churn outcomes. The confusion matrix indicates that the model is able to correctly classify a large proportion of both retained and churned users, making it useful for academic demonstration of supervised classification.

### 16.3 Practical Implications

From a business perspective, the findings suggest that engagement variables can serve as useful indicators of customer retention risk. Users with lower watch hours and longer inactivity periods may benefit from targeted interventions such as reminders, content recommendations, or promotional retention campaigns.

---

## 17. Key Findings

- The dataset contains no missing values in raw form.
- Engagement-related features such as `watch_hours` and `avg_watch_time_per_day` are central to understanding user behavior.
- Churn has a meaningful relationship with inactivity and lower engagement.
- Multiple linear regression performs better than simple linear regression on test data.
- Logistic regression provides strong churn prediction accuracy.
- Visual analysis supports the statistical conclusions drawn from the dataset.

---

## 18. Conclusion

This project successfully applies Python-based data analysis and supervised learning methods to a streaming-service dataset. The study covers the full analytical workflow: data understanding, EDA, missing value inspection, outlier detection, distribution analysis, regression modeling, classification, and model evaluation.

From an academic perspective, the project demonstrates the practical application of descriptive statistics, data visualization, predictive modeling, and evaluation metrics. From a domain perspective, it shows how customer streaming behavior can be analyzed to understand engagement and churn patterns. Overall, the project connects theory with implementation and provides a strong foundation for further work in OTT analytics and customer behavior modeling.

### 18.1 Final Submission Note

Before final submission in Word or PDF format, ensure that the document contains:

- title page details
- a table of contents
- selected code snippets
- key tables and screenshots
- observations written in paragraph form
- conclusion and references

---

## 19. References

- Kaggle: Netflix Customer Churn Dataset
- Python documentation for Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn
- Project requirements provided in `tasks.md`

---

## Appendix Guidance

Use the following approach while converting this markdown into Google Docs or Word:

- paste code only where a task explicitly requires methodology or implementation evidence
- insert tables as screenshots or pasted DataFrames with captions
- insert graphs with figure numbers such as *Figure 1*, *Figure 2*, and so on
- add short captions below every table and figure
- use paragraph-based interpretation instead of bullet-only explanations in the final version

### Suggested Caption Style

- **Table 1:** Dataset preview showing the first five observations
- **Table 2:** Summary statistics of numerical variables
- **Figure 1:** Distribution of watch hours using histogram and boxplot
- **Figure 2:** Correlation heatmap of numerical variables
- **Figure 3:** Confusion matrix for logistic regression model
