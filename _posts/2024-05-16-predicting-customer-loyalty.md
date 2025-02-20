---
layout: post
title: Predicting Customer Loyalty Using ML
image: "/posts/loyalty_header_new.png"
tags: [Customer Loyalty, Machine Learning, Regression, Python]
---

In this project, we're going to be looking to get an understanding of customer loyalty for our client. Customer loyalty scores could only be found for a proportion of the customer base, so we're going to be using Machine Learning to predict the scores for the remaining customers.

<br>

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Key Definition](#overview-definition)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Linear Regression](#linreg-title)
- [04. Decision Tree](#regtree-title)
- [05. Random Forest](#rf-title)
- [06. Modelling Summary](#modelling-summary)
- [07. Predicting Missing Loyalty Scores](#modelling-predictions)
- [08. Growth & Next Steps](#growth-next-steps)

<br>

___

# Project Overview  <a name="overview-main"></a>

<br>

### Context <a name="overview-context"></a>

In a bid to improve the accuracy, and relevancy, of their customer tracking, targeting, and communications, our client, a grocery retailer, hired a market research consultancy to append market level customer loyalty information to their database. This data would take the form of a score, which measures the proportion of total grocery spend that a given customer will spend with our client vs the competition, regardless of total spend volume.

Unfortunately, the company could only tag around 50% of the customer base and so were still missing a significant amount of customer loyalty information. Therefore, to ensure our client is able to understand the full picture of its customers' loyalty, we were tasked to accurately predict the *loyalty scores* for any customers who could not be tagged.

In order to solve this problem, we're going to use the available customer metrics and *loyalty scores* for those customers who were tagged to help us build out a predictive model. This model will be able to find relationships between our data that can then be used to predict the *loyalty score* metric for any customers who could not be tagged.

<br>

### Key Definition  <a name="overview-definition"></a>

The *loyalty score* metric measures the % of grocery spend, at the market level, that each customer allocates to the client vs all of the competitors.

<u>Example 1:</u> Customer X has a total grocery spend of $100 and all of this is spent with our client. Customer X has a *loyalty score* of 1.0

<u>Example 2:</u> Customer Y has a total grocery spend of $200 but only 20% is spent with our client. The remaining 80% is spend with competitors. Customer Y has a *customer loyalty score* of 0.2

<br>

### Actions <a name="overview-actions"></a>

Our first task is to gather the key customer metrics that may help us to predict the relevant *loyalty scores* and append this data with our dependent variable. Once we have compiled the necessary data from the tables in the database, we can separate out those customers who already have *loyalty scores* (the dependent variable) present, and those who do not.

Given we are looking to predict a numeric output, we will need to consider a regression model, and there are three potential approaches that we will test, namely:

* Linear Regression
* Decision Tree
* Random Forest

<br>

### Results <a name="overview-results"></a>

After training and testing our different modelling approaches, we found that the Random Forest had the highest predictive accuracy.

**Metric 1: Adjusted R-Squared (Test Set)**

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.754

**Metric 2: R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853

While the accuracy of the model will always be important, it may not always be the sole consideration in choosing which model to use. For example, in some cases, it may be more important to explicitly understand the weighted drivers of prediction. However, given the key goal for our work was to provide accurate predictions of the *loyalty score* metric for those customers who were missing this data, we chose to take forward the Random Forest model for our predictions.

<br>

### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high, other modelling approaches could be tested, especially those somewhat similar to Random Forest, e.g. XGBoost or LightGBM, to see if even more accuracy could be gained.

Another consideration would be the input variables that were available for our model. For example, we may want to look at whether there is any additional relevant data available, or if further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty.

<br>
<br>

___

# Data Overview  <a name="data-overview"></a>

The data we will be using is held in multiple tables within the client database. For our dependent variable, i.e. the *loyalty score* metric that we will be predicting, we can, for the 50% of customers where this data exists, find it in the *loyalty_scores* table.

For the customer metrics we hypothesise will be able to predict the missing loyalty scores, we will also extract data from the *transactions* and *customer_details* tables within the client database.

Before joining all of these tables together, we can aggregate some of our sales data from the *transactions* table to engineer some more useful features for our model. Once this is complete, we will then use pandas in Python to merge these tables together for all customers, creating a single dataset that can be used for modelling.

```python
# import required packages
import pandas as pd
import pickle

# import required data tables
loyalty_scores = ...
customer_details = ...
transactions = ...

# merge loyalty score data and customer details data, at customer level
data_for_regression = pd.merge(customer_details, loyalty_scores, how = "left", on = "customer_id")

# aggregate sales data from transactions table
sales_summary = transactions.groupby("customer_id").agg({"sales_cost" : "sum",
                                                         "num_items" : "sum",
                                                         "transaction_id" : "nunique",
                                                         "product_area_id" : "nunique"}).reset_index()

# rename columns for clarity
sales_summary.columns = ["customer_id", "total_sales", "total_items", "transaction_count", "product_area_count"]

# engineer an average basket value column for each customer
sales_summary["average_basket_value"] = sales_summary["total_sales"] / sales_summary["transaction_count"]

# merge the sales summary with the overall customer data
data_for_regression = pd.merge(data_for_regression, sales_summary, how = "inner", on = "customer_id")

# split out data for modelling (loyalty score is present)
regression_modelling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]

# split out data for scoring post-modelling (loyalty score is missing)
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]

# for scoring set, drop the loyalty score column (as it is blank/redundant)
regression_scoring.drop(["customer_loyalty_score"], axis = 1, inplace = True)

# save our datasets for future use
pickle.dump(regression_modelling, open("data/customer_loyalty_modelling.p", "wb"))
pickle.dump(regression_scoring, open("data/customer_loyalty_scoring.p", "wb"))
```

Once we have completed our data preprocessing in Python, we are left with the dataset described below, which we will use for modelling.

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| loyalty_score | Dependent | The % of total grocery spend that each customer allocates to ABC Grocery vs. competitors |
| distance_from_store | Independent | "The distance in miles from the customers home address, and the store" |
| gender | Independent | The gender provided by the customer |
| credit_score | Independent | The customers most recent credit score |
| total_sales | Independent | Total spend by the customer in ABC Grocery within the latest 6 months |
| total_items | Independent | Total products purchased by the customer in ABC Grocery within the latest 6 months |
| transaction_count | Independent | Total unique transactions made by the customer in ABC Grocery within the latest 6 months |
| product_area_count | Independent | The number of product areas within ABC Grocery that the customers have shopped in within the latest 6 months |
| average_basket_value | Independent | The average spend per transaction for the customer in ABC Grocery within the latest 6 months |

<br>
<br>

___

# Modelling Overview

Based upon the customer metrics we described above, we will now look to build a model to accurately predict the "loyalty_score" metric for those customers that it was possible to tag.

If we are successful in building an accurate model, we will be able to use this to predict the loyalty scores for customers where an initial tagging was not possible.

Given we are looking to predict a numeric output, we will need to consider a regression model, and there are three potential approaches that we will test, namely:

* Linear Regression
* Decision Tree
* Random Forest

<br>
<br>

___

# Linear Regression <a name="linreg-title"></a>

To model our data using Linear Regression, we will utilise the scikit-learn library within Python. The code below is broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>

### Data Import <a name="linreg-import"></a>

Since we saved our modelling data as a pickle file, we can now import this. At this point, we will also remove the id column and ensure our data is shuffled.

```python
# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

# import modelling data
data_for_model = pickle.load(open("data/customer_loyalty_modelling.p", "rb"))

# drop unnecessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)
```

<br>

### Data Preprocessing <a name="linreg-preprocessing"></a>

For Linear Regression, we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* The effect of outliers
* Encoding categorical variables to numeric form
* Multicollinearity & Feature Selection

<br>

##### Missing Values

After an initial review of the data, we see that the number of missing values is extremely low, so instead of applying any imputation (i.e. mean, most common value) we will just remove these rows.

```python
# remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)
```

<br>

##### Outliers

There is no definitive right or wrong way to deal with outliers and, in fact, just because a value is very high or very low, does not necessarily mean it is not correct or even valuable. However, outliers are worth especially careful consideration for a Linear Regression model, as these can significantly hamper the model's ability to generalise well across *all* data.

In this code section, we use **.describe()** from Pandas to investigate the spread of values for each of our predictors. The results of this can be seen in the table below.

| **metric** | **distance_from_store** | **credit_score** | **total_sales** | **total_items** | **transaction_count** | **product_area_count** | **average_basket_value** |
|---|---|---|---|---|---|---|---|
| mean | 2.02 | 0.60 | 1846.50 | 278.30 | 44.93 | 4.31 | 36.78 |
| std | 2.57 | 0.10 | 1767.83 | 214.24 | 21.25 | 0.73 | 19.34 |
| min | 0.00 | 0.26 | 45.95 | 10.00 | 4.00 | 2.00 | 9.34 |
| 25% | 0.71 | 0.53 | 942.07 | 201.00 | 41.00 | 4.00 | 22.41 |
| 50% | 1.65 | 0.59 | 1471.49 | 258.50 | 50.00 | 4.00 | 30.37 |
| 75% | 2.91 | 0.66 | 2104.73 | 318.50 | 53.00 | 5.00 | 47.21 |
| max | 44.37 | 0.88 | 9878.76 | 1187.00 | 109.00 | 5.00 | 102.34 |

Based on this investigation, we see the *max* column values for the *distance_from_store*, *total_sales*, and *total_items* variables to be much higher than the *median* value. For example, the median *distance_to_store* is 1.65 miles, but the maximum is over 44 miles!

As a result of this, we will take steps to remove the necessary outliers in order to facilitate generalisation across the full dataset.

We do this using the "boxplot approach", where we remove any rows where the values within those columns are outside of the interquartile range multiplied by 2.

```python
outlier_investigation = data_for_model.describe()
outlier_columns = ["distance_from_store", "total_sales", "total_items"]

# boxplot approach
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)
```

<br>

##### Split Out Data For Modelling

Our next step is to split our data to ensure we have a clear separation between our predictor variables and our dependent variable, and also to ensure we have distinct training and test datasets to properly evaluate our model later down the line.

We will first create objects **X** and **y**, which will contain only predictor variables, and only the dependent variable, respectively.

Secondly, we will split both of these objects into training and test datasets. For this exercise, we are going to take 80% of the data for training. This will allow us to use the remaining 20% to fairly validate our prediction accuracy based on data that had no influence in the training of the model.

```python
# split data into X and y objects for modelling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

<br>

##### Categorical Predictor Variables

When looking to assess the relationship between an input variable and the dependent variable, if the input variable is not in numeric form, the Linear Regression algorithm will not be able to properly understand the data as it cannot assign a numerical meaning to it. As a result, we will need to transform any of our categorical variables into numeric form.

For our dataset, there is only a single categorical variable to deal with, which is *gender*. This variable has values of "M" for Male, "F" for Female, and "U" for Unknown. Additionally, this variable is nominal data, i.e. it is not *ordered*. For example, Male is not higher or lower than Female, or vice versa.

Therefore, one approach we can take here is to apply One Hot Encoding to the *gender* column. This will allow us to represent our categorical variable as a number of new columns, one per original value, with a new binary value assigned. For example, where we previously had a *gender* value of "F" for Female, we would now have a new *gender_F* column with a value of 1 and a value of 0 in two additional new *gender_M* and *gender_U* columns. We can use these new columns as our input variables and discard the original categorical data column.

You will also note in the code below, that we are using the parameter *drop="first"*. This parameter means that we will remove one of our three new columns, and thus avoid the *dummy variable trap*, which occurs when our newly created columns can perfectly predict each other. When this occurs, we risk breaking our assumption of no multicollinearity, which is a requirement, or at least an important consideration, for some models, including Linear Regression.

Multicollinearity occurs when two or more input variables are *highly* correlated with each other. While this may not necessarily affect the accuracy of predictions generated by our model, it can make it much more difficult to understand the respective importance of each feature in predicting the dependent variable. Additionally, it can undermine the level of confidence we have in any statistics regarding the performance of the model.

One additional consideration for using One Hot Encoder is that, while we will *apply* the logic to both our training and test data, we only want the One Hot Encoding logic to *learn* the rules based on our training data. As a result, you will see in the code that we have applied *fit_transform* to our training data, but only *transform* to our test data. This helps us to avoid any *data leakage* where the test data effectively *learns* information about the training data, and thus we cannot completely trust our model's performance metrics. For example, if we had a slightly different set of unique values for the categorical variable between our training and test data, we may even end up with different columns in our data and thus not be able to properly validate our model.

Once we have completed all of the above steps in the code, we will also turn our training and test objects back into Pandas DataFrames with the relevant column names applied.

```python
# list of categorical variables that need encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse_output = False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)
```

<br>

##### Feature Selection

Feature Selection is the process used to select the input variables that are most important to your Machine Learning task. It can be a very important addition or at least, consideration, in certain scenarios. The potential benefits of Feature Selection are:

* **Improved Model Accuracy** - eliminating noise can help true relationships stand out
* **Lower Computational Cost** - our model becomes faster to train, and faster to make predictions
* **Explainability** - understanding and explaining outputs for stakeholder and customers becomes much easier

There are many ways to apply Feature Selection. These range from simple methods such as a *Correlation Matrix* showing variable relationships, to *Univariate Testing* which helps us understand statistical relationships between variables, and then to even more powerful approaches like *Recursive Feature Elimination (RFE)* which is an approach that starts with all input variables, and then iteratively removes those with the weakest relationships with the output variable.

For our task, we will apply a variation of Recursive Feature Elimination called *Recursive Feature Elimination With Cross Validation (RFECV)* where we split the data into many "chunks" and iteratively train and validate models on each "chunk" separately. This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of these models is. From the suite of model scenarios that are created, the algorithm can determine which provided the best accuracy and thus can infer the best set of input variables to use.

```python
# instantiate RFECV & the model type to be utilised
regressor = LinearRegression()
feature_selector = RFECV(regressor)

# fit RFECV onto our training & test data
fit = feature_selector.fit(X_train, y_train)

# extract & print the optimal number of features
optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

# limit our training & test sets to only include the selected variables
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]
```

The below code then produces a plot that visualises the cross-validated accuracy with each potential number of features:

```python
plt.style.use('seaborn-v0_8-poster')
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()
```

This creates the below plot, which shows us that the highest cross-validated accuracy (0.8625) is actually when we include all eight of our original input variables. This is marginally higher than 6 included variables, and 7 included variables. Thus, we will continue with all 8.

![alt text](/img/posts/lin-reg-feature-selection-plot.png "Linear Regression Feature Selection Plot")

<br>

### Model Training <a name="linreg-model-training"></a>

We will then instantiate and train our Linear Regression model:

```python
# instantiate our model object
regressor = LinearRegression()

# fit our model using our training & test sets
regressor.fit(X_train, y_train)
```

<br>

### Model Performance Assessment <a name="linreg-model-assessment"></a>

<br>

##### Predict On The Test Set

To assess how well our model is predicting on new data, we use the trained model object (here called *regressor*) and ask it to predict the *loyalty_score* variable for the test set.

```python
# predict on the test set
y_pred = regressor.predict(X_test)
```

<br>

##### Calculate R-Squared

R-Squared is a metric that shows the percentage of variance in our output variable *y* that is being explained by our input variable(s) *X*. It is a value that ranges between 0 and 1, with a higher value showing a higher level of explained variance. Another way of explaining this would be to say that, if we had an R-squared score of 0.8 it would suggest that 80% of the variation of our output variable is being explained by our input variables - and something else, or some other variables must account for the other 20%.

To calculate R-squared, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python
# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```

The resulting R-squared score from this is **0.78**.

<br>

##### Calculate Cross Validated R-Squared

An even more powerful and reliable way to assess model performance is to utilise Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we break our data into a number of different "chunks". We then iteratively train the model on all but one of the "chunks", and test the model on the remaining "chunk", until each has had a chance to be the test set.

The result of this is that we are provided a number of test set validation results. We can take the average of these to give a much more robust and reliable view of how our model will perform on new, unseen data.

In the code below, we put this into place. We first specify that we want 4 "chunks", and then we pass in our regressor object, training set, and test set. We also specify the metric we want to assess with, which in this case, will be R-squared. Finally, we take a mean of all four test set results.

```python
# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()
```

The mean cross-validated R-squared score from this is **0.853**.

<br>

##### Calculate Adjusted R-Squared

When applying Linear Regression with *multiple* input variables, the R-squared metric on its own *can* end up being an overinflated view of goodness of fit. This is because each input variable will have an *additive* effect on the overall R-squared score. In other words, every input variable added to the model *increases* the R-squared value, and *never decreases* it, even if the relationship is by chance.

**Adjusted R-Squared** is a metric that compensates for the addition of input variables and only increases if the variable improves the model above what would be obtained by probability. It is best practice to use Adjusted R-Squared when assessing the results of a Linear Regression with multiple input variables, as it gives a fairer perception of the fit of the data.

```python
# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```

The resulting *adjusted* R-squared score from this is **0.754** which as expected, is slightly lower than the score we got for R-squared on its own.

<br>

### Model Summary Statistics <a name="linreg-model-summary"></a>

Although our overall goal for this project is predictive accuracy, rather than an explicit understanding of the relationships of each of the input variables and the output variable, it is always interesting to look at the summary statistics for these.

```python
# extract model coefficients
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names, coefficients], axis = 1)
summary_stats.columns = ["input_variable", "coefficient"]

# extract model intercept
regressor.intercept_
```

The information from that code block can be found in the table below:

| **input_variable** | **coefficient** |
|---|---|
| intercept | 0.516 |
| distance_from_store | -0.201 |
| credit_score | -0.028 |
| total_sales | 0.000 |
| total_items | 0.001 |
| transaction_count | -0.005 |
| product_area_count | 0.062 |
| average_basket_value | -0.004 |
| gender_M | -0.013 |

If we had a single input variable, the coefficient of this variable, in conjunction with the intercept, would yield the equation for a line of best fit. For two input variables, this would become a plane of best fit. For our model, and others with higher dimensionality, the coefficients and intercept effectively generalise to a hyperplane of best fit, although naturally this becomes very difficult for us to visualise.

The coefficients themselves help us to understand the impact of changing each of the respective input variables. That is, with *everything else remaining constant*, a coefficient tells us how many units our output variable, i.e. the *loyalty_score* in our case, would change with a *one unit change* in the associated input variable.

To provide an example of this - in the table above, we can see that the *distance_from_store* input variable has a coefficient value of -0.201. This is saying that *loyalty_score* decreases by 0.201 (or 20% as loyalty score is a percentage, or at least a decimal value between 0 and 1) for *every additional mile* that a customer lives from the store. This makes intuitive sense, as customers who live a long way from this store, most likely live near *another* store where they might do some of their shopping as well, whereas customers who live near this store, probably do a greater proportion of their shopping at this store, and hence have a higher loyalty score.

<br>
<br>

___

# Decision Tree <a name="regtree-title"></a>

To model our data using a Decision Tree, we will again utilise the scikit-learn library within Python. We will also use a similar structure to before where the code will be broken up into the 4 key sections listed below:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>

### Data Import <a name="regtree-import"></a>

Since we saved our modelling data as a pickle file, we can now import this. At this point, we will also remove the id column and ensure our data is shuffled.

```python
# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# import modelling data
data_for_model = pickle.load(open("data/customer_loyalty_modelling.p", "rb"))

# drop unnecessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)
```

<br>

### Data Preprocessing <a name="regtree-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables, Decision Trees are not, so the required preprocessing here is lighter. We still however will put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>

##### Missing Values

After our initial review of the data, we saw that the number of missing values was extremely low, so instead of applying any imputation (i.e. mean, most common value), we will again just remove these rows.

```python
# remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)
```

<br>

##### Split Out Data For Modelling

For the Decision Tree model, we will split our data in exactly the same way as we did for Linear Regression. Firstly, we will split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. Once again, we allocate 80% of the data for training, and the remaining 20% for validation.

```python
# split data into X and y objects for modelling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

<br>

##### Categorical Predictor Variables

Just like the Linear Regression algorithm, the Decision Tree cannot deal with categorical data where it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As we saw previously, our dataset contains one categorical variable, *gender*, which has values of "M" for Male, "F" for Female, and "U" for Unknown. Additionally, we know this variable is nominal data, i.e. it is not *ordered*, and so once again, we can apply One Hot Encoding to this column.

```python
# list of categorical variables that need encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)
```

<br>

### Model Training <a name="regtree-model-training"></a>

We will then instantiate and train our Decision Tree model with the code below. We use the *random_state* parameter to ensure we get reproducible results, and this helps us to understand any improvements in performance with changes to model hyperparameters.

```python
# instantiate our model object
regressor = DecisionTreeRegressor(random_state = 42)

# fit our model using our training & test sets
regressor.fit(X_train, y_train)
```

<br>

### Model Performance Assessment <a name="regtree-model-assessment"></a>

<br>

##### Predict On The Test Set

To assess how well our model is predicting on new data, we use the trained model object (here called *regressor*) and ask it to predict the *loyalty_score* variable for the test set.

```python
# predict on the test set
y_pred = regressor.predict(X_test)
```

<br>

##### Calculate R-Squared

To calculate R-squared, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test).

```python
# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```

The resulting R-squared score from this is **0.898**.

<br>

##### Calculate Cross Validated R-Squared

As we did when testing Linear Regression, we will again utilise Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we break our data into a number of different "chunks". We then iteratively train the model on all but one of the "chunks", and test the model on the remaining "chunk", until each has had a chance to be the test set.

The result of this is that we are provided a number of test set validation results. We can take the average of these to give a much more robust and reliable view of how our model will perform on new, unseen data.

In the code below, we put this into place. We again specify that we want 4 "chunks", and then we pass in our regressor object, training set, and test set. We also specify the metric we want to assess with, in this case, we stick with R-squared. Finally, we take a mean of all four test set results.

```python
# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()
```

The mean cross-validated R-squared score from this is **0.871**, which is slighter higher than we saw for Linear Regression.

<br>

##### Calculate Adjusted R-Squared

Just like we did with Linear Regression, we will also calculate the *Adjusted R-Squared*, which compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.

```python
# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```

The resulting *adjusted* R-squared score from this is **0.887**, which as expected, is slightly lower than the score we got for R-squared on its own.

<br>

### Decision Tree Regularisation <a name="regtree-model-regularisation"></a>

Decision Trees can be prone to overfitting. Without any limits on the amount of splitting that occurs in a Decision Tree, we will end up with a model that learns the training data perfectly. For example, we could end up with a branch for each individual data point within our Decision Tree. As a result, it will be much more difficult for us to obtain a reliable set of predictions on any *new* data, and thus we look to have a more *generalised* set of rules that will make our model more robust.

One effective method of avoiding this overfitting, is to apply a *max depth* to the Decision Tree, meaning we only allow it to split the data a certain number of times before it is required to stop.

Unfortunately, we don't necessarily know the *best* number of splits to use for this, so below we will loop over a variety of values and assess which gives us the best predictive performance.

```python
# finding the best max_depth

# set up range for search, and empty list to append accuracy scores to
max_depth_list = list(range(1,9))
accuracy_scores = []

# loop through each possible depth, train and validate model, append test set accuracy
for depth in max_depth_list:
    
    regressor = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
# store max accuracy, and optimal depth    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# plot accuracy by max depth
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()
```

That code also gives us the below plot, which visualises the results.

![alt text](/img/posts/regression-tree-max-depth-plot.png "Decision Tree Max Depth Plot")

The plot shows that the *maximum* classification accuracy on the test set is found when applying a *max_depth* value of 7. However, given we lose very little accuracy with a *max depth* value of 4, and especially since this would result in a simpler model, which would generalise even better on new data, we make the executive decision to retrain our Decision Tree with a maximum depth of 4.

<br>

### Visualise Our Decision Tree <a name="regtree-visualise"></a>

At this point, we can also use the plot_tree functionality, that we imported from scikit-learn, to see the decisions that have been made in the (refitted) tree.

```python
# re-fit our model using max depth of 4
regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)
regressor.fit(X_train, y_train)

# plot the nodes of the decision tree
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 16)
```

That code gives us the plot below:

![alt text](/img/posts/regression-tree-nodes-plot.png "Decision Tree Max Depth Plot")

This is a very powerful visual, and one that can be shown to stakeholders in the business to ensure they understand exactly what is driving the predictions.

One interesting thing to note, is that the *very first split* appears to be using the variable, *distance from store*, so it would seem that this is a very important variable when it comes to predicting loyalty!

<br>
<br>

___

# Random Forest <a name="rf-title"></a>

We will again utilise the scikit-learn library within Python to model our data using a Random Forest. We will also maintain the 4 key code sections below:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>

### Data Import <a name="rf-import"></a>

Again, since we saved our modelling data as a pickle file, we can now import this. We will, once again, remove the id column and ensure that our data is shuffled.

```python
# import required packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# import modelling data
data_for_model = pickle.load(open("data/customer_loyalty_modelling.p", "rb"))

# drop unnecessary columns
data_for_model.drop("customer_id", axis = 1, inplace = True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)
```

<br>

### Data Preprocessing <a name="rf-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables, Random Forests, just like Decision Trees, are not, so the required preprocessing here is lighter. We still however will put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>

##### Missing Values

We saw from our previous review of the data, that the number of missing values is extremely low, so instead of applying any imputation (i.e. mean, most common value), we will, again, just remove these rows.

```python
# remove rows where values are missing
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)
```

<br>

##### Split Out Data For Modelling

For our Random Forest model, we will split our data in exactly the same way we did for Linear Regression. Firstly, we will split our data into an **X** object, which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training. Once again, we allocate 80% of the data for training, and the remaining 20% for validation.

```python
# split data into X and y objects for modelling
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

<br>

##### Categorical Predictor Variables

Just like the Linear Regression algorithm, Random Forests cannot deal with categorical data in where it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As we saw previously, our dataset contains one categorical variable, *gender*, which has values of "M" for Male, "F" for Female, and "U" for Unknown. Additionally, we know this variable is nominal data, i.e. it is not *ordered*, and so once again, we can apply One Hot Encoding to this column.

```python
# list of categorical variables that need encoding
categorical_vars = ["gender"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)
```

<br>

### Model Training <a name="rf-model-training"></a>

We will then instantiate and train our Random Forest model using the below code. We use the *random_state* parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

We leave the other parameters at their default values, meaning that we will just be building 100 Decision Trees in this Random Forest.

```python
# instantiate our model object
regressor = RandomForestRegressor(random_state = 42)

# fit our model using our training & test sets
regressor.fit(X_train, y_train)
```

<br>

### Model Performance Assessment <a name="rf-model-assessment"></a>

<br>

##### Predict On The Test Set

To assess how well our model is predicting on new data, we use the trained model object (here called *regressor*) and ask it to predict the *loyalty_score* variable for the test set.

```python
# predict on the test set
y_pred = regressor.predict(X_test)
```

<br>

##### Calculate R-Squared

To calculate R-squared, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test).

```python
# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```

The resulting R-squared score from this is **0.957** - higher than both Linear Regression and the Decision Tree.

<br>

##### Calculate Cross Validated R-Squared

As we did when testing Linear Regression & our Decision Tree, we will again utilise Cross Validation (for more details on how this works, please refer to the Linear Regression section above).

```python
# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()
```

The mean cross-validated R-squared score from this is **0.923**, which again is higher than we saw for both Linear Regression and our Decision Tree.

<br>

##### Calculate Adjusted R-Squared

Just like we did with Linear Regression and our Decision Tree, we will also calculate the *Adjusted R-Squared*, which compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.

```python
# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```

The resulting *adjusted* R-squared score from this is **0.955**, which as expected, is slightly lower than the score we got for R-squared on its own, but is again higher than for our other models.

<br>

### Feature Importance <a name="rf-model-feature-importance"></a>

In our Linear Regression model, to understand the relationships between input variables and our output variable, loyalty score, we examined the coefficients. With our Decision Tree, we looked at what the earlier splits were. These allowed us some insight into which input variables were having the most impact.

Random Forests are an ensemble model, made up of many, many Decision Trees, each of which is different due to the randomness of the data being provided, and the random selection of input variables available at each potential split point.

Because of this, we end up with a powerful and robust model, but because of the random or different nature of all these Decision trees, the model gives us a unique insight into how important each of our input variables are to the overall model.

As we’re using random samples of data, and input variables for each Decision Tree, there are many scenarios where certain input variables are being held back, and this enables us a way to compare how accurate the model's predictions are if that variable is or isn’t present.

So, at a high level, in a Random Forest we can measure *importance* by asking, *how much would accuracy decrease if a specific input variable was removed or randomised?*

If this decrease in performance, or accuracy, is large, then we’d deem that input variable to be quite important, and if we see only a small decrease in accuracy, then we’d conclude that the variable is of less importance.

At a high level, there are two common ways to tackle this. The first, often just called **Feature Importance**, is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the Mean Squared Error (for a Regression problem) was before the split was made, and compare this to the Mean Squared Error after the split was made. We can take the *average* of these improvements across all Decision Trees in the Random Forest to get a score that tells us *how much better* we’re making the model by using that input variable.

If we do this for *each* of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model.

The other approach, often called **Permutation Importance**, cleverly uses some data that has gone *unused* at the point when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping").

These observations, which were not randomly selected for each Decision Tree, are known as *Out of Bag* observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the *Out of Bag* observations are gathered and then passed through. Once all of these observations have been run through the Decision Tree, we obtain an accuracy score for these predictions, which in the case of a regression problem could be Mean Squared Error or R-squared.

In order to understand the *importance*, we *randomise* the values within one of the input variables. This process essentially destroys any relationship that might exist between that input variable and the output variable. We can then run that updated data through the Decision Tree again, obtaining a second accuracy score. The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

*Permutation Importance* is often preferred over *Feature Importance*, which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.

Let's put them both in place and plot the results.

```python
# calculate feature importance
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

# plot feature importance
plt.barh(feature_importance_summary["input_variable"], feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# calculate permutation importance
result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)
permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

# plot permutation importance
plt.barh(permutation_importance_summary["input_variable"], permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()
```

That code gives us the below plots - the first being for *Feature Importance* and the second for *Permutation Importance*.

![alt text](/img/posts/rf-regression-feature-importance.png "Random Forest Feature Importance Plot")

![alt text](/img/posts/rf-regression-permutation-importance.png "Random Forest Permutation Importance Plot")

The overall story from both approaches is very similar, in that, by far, the most important or impactful input variable is *distance_from_store*, which is the same insight we derived when assessing our Linear Regression and Decision Tree models.

There are slight differences in the order, or *importance*, for the remaining variables, but overall, they have provided similar findings.

<br>
<br>

___

# Modelling Summary  <a name="modelling-summary"></a>

The most important outcome for this project was predictive accuracy, rather than explicitly understanding the drivers of prediction. Based upon this, we chose the model that performed the best when predicted on the test set - the Random Forest.

**Metric 1: Adjusted R-Squared (Test Set)**

* Random Forest = 0.955
* Decision Tree = 0.886
* Linear Regression = 0.754

**Metric 2: R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.925
* Decision Tree = 0.871
* Linear Regression = 0.853

Even though we were not specifically interested in the drivers of prediction, it was interesting to see across all three modelling approaches, that the input variable with the biggest impact on the prediction was *distance_from_store*, rather than variables such as *total sales*. This is interesting information for the business, so discovering this as we went was worthwhile.

<br>

# Predicting Missing Loyalty Scores <a name="modelling-predictions"></a>

After all the work above, we have been able to select the best model to take forward, namely the Random Forest. However, we still need to predict the *loyalty_scores* for those customers that the market research consultancy was unable to tag.

Before we do this, we also need to prepare the data that we will input into our model. We need to ensure the data is in exactly the same format as the data that was used to train the model.

In the following code, we will:

* Import the required packages for preprocessing
* Import the data for those customers who are missing a *loyalty_score* value
* Import our model object & any preprocessing artifacts
* Drop columns that were not used when training the model (customer_id)
* Drop rows with missing values
* Apply One Hot Encoding to the gender column (using transform only)
* Make the predictions using .predict()

```python
# import required packages
import pandas as pd
import pickle

# import customers for scoring
to_be_scored = ...

# import model and model objects
regressor = ...
one_hot_encoder = ...

# drop unused columns
to_be_scored.drop(["customer_id"], axis = 1, inplace = True)

# drop missing values
to_be_scored.dropna(how = "any", inplace = True)

# apply one hot encoding (transform only)
categorical_vars = ["gender"]
encoder_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars])
encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)
encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)
to_be_scored = pd.concat([to_be_scored.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis = 1)
to_be_scored.drop(categorical_vars, axis = 1, inplace = True)

# make our predictions!
loyalty_predictions = regressor.predict(to_be_scored)
```

Just like that, we have made our *loyalty_score* predictions for these missing customers. Due to the impressive metrics on the test set, we can be reasonably confident with these scores. This extra customer information will ensure our client can undertake more accurate and relevant customer tracking, targeting, and communications.

<br>
<br>

___

# Growth & Next Steps <a name="growth-next-steps"></a>

While predictive accuracy was relatively high, other modelling approaches could be tested, especially those somewhat similar to Random Forest, e.g. XGBoost, LightGBM, to see if even more accuracy could be gained.

We could even look to tune the hyperparameters of the Random Forest, notably regularisation parameters such as tree depth, as well as potentially training on a higher number of Decision Trees in the Random Forest.

Another consideration would be the input variables that were available for our model. For example, we may want to look at whether there is any additional relevant data available, or if further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty.

