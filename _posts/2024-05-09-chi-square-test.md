---
layout: post
title: Assessing Campaign Performance Using Chi-Square Test For Independence
image: "/posts/ab-testing-title-img.png"
tags: [AB Testing, Hypothesis Testing, Chi-Square, Python]
---

In this project, we're going to take a look at the impact of two different promotional mailers that were sent out to promote a new service. To do this we'll be using a Hypothesis Test and more specifically, we will apply a Chi-Square Test for Independence.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Concept Overview](#concept-overview)
- [02. Data Overview & Preparation](#data-overview)
- [03. Applying Chi-Square Test For Independence](#chi-square-application)
- [04. Analysing The Results](#chi-square-results)
- [05. Discussion](#discussion)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Our client, a grocery retailer, recently decided to offer a new premium membership tier to their customers. This new subscription, named "Delivery Club", would entitle members to free grocery deliveries at a cost of \$100 per year, rather than the usual cost of \$10 for each delivery.

In order to advertise their new offering, our client chose to run a promotional campaign. As part of this campaign, the retailer decided to split their customers into three randomly assigned groups, which would allow them to test the success of different promotional approaches. These three approaches were split as follows:

1. Customers who would receive a low cost, low quality mailer
2. Customers who would receive a high cost, high quality mailer
3. A control group of customers who would receive no mailer

Following the campaign, our client has seen a far higher signup rate by customers who were contacted with a mailer, compared to those in the control group. However, in order to optimise their approach to future promotional activities, they would now like to go a step further and understand if there is a significant difference in signup rate between the high quality and low quality mailers. Given the difference in the cost involved in the production of these, the outcome could have a significant impact on their return on investment.

<br>
<br>

### Actions <a name="overview-actions"></a>

To make this assessment, we're going to apply a Chi-Square Test for Independence, as this will allow us to effectively compare the *rates* of the two groups. However, this is not the only option available to us. We could alternatively use the *Z-Test For Proportions* in this scenario, but on this occasion, there are a few key reasons we have stuck to the Chi-Square Test for Independence:

* The resulting test statistic for both tests will be the same
* The Chi-Square Test can be represented using 2x2 tables of data - meaning it can be easier to explain to stakeholders
* The Chi-Square Test can extend out to more than 2 groups - meaning the client can have one consistent approach to measuring signficance

We'll cover the Chi-Square Test for Independence in more detail in the dedicated section later on in this post.

From the data provided by the client, we can use the *campaign_data* table to separate the three groups of customers into those who received the low cost, low quality mailer ("Mailer 1"), those who received the high cost, high quality mailer ("Mailer 2"), and those customers who were part of the control group, which we will exclude from our test.

We can then set out our hypotheses and Acceptance Criteria for the test:

* **Null Hypothesis:** There is no relationship between mailer type and signup rate. They are independent.
* **Alternate Hypothesis:** There is a relationship between mailer type and signup rate. They are not independent.
* **Acceptance Criteria:** 0.05

To meet the requirements of the Chi-Square Test For Independence, we aggregated this data down to a 2x2 matrix for *signup_flag* by *mailer_type*. We can then feed this into the algorithm (using the *scipy* library) to calculate the Chi-Square Statistic, p-value, Degrees of Freedom, and expected values.

<br>
<br>

### Results & Discussion <a name="overview-results"></a>

Taking a look at the data, we can use our observed outcomes to calculate the signup rate of each group:

* Mailer 1 (low cost, low quality): **32.8%** signup rate
* Mailer 2 (high cost, high quality): **37.8%** signup rate

When we apply the Chi-Square Test, we also get the following statistics:

* Chi-Square Statistic: **1.94**
* p-value: **0.16**

The Critical Value for our specified Acceptance Criteria of 0.05 is **3.84**.

Based upon these statistics, we retain the null hypothesis, and conclude that there is not sufficient evidence that the observed difference occurs due to anything other than chance, thus we determine there is no relationship between mailer type and signup rate.

What we are effectively saying here is that, even though we can see that the higher cost/quality Mailer 2 had a higher signup rate (37.8%) compared to the lower cost/quality Mailer 1 (32.8%), based on our predefined Acceptance Criteria of 0.05, this difference is not significant enough to suggest that the type of mailer received had an impact on the decision to sign up.

From our initial observations, the client may have ended up spending more money on higher quality mailers for future promotional campaigns, expecting this to yield a higher signup rate. While the results of our Hypothesis Test do not say there *definitely is not a difference between the two mailers*, it allows us to advise that the evidence does not support any rigid conclusions *at this point in time*. In fact, it shows that the extra investment in a higher quality mailer may not *necessarily* lead to any extra revenue at all. 

Running more A/B Tests like this, gathering more data, and then re-running this test may provide us, and the client more insight!

<br>
<br>

___

# Concept Overview  <a name="concept-overview"></a>

<br>

#### A/B Testing

An A/B Test can be described as a randomised experiment containing two groups, A & B, that receive different experiences. Within an A/B Test, we look to understand and measure the response of each group - and the information from this helps drive future business decisions.

Applications of A/B testing can range from testing different online ad strategies, different email subject lines when contacting customers, or testing the effect of mailing customers a coupon, vs a control group. Companies like Amazon are running these tests in an almost never-ending cycle, testing new website features on randomised groups of customers, all with the aim of finding what works best so they can stay ahead of their competition. Reportedly, Netflix will even test different images for the same film or TV series, to different segments of their customer base, to see if certain images pull more viewers in.

<br>

#### Hypothesis Testing

A Hypothesis Test is used to assess the plausibility, or likelihood of an assumed viewpoint based on sample data - in other words, it helps us assess whether a certain view we have about some data is likely to be true or not.

There are many different scenarios we can run Hypothesis Tests on, and they all have slightly different techniques and formulas, however they all have some shared, fundamental steps and logic that underpin how they work.

<br>

**The Null Hypothesis**

In any Hypothesis Test, we start with the Null Hypothesis. The Null Hypothesis is where we state our initial viewpoint, and in statistics, and specifically Hypothesis Testing, our initial viewpoint is always that the result is purely by chance or that there is no relationship or association between two outcomes or groups.

<br>

**The Alternate Hypothesis**

The aim of the Hypothesis Test is to look for evidence to support or reject the Null Hypothesis. If we reject the Null Hypothesis, that would mean we’d be supporting the Alternate Hypothesis. The Alternate Hypothesis is essentially the opposite viewpoint to the Null Hypothesis - that the result is *not* by chance, or that there *is* a relationship between two outcomes or groups.

<br>

**The Acceptance Criteria**

In a Hypothesis Test, before we collect any data or run any numbers - we specify an Acceptance Criteria. This is a p-value threshold at which we’ll decide to reject or support the null hypothesis. It is essentially a line we draw in the sand saying "if I was to run this test many many times, what proportion of those times would I want to see different results come out, in order to feel comfortable, or confident, that my results are not just some unusual occurrence".

Conventionally, we set our Acceptance Criteria to 0.05 - but this does not have to be the case.  If we need to be more confident that something did not occur through chance alone, we could lower this value down to something much smaller, meaning that we only come to the conclusion that the outcome was special or rare if it’s extremely rare.

So to summarise, in a Hypothesis Test, we test the Null Hypothesis using a p-value and then decide it’s fate based on the Acceptance Criteria.

<br>

**Types Of Hypothesis Test**

There are many different types of Hypothesis Tests, each of which is appropriate for use in differing scenarios - depending on a) the type of data that you’re looking to test and b) the question that you’re asking of that data.

In the case of our task here, where we are looking to understand the difference in signup *rate* between two groups - we will utilise the Chi-Square Test For Independence.

<br>

#### Chi-Square Test For Independence

The Chi-Square Test For Independence is a type of Hypothesis Test that assumes observed frequencies for categorical variables will match the expected frequencies.

The *assumption* is that the Null Hypothesis is, as discussed above, always the viewpoint that the two groups will be equal. With the Chi-Square Test For Independence we look to calculate a statistic which, based on the specified Acceptance Criteria, will mean we either reject or support this initial assumption.

The *observed frequencies* are the true values that we’ve seen.

The *expected frequencies* are essentially what we would *expect* to see based on all of the data.

**Note:** Another option when comparing "rates" is a test known as the *Z-Test For Proportions*.  While, we could absolutely use this test here, we have chosen the Chi-Square Test For Independence because:

* The resulting test statistic for both tests will be the same
* The Chi-Square Test can be represented using 2x2 tables of data - meaning it can be easier to explain to stakeholders
* The Chi-Square Test can extend out to more than 2 groups - meaning the business can have one consistent approach to measuring signficance

___

<br>

# Data Overview & Preparation  <a name="data-overview"></a>

Our goal for this task is to find evidence that the signup rate for the Delivery Club was different for those who received "Mailer 1" (low cost/quality) compared with those who received "Mailer 2" (high cost/quality).

Luckily, within the client database, we have a *campaign_data* table which includes a list of customer ids, alongside the type of mailer they received ("Mailer 1", "Mailer 2", or "Control"), as well as whether they ultimately signed up to the new membership tier. For our purposes, we do not need the control group, so we can remove these customers from our data once loaded.

The code below shows our initial steps to:

* Load in the Python libraries we require for importing the data and performing the Chi-Square Test (using scipy)
* Import the required data from the *campaign_data* table
* Exclude customers in the control group, giving us a dataset with Mailer 1 & Mailer 2 customers only

<br>

```python

# install the required python libraries
import pandas as pd
from scipy.stats import chi2_contingency, chi2

# import campaign data
campaign_data = ...

# remove customers who were in the control group
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]

```
<br>
A sample of this data (the first 10 rows) can be seen below:
<br>
<br>

| **customer_id** | **campaign_name** | **mailer_type** | **signup_flag** |
|---|---|---|---|
| 74 | delivery_club | Mailer1 | 1 |
| 524 | delivery_club | Mailer1 | 1 |
| 607 | delivery_club | Mailer2 | 1 |
| 343 | delivery_club | Mailer1 | 0 |
| 322 | delivery_club | Mailer2 | 1 |
| 115 | delivery_club | Mailer2 | 0 |
| 1 | delivery_club | Mailer2 | 1 |
| 120 | delivery_club | Mailer1 | 1 |
| 52 | delivery_club | Mailer1 | 1 |
| 405 | delivery_club | Mailer1 | 0 |
| 435 | delivery_club | Mailer2 | 0 |

<br>
In the DataFrame we have:

* customer_id
* campaign_name (only the delivery_club campaign data is present)
* mailer_type (either Mailer1 or Mailer2)
* signup_flag (either 1 or 0)

___

<br>

# Applying Chi-Square Test For Independence <a name="chi-square-application"></a>

<br>

#### State Hypotheses & Acceptance Criteria For Test

To kick off our Hypothesis Test, we first need to make sure we have clearly stated our Null Hypothesis, Alternate Hypothesis, and the Acceptance Criteria as described in the section above.

We will code these in explcitly and clearly so we can utilise them later to explain the results. We specify the common Acceptance Criteria value of 0.05.

```python

# specify hypotheses & acceptance criteria for test
null_hypothesis = "There is no relationship between mailer type and signup rate. They are independent."
alternate_hypothesis = "There is a relationship between mailer type and signup rate. They are not independent."
acceptance_criteria = 0.05

```

<br>

#### Calculate Observed Frequencies & Expected Frequencies

As we discussed previously, the *observed frequencies* in a Chi-Square Test for Independence are the true values that we have seen. In other words, these are the actual rates per group in the data itself. The *expected frequencies* are what we would *expect* to see based on *all* of the data combined.

The below code:

* Summarises our dataset to a 2x2 matrix for *signup_flag* by *mailer_type*
* Based on this, calculates the:
    * Chi-Square Statistic
    * p-value
    * Degrees of Freedom
    * Expected Values
* Prints out the Chi-Square Statistic & p-value from the test
* Calculates the Critical Value based upon our Acceptance Criteria & the Degrees Of Freedom
* Prints out the Critical Value

```python

# aggregate our data to get observed values
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values

# run the chi-square test
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False)

# print chi-square statistic
print(chi2_statistic)
>> 1.94

# print p-value
print(p_value)
>> 0.16

# find the critical value for our test
critical_value = chi2.ppf(1 - acceptance_criteria, dof)

# print critical value
print(critical_value)
>> 3.84

```
<br>
Taking the observed values, we can calculate the signup rate of each group:

* Mailer 1 (low cost, low quality): **32.8%** signup rate
* Mailer 2 (high cost, high quality): **37.8%** signup rate

These observed values suggest that the higher cost/quality of Mailer 2 does positively impact the signup rate for the Delivery Club. However, only once we have performed the Chi-Square Test for Independence will we be able to better understand how confidently we can say that this conclusion is robust, or whether it may have occurred by chance.

After applying the test, we have a Chi-Square Statistic of **1.94** and a p-value of **0.16**.  The critical value for our specified Acceptance Criteria of 0.05 can be calculated as **3.84**.

**Note:** When applying the Chi-Square Test above, we use the parameter *correction = False* which means we are applying what is known as the *Yate's Correction*, which is applied when your Degrees of Freedom is equal to one. This correction helps to prevent overestimation of statistical signficance in this case.

___

<br>

# Analysing The Results <a name="chi-square-results"></a>

We now have everything we need to interpret the results of our Chi-Square Test and make the appropriate conclusions.

From our results, we can see that, since our resulting p-value of **0.16** is *greater* than our Acceptance Criteria of 0.05 then we will *retain* the Null Hypothesis and conclude that there is not sufficient evidence that the observed difference occurs due to anything other than chance. Thus we determine there is no significant difference between the signup rates of Mailer 1 and Mailer 2.

We could also make the same conclusion based upon our resulting Chi-Square statistic of **1.94** being *lower* than our Critical Value of **3.84**.

In order to make our output more dynamic, we have created code to automatically interpret the results and explain the outcome to us:

```python

# print the results (based upon p-value)
if p_value <= acceptance_criteria:
    print(f"As our p-value of {p_value} is lower than our acceptance_criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that there is sufficient evidence to support the claim that: {alternate_hypothesis}")
else:
    print(f"As our p-value of {p_value} is higher than our acceptance_criteria of {acceptance_criteria} - we retain the null hypothesis, and conclude that there is not sufficient evidence that the observed difference occurs due to anything other than chance, thus we determine: {null_hypothesis}")

>> As our p-value of 0.16351 is higher than our acceptance_criteria of 0.05 - we retain the null hypothesis, and conclude that there is not sufficient evidence that the observed difference occurs due to anything other than chance, thus we determine: There is no relationship between mailer type and signup rate. They are independent.


# print the results (based upon p-value)
if chi2_statistic >= critical_value:
    print(f"As our chi-square statistic of {chi2_statistic} is higher than our critical value of {critical_value} - we reject the null hypothesis, and conclude that there is sufficient evidence to support the claim that: {alternate_hypothesis}")
else:
    print(f"As our chi-square statistic of {chi2_statistic} is lower than our critical value of {critical_value} - we retain the null hypothesis, and conclude that there is not sufficient evidence that the observed difference occurs due to anything other than chance, thus we determine: {null_hypothesis}")
    
>> As our chi-square statistic of 1.9414 is lower than our critical value of 3.841458820694124 - we retain the null hypothesis, and conclude that there is not sufficient evidence that the observed difference occurs due to anything other than chance, thus we determine: There is no relationship between mailer type and signup rate. They are independent.

```
<br>

We can see from above that, conversely to our initial observation, there is not enough evidence to support the conclusion that a significant difference exists in the signup rates for Mailer 1 and Mailer 2. Therefore, we retain the null hypothesis.

___

<br>

# Discussion <a name="discussion"></a>

What we are effectively saying here is that, even though we can see that the higher cost/quality Mailer 2 had a higher signup rate (37.8%) compared to the lower cost/quality Mailer 1 (32.8%), based on our predefined Acceptance Criteria of 0.05, this difference is not significant enough to suggest that the type of mailer received had an impact on the decision to sign up.

From our initial observations, the client may have ended up spending more money on higher quality mailers for future promotional campaigns, expecting this to yield a higher signup rate. While the results of our Hypothesis Test do not say there *definitely is not a difference between the two mailers*, it allows us to advise that the evidence does not support any rigid conclusions *at this point in time*. In fact, it shows that the extra investment in a higher quality mailer may not *necessarily* lead to any extra revenue at all. 

Running more A/B Tests like this, gathering more data, and then re-running this test may provide us, and the client more insight!

