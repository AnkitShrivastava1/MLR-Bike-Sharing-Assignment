# Bike Sharing Analysis

## Overview
This project involves analyzing bike sharing data to understand the key factors influencing bike rentals. The analysis includes preprocessing data, encoding categorical variables, scaling features, building a multiple linear regression model, and validating the model's assumptions.

## Process

1. Collect Data:
- Gather the dataset with dependent and independent variables.

2. Preprocess Data:
- Identify and remove null values.
- Look for outliers and handle them.
- Handle missing data.

3. Plot Data:
- Use boxplots, pairplots, correlations, scatterplots, etc., to find any obvious trends in numerical and categorical variables.

4. Encode Categorical Variables:
- Convert categorical variables to numerical using OneHotEncoder or similar techniques.
- Use `drop_first=True` to avoid multicollinearity.

5. Split Data:
- Divide the data into training and testing sets.

6. Scale Data:
- Normalize or standardize features to remove the effect of scaling of variables.
- Apply scaling to both training and test data.

7. Run RFE:
- Perform Recursive Feature Elimination (RFE) to get an initial assessment/ranking of relevant variables.
- Remove low-ranked (>1) variables.

8. Build & Train Model on Training Data:
- Initialize the linear regression model.
- Run OLS (Ordinary Least Squares) to get model parameters (R-square, F-stat, p-values, etc.).
- Run VIF (Variance Inflation Factor) to find multicollinear variables.
- Remove variables with high VIF (>5) or high p-value (>5%).
- Repeat until all variables have low VIF, low p-values, and a high enough R-square.

9. Check Model:
- Perform checks on residuals to validate assumptions of linearity, zero mean, normality, and homoscedasticity.

10. Test Model:
- Test your model on the test data.
- Check residuals.
- Calculate performance metrics (R-squared, RMSE).

## Model Equation
The final model equation is:
cnt = 0.0421 + (0.2399 * yr) + (0.1317 * season_4) + (0.0814 * season_2) - (0.2181 * weathersit_3) + (0.6454 * atemp) - (0.1126 * windspeed)


## Key Insights
- atemp (0.6454): As the apparent temperature (`atemp`) increases, the count of bike rentals increases significantly, indicating that warmer temperatures encourage more people to rent bikes.
- yr (0.2399): The count of bike rentals increases significantly over time, possibly due to the growing popularity of bike-sharing programs or improvements in service availability.
- weathersit_3 (-0.2181): Adverse weather conditions (`weathersit_3`) significantly decrease the count of bike rentals, showing that people are less likely to rent bikes in poor weather.
- season_4 (0.1317): Bike rentals increase slightly during the fourth season, suggesting a seasonal preference for biking.
- season_2 (0.0814): Bike rentals increase slightly during the second season, indicating another seasonal trend.
- windspeed (-0.1126): Higher windspeed negatively impacts bike rentals, as strong winds make biking less comfortable.
- const (0.0421): Represents the baseline effect on bike rentals when all other variables are zero.

## Important Libraries Required
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.api as sm
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
