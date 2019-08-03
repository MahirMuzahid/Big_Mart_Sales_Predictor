#Import Plugins

from sklearn import svm
import csv
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
import seaborn as sns
import pandas as pd
import numpy as np

#taking data
test_data = pd.read_csv('Test.csv')
train_data = pd.read_csv('Train.csv')

#fill miss spell word
train_data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
test_data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)

#calculating number of years
train_data['num_years'] = train_data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x)
test_data['num_years'] = test_data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x)

#taking full data in a list
full_data = [train_data, test_data]

#filling blank data
for data in full_data:
    data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace = True)
    data['Outlet_Size'].fillna('Medium',inplace = True)

#Handeling values
col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
train_datar = pd.get_dummies(train_data, columns = col, drop_first = True)
test_datar = pd.get_dummies(test_data, columns = col,drop_first = True)
feat_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'num_years',
       'Item_Fat_Content_Regular', 'Item_Type_Breads', 'Item_Type_Breakfast',
       'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
       'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
       'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
       'Outlet_Size_Medium', 'Outlet_Size_Small',
       'Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3',
       'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3']

#initalizing train adn test daa
X = train_datar[feat_cols]
y = train_datar['Item_Outlet_Sales']

#split train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)

#train data with XGBoost
from xgboost.sklearn import XGBRegressor
XGB = XGBRegressor()
XGB.fit(X_train, y_train)
y_pred = XGB.predict(X_test)

#Calculating RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))

#Predicting value
X_t = test_datar[feat_cols]
y_result = XGB.predict(X_t)

#Saving output value in csv
r = pd.DataFrame()
r['Item_Identifier'] = test_datar['Item_Identifier']
r['Outlet_Identifier'] = test_datar['Outlet_Identifier']
r["Item_Outlet_Sales"] = y_result
result = r.sort_index()
result.to_csv('Bigmart_Sales.csv',index = False)