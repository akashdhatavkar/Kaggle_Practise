##----- Importing Required Packages -----##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from scipy.stats import zscore
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost

##----- Importing data -----##
data = pd.read_csv("D:\\Kaggle Datasets\\Kaggle_Practise\\Zomato Data\\zomato.csv", encoding='latin-1')
countryCode = pd.read_excel("D:\\Kaggle Datasets\\Kaggle_Practise\\Zomato Data\\Country-Code.xlsx")

##----- Merging the 2 data Frames -----##
## Checking if any data has more codes
np.array(set(data["Country Code"].unique()).symmetric_difference(countryCode['Country Code'].unique()))
## Both datasets have the same codes

data = pd.merge(data, countryCode, how = "inner")

##----- Set option to see all 22 columns -----##
pd.set_option("display.max_columns", 22)
data.head()

##----- Renaming columns to remove spacing between headers -----##
data.columns=['Restaurant_ID', 'Restaurant_Name', 'Country_Code', 'City', 'Address',
       'Locality', 'Locality_Verbose', 'Longitude', 'Latitude', 'Cuisines',
       'Average_Cost_for_two', 'Currency', 'Has_Table_booking',
       'Has_Online_delivery', 'Is_delivering_now', 'Switch_to_order_menu',
       'Price_range', 'Aggregate_rating', 'Rating_color', 'Rating_text',
       'Votes', 'Country']

##----- EDA -----##
data.isnull().sum()
## 9 null values in Cuisines
data.loc[data.Cuisines.isnull() == True] 
## All missing values are for country USA

## Checking Food Prices by Country
prcCnty = data.groupby(["Country"])["Price_range"].mean().sort_values(ascending = False)
sns.set_context("poster")
figure(figsize = (20,15))
sns.barplot(x = prcCnty, y = prcCnty.index, palette = "magma")
plt.xlabel("Avg_Price_Range")
plt.ylabel("Country")
plt.title("Price Range by Country")    
## Singapore has higher priced restraunts while India has the lowest

## Country with most votes
cntyVot = data.groupby("Country")["Votes"].mean().sort_values(ascending = False)
sns.set_context("poster")
figure(figsize = (20,15))
sns.barplot(x = cntyVot, y = cntyVot.index, palette = "icefire_r")
plt.xlabel("Average Votes")
plt.ylabel("Country")
plt.title("Country by Average Votes")
## Indonesia has the highest average votes while Brazil has the least

## Checking Rating_color & Rating_text info
data.Rating_color.value_counts()
data.Rating_text.value_counts()

## Dropping redundant columns
data.drop(['Country_Code','Restaurant_ID', 'Restaurant_Name','Address','Locality',
           'Locality_Verbose','Longitude', 'Latitude', 'Switch_to_order_menu','Rating_color'],
          axis=1,inplace=True)

data.columns

##----- Creating X and Y datasets for modelling -----##
x = data.drop("Aggregate_rating", axis = 1)
y = data.loc[:,"Aggregate_rating"]

##----- Splitting Data in train and test -----##
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

##----- Checking for most popular cuisines -----##
data.Cuisines.value_counts()
## Most cuisines are repeated and some restaurants offer more than 1 cuisine

##----- Creating a column which counts the number of cuisines offered by the restaurant -----##
X_train["no_of_cuisines"] = X_train.Cuisines.str.count(",")+1
X_test["no_of_cuisines"] = X_test.Cuisines.str.count(",")+1

X_train.no_of_cuisines.value_counts()
X_test.no_of_cuisines.value_counts()

## Checking mean for USA so that we can impute the missing cuisine values in the count of cuisines
X_train.groupby("Country")["no_of_cuisines"].mean()
## 2 cuisines seems to be the mean. so replacing the na with 2 

##----- Imputing missing values with mean -----##
X_train.no_of_cuisines.fillna(2, inplace = True)
X_test.no_of_cuisines.fillna(2, inplace = True)

##----- Creating a Continent Column -----##
def continent(x):
    if x in ["United States", "Canada", "Brazil"]:
        return("Americas")
    elif x in ["India", "Phillipines", "Sri Lanka", "UAE", "Indonesia", "Qatar", "Singapore"]:
        return("Asia")
    elif x in ["Australia", "New Zealand"]:
        return("Australia_continent")
    elif x in ["Turkey", "United Kingdom"]:
        return("Europe")
    else:
        return("Africa")

X_train["Continent"] = X_train.Country.apply(continent)
X_test["Continent"] = X_test.Country.apply(continent)

##----- Each countrys rates are given in local currency, converting it to a standard rate -----##
conversion_rates = {'Botswana Pula(P)':0.095, 'Brazilian Real(R$)':0.266,'Dollar($)':1,'Emirati Diram(AED)':0.272,
    'Indian Rupees(Rs.)':0.014,'Indonesian Rupiah(IDR)':0.00007,'NewZealand($)':0.688,'Pounds(£)':1.314,
    'Qatari Rial(QR)':0.274,'Rand(R)':0.072,'Sri Lankan Rupee(LKR)':0.0055,'Turkish Lira(TL)':0.188}

X_train["New_cost"] = X_train.Average_Cost_for_two * X_train.Currency.map(conversion_rates)        
X_test["New_cost"] = X_test.Average_Cost_for_two * X_test.Currency.map(conversion_rates)        

## Comparing avg_cost_for_two by Country
cntyFor2 = X_train.groupby("Country")["New_cost"].mean().sort_values(ascending = False)
sns.set_context("poster")
figure(figsize = (20,15))
sns.barplot(x = cntyFor2, y = cntyFor2.index, palette = "rocket")
plt.xlabel("Average_cost_for_2 for Country")
plt.ylabel("Country")

##----- Converting Ratings_text to a score value -----##
## Since most of the values are "Average" setting Not rated to Average score(Mode)
ratings_score = {"Excellent": 5, "Very Good": 4, "Good": 3, "Average": 2, "Not rated": 2, "Poor": 1}
X_train.Rating_text = X_train.Rating_text.map(ratings_score)
X_test.Rating_text = X_test.Rating_text.map(ratings_score)

##----- Binary encoding for variables that have yes and no values -----##
X_train.Has_Table_booking.value_counts()
X_train.Has_Online_delivery.value_counts()
X_train.Is_delivering_now.value_counts()

binary = {"Yes": 1, "No": 0}
X_train.Has_Table_booking = X_train.Has_Table_booking.map(binary)
X_train.Has_Online_delivery = X_train.Has_Online_delivery.map(binary)
X_train.Is_delivering_now = X_train.Is_delivering_now.map(binary)

X_test.Has_Table_booking = X_test.Has_Table_booking.map(binary)
X_test.Has_Online_delivery = X_test.Has_Online_delivery.map(binary)
X_test.Is_delivering_now = X_test.Is_delivering_now.map(binary)

##----- Droppping columns not needed for modelling -----##
X_train.drop(['City', 'Cuisines', 'Average_Cost_for_two', "Currency"], axis = 1, inplace = True)
X_test.drop(['City', 'Cuisines', 'Average_Cost_for_two', "Currency"], axis = 1, inplace = True)

##----- Creating dummies for country and Continent -----##
X_train.Continent = pd.Categorical(X_train.Continent)
conti = pd.get_dummies(X_train.Continent)
X_train = pd.concat([X_train, conti], axis = 1)
X_train = X_train.drop("Continent", axis = 1)

X_train.Country = pd.Categorical(X_train.Country)
country = pd.get_dummies(X_train.Country)
X_train = pd.concat([X_train, country], axis = 1)
X_train = X_train.drop("Country", axis = 1)

X_test.Continent = pd.Categorical(X_test.Continent)
conti = pd.get_dummies(X_test.Continent)
X_test = pd.concat([X_test, conti], axis = 1)
X_test = X_test.drop("Continent", axis = 1)

X_test.Country = pd.Categorical(X_test.Country)
country = pd.get_dummies(X_test.Country)
X_test = pd.concat([X_test, country], axis = 1)
X_test = X_test.drop("Country", axis = 1)

##----- Renameing columns -----##
X_train.columns=['Table_booking', 'Online_delivery', 'Delivering_now', 'Price_range',
       'Rating_text', 'Votes', 'no_of_cuisines', 'New_cost', 'Africa',
       'Americas', 'Asia', 'Australia_continent', 'Europe', 'Australia',
       'Brazil', 'Canada', 'India', 'Indonesia', 'NewZealand', 'Phillipines',
       'Qatar', 'Singapore', 'SouthAfrica', 'SriLanka', 'Turkey', 'UAE',
       'UnitedKingdom', 'UnitedStates']

X_test.columns=['Table_booking', 'Online_delivery', 'Delivering_now', 'Price_range',
       'Rating_text', 'Votes', 'no_of_cuisines', 'New_cost', 'Africa',
       'Americas', 'Asia', 'Australia_continent', 'Europe', 'Australia',
       'Brazil', 'Canada', 'India', 'Indonesia', 'NewZealand', 'Phillipines',
       'Qatar', 'Singapore', 'SouthAfrica', 'SriLanka', 'Turkey', 'UAE',
       'UnitedKingdom', 'UnitedStates']

##----- Scaling features -----##
train_scale = pd.DataFrame(zscore(X_train, axis = 1))
test_scale = pd.DataFrame(zscore(X_test, axis = 1))

## Scaling removes col names. So naming columns
train_scale.columns=['Table_booking', 'Online_delivery', 'Delivering_now', 'Price_range',
       'Rating_text', 'Votes', 'no_of_cuisines', 'New_cost', 'Africa',
       'Americas', 'Asia', 'Australia_continent', 'Europe', 'Australia',
       'Brazil', 'Canada', 'India', 'Indonesia', 'NewZealand', 'Phillipines',
       'Qatar', 'Singapore', 'SouthAfrica', 'SriLanka', 'Turkey', 'UAE',
       'UnitedKingdom', 'UnitedStates']

test_scale.columns=['Table_booking', 'Online_delivery', 'Delivering_now', 'Price_range',
       'Rating_text', 'Votes', 'no_of_cuisines', 'New_cost', 'Africa',
       'Americas', 'Asia', 'Australia_continent', 'Europe', 'Australia',
       'Brazil', 'Canada', 'India', 'Indonesia', 'NewZealand', 'Phillipines',
       'Qatar', 'Singapore', 'SouthAfrica', 'SriLanka', 'Turkey', 'UAE',
       'UnitedKingdom', 'UnitedStates']

## Should not have scaled all variables. Binary data also got scaled (Shouldn't have done that)

##----- Rounding values to 2 decimal points -----##
train_scale = np.round(train_scale, decimals = 4)
test_scale = np.round(test_scale, decimals  = 4)

##----- Fitiing Linear Regression -----##
lm = LinearRegression()
lm.fit(train_scale, y_train)
lm_pred = lm.predict(test_scale)

print("R-square: ", r2_score(y_test, np.round(lm_pred, decimals = 1)))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, np.round(lm_pred, decimals = 1))))

##----- Fitting Decision Tree -----##
tree = DecisionTreeRegressor()
tree.fit(train_scale, y_train)
tree_pred = tree.predict(test_scale)
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, np.round(tree_pred, decimals = 1))))

##----- Fitting Random Forest -----##
rf = RandomForestRegressor()
rf.fit(train_scale, y_train)
rf_pred = rf.predict(test_scale)
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, np.round(rf_pred, decimals = 1))))

##----- fitting xgboost -----##
xgb = xgboost.XGBRegressor(n_estimators = 100, learning_rate = 0.3, max_depth = 4)
xgb.fit(train_scale, y_train)
xgb_pred = xgb.predict(test_scale)
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, np.round(xgb_pred, decimals = 1))))
print('R square value using XGBoost',r2_score(y_test,xgb_pred))
print('Variance covered by XG Boost Regression : ',explained_variance_score(xgb_pred,y_test))
