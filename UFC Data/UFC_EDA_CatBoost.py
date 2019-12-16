##----- Importing Required Packages -----##
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, make_scorer, recall_score, f1_score, confusion_matrix

##----- Reading in Data -----##
ufc = pd.read_csv("D:\\Kaggle Datasets\\Kaggle_Practise\\UFC Data\\data.csv")

##----- EDA -----##
len(ufc) ## 5144 rows
ufc.isnull().sum()/len(ufc) ## Some columns have 25% missing values

list(ufc.select_dtypes(include = ["object"])) ## Check for categorical columns

##----- Plotting data -----##
ufc.Winner.unique()
pieData = ufc.groupby("Winner")["Winner"].count()
figure(num = None, figsize = (8,6), dpi = 100)
with plt.style.context(("ggplot")):
    plt.pie(pieData, labels = pieData.index, colors = ["blue", "green", "maroon"], autopct='%1.1f%%')
    #plt.legend()
    plt.show()

## Red has most wins
    
##----- Checking Which country has the most fights -----##
ufc["Country"] = ufc.location.apply(lambda x : x.split(",")[-1])
figure(num = None, figsize = (8,6), dpi = 100)
countryval = ufc.groupby("Country")["Country"].count()
countryval = countryval.sort_values(ascending = False)
figure(num = None, figsize = (8,6), dpi = 100)
with plt.style.context(("ggplot")):
    plt.bar(x = countryval.index[0:10], height = countryval[0:10], color = "maroon")
    plt.xlabel("Country")
    plt.ylabel("Number of Events")
    plt.show()

##----- Checking Fighters by participation -----##
fighters = pd.concat([ufc.R_fighter, ufc.B_fighter], ignore_index = True)
fighters.index = fighters
fighters = fighters.groupby(fighters.index).count() ## 1915 unique fighters
fighters = fighters.sort_values(ascending = False)
figure(num = None, figsize = (8,6), dpi = 100)
with plt.style.context(("ggplot")):
    plt.bar(x = fighters.index[0:10], height = fighters[0:10], color = "maroon")
    plt.xlabel("Fighter's Name")
    plt.ylabel("Number of Fights")
    plt.show()

##----- Most fights by year -----##
ufc["Year"] = ufc.date.apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d").year)
yearlyfights = ufc.groupby("Year")["Year"].count()
yearlyfights = yearlyfights.sort_index(ascending = True)
figure(num = None, figsize = (8,6), dpi = 100)
with plt.style.context(("ggplot")):
    plt.bar(x = yearlyfights.index, height = yearlyfights, color = "maroon")
    plt.xlabel("Year")
    plt.ylabel("Number of Fights")
    plt.show()

##----- Plotting age distribution of fighters -----##
rage = ufc.loc[:,["R_fighter", "R_age"]]
rage = rage.rename(columns = {"R_fighter" : "fighter", "R_age" : "age"})
bage = ufc.loc[:,["B_fighter", "B_age"]]
bage = bage.rename(columns = {"B_fighter" : "fighter", "B_age" : "age"})

ageDist = pd.concat([rage,bage], ignore_index = True)
ageDist.dropna(axis = 0, how = "any", inplace = True)
ageDist.drop_duplicates(subset = "fighter", keep = "first", inplace = True)

sns.distplot(ageDist.age, bins = 10)
plt.title('Age Distribution by Fighters')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

ageDist.max()
ageDist.min()

##----- Feature Engineering -----##
## Dropping old UFC Fights, because in 2000 UFC was sold to Station Casino executives 
## Lorenzo and Frank Fertitta and business associate Dana White
ufc = ufc.loc[ufc.Year > 1999]

##----- Dropping location and date column since we have country and year -----##
ufc = ufc.loc[:,~ufc.columns.isin(["date", "location"])]
ufc = ufc.loc[:,~ufc.columns.isin(["Referee"])]

X = ufc.loc[:,~ufc.columns.isin(["Winner"])]
Y = ufc.loc[:, "Winner"]

##----- Splitting Data into Train and Test -----##
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

##----- Normalizing the data -----##
num_cols = X_train.columns[X_train.dtypes.apply(lambda x : np.issubdtype(x, np.number))]

X_train[num_cols] = StandardScaler().fit_transform(X_train[num_cols])
X_test[num_cols] = StandardScaler().fit_transform(X_test[num_cols])

##----- Converting all NA values to -999 -----##
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

##----- Building CatBoost on all features -----##
cat_features = np.where(X_train.dtypes != float)[0]

##----- Finding the best Hyper-parameters -----##
#custom_scorer = make_scorer(precision_score, average = 'macro')
parameters = {
              'iterations': [200, 400, 600, 800],
              'depth': [4, 7, 9, 11],
              'learning_rate': [0.01, 0.01, 0.05],
              'l2_leaf_reg' : [1, 3, 5, 7, 9]
}

#model_grid = CatBoostClassifier(random_state = 42, task_type="GPU", silent=True, od_type="Iter")
#clf_grid = GridSearchCV(model_grid, parameters, cv=3, scoring=custom_scorer)
#clf_grid.fit(X_train, y_train, cat_features = cat_features)
#print('Best Parameters: ', clf_grid.best_params_)
#print('Best Precision: ', clf_grid.best_score_)

##----- Fitting the Model -----##
average = "macro"
model = CatBoostClassifier(task_type = "GPU", learning_rate = 0.01, iterations = 600, l2_leaf_reg = 5,
                           random_seed = 42, od_type = "Iter", depth = 4, silent = True)

model.fit(X_train, y_train, cat_features = cat_features)
train_pred = model.predict(X_train)
precision_train = precision_score(y_train, train_pred, average=average)
recall_train = recall_score(y_train, train_pred, average=average)
f1score_train = f1_score(y_train, train_pred, average=average)
print(f'Precision (Train): {precision_train}')
print(f'Recall (Train): {recall_train}')
print(f'F1-Score (Train): {f1score_train}')

test_pred = model.predict(X_test)
precision_test = precision_score(y_test, test_pred, average=average)
recall_test = recall_score(y_test, test_pred, average=average)
f1score_test = f1_score(y_test, test_pred, average=average)
print(f'Precision (Test): {precision_test}')
print(f'Recall (Test): {recall_test}')
print(f'F1-Score (Test): {f1score_test}')

##----- Finding important features -----##
def rf_feature_importance(m, df):
    return(pd.DataFrame({"Feature": df.columns, "Importance": m.feature_importances_}
                        ).sort_values("Importance",ascending = False))

fi = rf_feature_importance(model, X_train)

##-----  Keeping on features with importance > 1 -----##
to_keep = fi.loc[fi.Importance > 1, "Feature"]
x_train = X_train[to_keep].copy()
x_test = X_test[to_keep].copy()

cat_features = np.where(x_train.dtypes != float)[0]
model.fit(x_train, y_train, cat_features = cat_features)
train_pred1 = model.predict(x_train)
precision_train = precision_score(y_train, train_pred1, average=average)
recall_train = recall_score(y_train, train_pred1, average=average)
f1score_train = f1_score(y_train, train_pred1, average=average)
print(f'Precision (Train): {precision_train}')
print(f'Recall (Train): {recall_train}')
print(f'F1-Score (Train): {f1score_train}')

test_pred1 = model.predict(x_test)
precision_test = precision_score(y_test, test_pred1, average=average)
recall_test = recall_score(y_test, test_pred1, average=average)
f1score_test = f1_score(y_test, test_pred1, average=average)
print(f'Precision (Test): {precision_test}')
print(f'Recall (Test): {recall_test}')
print(f'F1-Score (Test): {f1score_test}')
## Precision Decreased

##----- Confusion Matrix -----##
cf = confusion_matrix(y_test, test_pred1)
ax = sns.heatmap(cf, annot = True, fmt = "d")
ax.set_ylabel("Predicted")
ax.set_xlabel("Actual")
ax.set_ylim(3,0)