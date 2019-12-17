##----- Importing required packages -----##
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.cluster.vq import kmeans, vq

##----- Reading the data -----##
data = pd.read_csv("D:\\Kaggle Datasets\\Kaggle_Practise\\Goodreads Data\\books.csv", error_bad_lines = False)
data.index = data.bookID
data.info()

##----- Replacing J.K Rowling - Mary Grand Pre to just JK Rownling -----##
data.replace(to_replace = "J.K. Rowling-Mary GrandPré" , value = "J.K. Rowling", inplace = True)

##----- EDA -----##
## Looking at the 20 more occuring books
sns.set_context("poster")
figure(figsize = (20,15))
books = data.title.value_counts()[:20]
sns.barplot(x = books, y = books.index, palette = "deep")
plt.title("Most Occuring Books")
plt.xlabel("Number of books")
plt.ylabel("Book Title")
plt.show()

## Books by language
sns.set_context("paper")
figure(figsize = (20,15))
lang = data.groupby("language_code")["title"].count()
sns.barplot(x = lang, y = lang.index, palette = "deep")

## Top 10 best rated books
topRated = data.sort_values("ratings_count", ascending = False).head(10).set_index("title")
figure(figsize = (20,15))
sns.barplot(x = topRated.ratings_count, y = topRated.index, palette = "rocket")

## Authors with most books
sns.set_context("talk")
authorsBooks = data.groupby("authors")["title"].count().reset_index().sort_values("title", ascending = False).head(10)
figure(figsize = (20,15))
sns.barplot(x = authorsBooks.title, y = authorsBooks.authors, palette = "icefire_r")
plt.title("Top 10 authors with Most Books")
plt.xlabel("Number of Books")
plt.ylabel("Author")

#----- Checking the rating distribution of the books -----##
figure(figsize = (10,10))
sns.distplot(data.average_rating.astype(float), bins = 20)

##----- Checking for relation between ratings and reviews count -----##
figure(figsize = (10,10))
sns.set_context("poster")
sns.jointplot(x = data.text_reviews_count, y = data.average_rating, kind = "scatter", )

## Books with higest reviews
mostRev = data.sort_values("text_reviews_count", ascending = False).head(10).set_index("title")
figure(figsize = (20,15))
sns.set_context("poster")
sns.barplot(x = mostRev.text_reviews_count, y = mostRev.index, palette = "magma")
plt.title("Top 10 most Reviewed Books")

##----- Clustering based on average ratings and ratings count -----##
trial = data.loc[:,["average_rating", "ratings_count"]]
trial1 = data.loc[:,["average_rating", "ratings_count", "authors"]]
le = LabelEncoder()
trial1.authors = le.fit_transform(trial1.authors.values)
df = np.asarray([np.asarray(trial['average_rating']), np.asarray(trial['ratings_count']), 
                 np.asarray(trial1['authors'])]).T

## Using elbow plot to find the best K
distortions = []
for k in range(2,31):
    k_means = KMeans(n_clusters = k)
    k_means.fit(trial1)
    distortions.append(k_means.inertia_)

figure(figsize = (20,15))
plt.plot(range(2,31), distortions, "bx-")
plt.title("Elbow Curve")
## Best k is 5

## Taking k = 5
centroids, _ = kmeans(trial1, 5)

#assigning each sample to a cluster
#Vector Quantisation:
idx, _ = vq(trial1, centroids)

##----- Plotting using numpy's logical Indexing ----##
sns.set_context("poster")
figure(figsize = (20, 15))
ax = plt.axes(projection='3d')
ax.scatter3D(df[idx==0,0],df[idx==0,1],'or',#red circles
     df[idx==1,0],df[idx==1,1],'ob',#blue circles
     df[idx==2,0],df[idx==2,1],'oy', #yellow circles
     df[idx==3,0],df[idx==3,1],'om', #magenta circles
     df[idx==4,0],df[idx==4,1],'ok',#black circles
        )

plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8, )

## Viz 3D is difficult, so only doing ti for the 2 parameters

kmeans = KMeans(n_clusters = 5, n_init = 15, n_jobs = -1)
kmeans.fit(df)
data["cluster"] = kmeans.labels_
