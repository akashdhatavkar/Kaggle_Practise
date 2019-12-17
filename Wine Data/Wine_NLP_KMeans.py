##----- Importing Required Packages -----##
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans

##----- Reading Data -----##
wine = pd.read_csv("D:\\Kaggle Datasets\\Kaggle_Practise\\Wine Data\\winemag-data_first150k.csv")
print(wine.head())

##----- EDA -----##
wine.info()
wine.description.nunique()
wine.variety.nunique()

##----- Dropping duplicate descriptions -----##
wine.drop_duplicates(subset = "description", inplace =True)

##----- Most common variety of Wine -----##
wineVar = wine.groupby("variety").filter(lambda x: len(x) > 1500)
wineplt = wineVar.groupby("variety")["variety"].count().sort_values(ascending = False)
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
with plt.style.context(('ggplot')):
    plt.bar(wineplt.index, wineplt.values, color = "maroon")
    plt.xlabel("Wine Variety")
    plt.ylabel("No of Wines")
    plt.show()
## 15 most common types
    
wineVar.index[:15]

##----- NLP -----##
## Creating a list of Stop-Words to remove from our text to reduce vector space
punc = [",", ".", '"', "'", "?", "!", ":", ";", "(", ")", "{", "}", "[", "]", "%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = wineVar.description.values
vectorize = TfidfVectorizer(stop_words = stop_words)
X = vectorize.fit_transform(desc)

##----- Checking the vectorised text ----##
word_features = vectorize.get_feature_names()
word_features[600:650]

##----- Stemming words to their roots -----##
stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer(r"[a-zA-Z\"]+")

def tokenize(text):
    return(stemmer.stem(word) for word in tokenizer.tokenize(text.lower()))

vectorize2 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize)
X2 = vectorize2.fit_transform(desc)
words_features2 = vectorize2.get_feature_names()
words_features2[:50]

##---- Taking only the top 100 token -----##
vectorize3 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, max_features = 100)
X3 = vectorize3.fit_transform(desc)
words = vectorize3.get_feature_names()

##----- Fitting kMeans Clustering to the vectorisez text -----##
kmeans = KMeans(n_clusters = 15, n_init = 5, n_jobs = -1)
kmeans.fit(X3)

common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ":" + ",".join(words[word] for word in centroid))
    
wineVar["Cluster"] = kmeans.labels_
