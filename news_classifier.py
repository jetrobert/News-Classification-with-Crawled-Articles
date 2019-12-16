# coding: utf-8

# ### import libraries

import string
import math
import random
import sys
import os
import re
import collections
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.neural_network import MLPClassifier
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()


# ### prequirements 

removephrases = [
    'media playback is unsupported on your device',
    'image copyright',
    'getty images',
    'media caption',
    'image caption',
    '(reuters)',
    '(upi)',
    'getty',
    '(cnn)',
    'advertisement',
]


# ### category class


class ArticleCrawled():
    def __init__(self, heading, text, link, signature, shingles):
        self.heading = heading
        self.text = text
        self.link = link
        self.signature = signature
        self.shingles = shingles

class Category():
    def __init__(self, name, links, token_list, articles, test, test_articles, test_categories, train):
        self.name = name
        self.links = links
        self.token_list = token_list #a token is a sentence
        self.test = test
        self.test_articles = test_articles
        self.test_categories = test_categories
        self.train = train
        self.articles = articles


# ### read data


def ReadDataSet():
    print("Reading in data from files")
    total_sentences = 0
    data_dir = os.getcwd() + '/data'
    for dirname in os.listdir(data_dir):
        for filename in os.listdir(data_dir + "/" + dirname):
            if 'articles' in filename:
                #create new category if needed
                category_exists = False
                for category in categories:
                    if category.name == filename.split('.')[0]:
                        category_exists = True
                if not category_exists:
                    new_category = Category(name = filename.split('.')[0], links = [], 
                                            token_list = [], articles = [], train = [], 
                                            test = [], test_articles = [], test_categories = [])
                    categories.append(new_category)

    #read articles into category articles
    total_articles = 0
    category_count = []
    seen = set()
    for category in categories:
        for dirname in os.listdir(data_dir):
            try:
                filename = data_dir + "/" + dirname + "/" + category.name + ".txt"
                file = open((filename),"r") 
                for line in file: 
                    #must contain letters and be longer than 5 characters long
                    if any(c.isalpha()for c in line) and len(line) > 5:
                        article = line.split(":::::")
                        heading = article[0]
                        text = article[1]

                        if line not in seen:
                            seen.add(line)

                            minArticleLength = 100
                            if len(line.split()) > minArticleLength:
                                total_articles += 1
                                newArticle = ArticleCrawled(heading = heading, 
                                                            text = heading + ' ' + text, 
                                                            link = None, signature = [], 
                                                            shingles = [])
                                category.articles.append(newArticle)
                                category_count.append(category.name)
                file.close()
            except Exception as e:
                print(e)
    print ("Total Articles read: " + str(total_articles))
    return category_count


# ### explore data


categories = []
train_articles, test_articles = [],[]
train_categories, test_categories = [],[]
category_count = ReadDataSet()

if not os.path.exists("figure"):
    os.makedirs("figure")

labels = set(category_count)
counts = [category_count.count(label) for label in labels]
print(labels, counts)
plt.figure(figsize=(8,8))
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.title("Distribution of New Categories", fontsize = 16)
plt.savefig("figure/Distribution of New Categories.png")
plt.show()


# ### clean data


#trim out dataset
def CleanData():
    print("Cleaning data")
    cachedStopWords = stopwords.words("english")
    for category in categories:
        newList = []
        i = 0
        while i < len(category.articles):
            # lower the letters
            category.articles[i].text = category.articles[i].text.lower()

            #remove header divider
            category.articles[i].text = re.sub(":::::", ' ', category.articles[i].text)   

            #remove bad phrases
            for phrase in removephrases:
                category.articles[i].text = re.sub(phrase, '', category.articles[i].text)   

            #remove easy punctation
            category.articles[i].text = re.sub(r"[,.;@#?!&$-]+\ *", " ", category.articles[i].text)   

            #remove punctuation
            category.articles[i].text = "".join(l for l in category.articles[i].text if l not in string.punctuation)   

            #remove stop words
            category.articles[i].text = ' '.join([word for word in category.articles[i].text.split() if word not in cachedStopWords])

            #finally condense whitespace
            category.articles[i].text = re.sub('\s+',' ',category.articles[i].text)

            i += 1
    print("Finished cleaning data")


# ### visualize data


CleanData()



from wordcloud import WordCloud
for category in categories:
    article_list = []
    print(category.name)
    for article in category.articles:
        article_list.append(article.text)
        text = " ".join(article_list)
       
    wordcloud = WordCloud().generate(text)
    plt.figure()
    plt.subplots(figsize=(20,12))
    wordcloud = WordCloud(
        background_color="white",
        max_words=len(text),
        max_font_size=40,
        relative_scaling=.5).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("figure/"+category.name+".png")
    plt.show()


# ### split data


def SplitTrainTest():
    #shuffle arrays
    for category in categories:
        random.shuffle(category.articles)

    #find train and test sets for each category
    for category in categories:
        c_length = len(category.articles)
        category.train = category.articles[: int(.8 *c_length)]
        category.test = category.articles[int(.8 *c_length):c_length]
        for article in category.test:
            category.test_articles.append(article.text)
            category.test_categories.append([category.name])
            test_articles.append(article.text)
            test_categories.append([category.name])

    #extract train articles into array
    for category in categories:
        for article in category.train:
            train_articles.append(article.text)
            train_categories.append([category.name])



SplitTrainTest()


# ### Word Level TF-IDF Vectors as features


x_train = np.array(train_articles)
x_test = np.array(test_articles)
count_vect = CountVectorizer()
# choose the first 10,000 most frequent words
count_vect.set_params(max_features=10000)

tfidf_transformer = TfidfTransformer()
x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = tfidf_transformer.transform(x_test_counts)

label_list = ["science and health_articles", "Politics_articles", 
                   "entertainment and arts_articles", "sports_articles",
                   "tech_articles", "business_articles"]
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(label_list)
 #one-hot encode target column
y_train = np.array(label_encoder.transform(train_categories))
y_train_one = to_categorical(y_train)
y_test = np.array(label_encoder.transform(test_categories))
y_test_one = to_categorical(y_test)
print("Dimension of training set: ", x_train_tfidf.shape)
print("Dimension of test set: ", x_test_tfidf.shape)


# ### Logistic Regression


def LR():
    print("Results for Logistic Regression")
    
    classifier = LogisticRegression().fit(x_train_tfidf, y_train.ravel())
    y_pred = classifier.predict(x_test_tfidf)
    
    #transform numerical labels back to non-numerical labels 
    y_test_ori = list(label_encoder.inverse_transform(y_test))
    y_pred_ori = list(label_encoder.inverse_transform(y_pred))
    
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test_ori, y_pred_ori))
    
    #plot
    labels = ["Science & Health", "Politics", "Entertain & Arts", 
              "Sports", "Technology", "Business"]
    
    cm = confusion_matrix(y_test, y_pred) 

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index = labels, 
                         columns = labels)

    plt.figure(figsize = (12,9))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Logistic Regression \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)), fontsize=16)
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)
    plt.savefig("figure/logistic_regression.png")
    plt.show()


LR()


# ### Naive Bayes


def NaiveBayes():
    print("Results for Naive Bayes")
    
    classifier = MultinomialNB().fit(x_train_tfidf, y_train.ravel())
    y_pred = classifier.predict(x_test_tfidf)
    
    #transform numerical labels back to non-numerical labels 
    y_test_ori = list(label_encoder.inverse_transform(y_test))
    y_pred_ori = list(label_encoder.inverse_transform(y_pred))
    
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test_ori, y_pred_ori))
    
    #plot
    labels = ["Science & Health", "Politics", "Entertain & Arts", 
              "Sports", "Technology", "Business"]
    
    cm = confusion_matrix(y_test, y_pred) 

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index = labels, 
                         columns = labels)

    plt.figure(figsize = (12,9))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Naive Bayes \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)), fontsize=16)
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)
    plt.savefig("figure/naive_bayes.png")
    plt.show()



NaiveBayes()


# ### SVM


def SVM():
    print("Results for Support Vector Machine")
    
    classifier = svm.SVC(gamma='scale').fit(x_train_tfidf, y_train.ravel())
    y_pred = classifier.predict(x_test_tfidf)
    #transform numerical labels back to non-numerical labels 
    y_test_ori = list(label_encoder.inverse_transform(y_test))
    y_pred_ori = list(label_encoder.inverse_transform(y_pred))
    
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test_ori, y_pred_ori))
    
    #plot
    labels = ["Science & Health", "Politics", "Entertain & Arts", 
              "Sports", "Technology", "Business"]
    
    cm = confusion_matrix(y_test, y_pred) 

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index = labels, 
                         columns = labels)

    plt.figure(figsize = (12,9))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('SVM \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)), fontsize=16)
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)
    plt.savefig("figure/svm.png")
    plt.show()


SVM()


# ### Random Forest


def RandomForest():
    print("Results for Random Forest")

    classifier = RandomForestClassifier(n_estimators=40).fit(x_train_tfidf, y_train.ravel())
    y_pred = classifier.predict(x_test_tfidf)
    #transform numerical labels back to non-numerical labels 
    y_test_ori = list(label_encoder.inverse_transform(y_test))
    y_pred_ori = list(label_encoder.inverse_transform(y_pred))
    
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test_ori, y_pred_ori))
    
    #plot
    labels = ["Science & Health", "Politics", "Entertain & Arts", 
              "Sports", "Technology", "Business"]
    
    cm = confusion_matrix(y_test, y_pred) 

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index = labels, 
                         columns = labels)

    plt.figure(figsize = (12,9))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Random Forest \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)), fontsize=16)
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)
    plt.savefig("figure/random_forest.png")
    plt.show()


RandomForest()


# ### XGBoost


def Xgboost():
    print("Results for XGBoost")
    
    classifier = xgboost.XGBClassifier().fit(x_train_tfidf, y_train.ravel())
    y_pred = classifier.predict(x_test_tfidf)
    #transform numerical labels back to non-numerical labels 
    y_test_ori = list(label_encoder.inverse_transform(y_test))
    y_pred_ori = list(label_encoder.inverse_transform(y_pred))
    
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test_ori, y_pred_ori))
    
    #plot
    labels = ["Science & Health", "Politics", "Entertain & Arts", 
              "Sports", "Technology", "Business"]
    
    cm = confusion_matrix(y_test, y_pred) 

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index = labels, 
                         columns = labels)

    plt.figure(figsize = (12,9))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('XGBoost \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)), fontsize=16)
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)
    plt.savefig("figure/xgboost.png")
    plt.show()


Xgboost()


# ### Shallow Neural Network

def NN():
    print("Results for Shallow Neural Network")
    
    
    classifier = MLPClassifier(solver='adam', alpha=1e-5, 
                              hidden_layer_sizes=(100, 20), 
                              random_state=1, max_iter=400).fit(x_train_tfidf, y_train.ravel())
    y_pred = classifier.predict(x_test_tfidf)
    #transform numerical labels back to non-numerical labels 
    y_test_ori = list(label_encoder.inverse_transform(y_test))
    y_pred_ori = list(label_encoder.inverse_transform(y_pred))
    
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test_ori, y_pred_ori))
    
    #plot
    labels = ["Science & Health", "Politics", "Entertain & Arts", 
              "Sports", "Technology", "Business"]
    
    cm = confusion_matrix(y_test, y_pred) 

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                         index = labels, 
                         columns = labels)

    plt.figure(figsize = (12,9))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Shallow Neural Network \nAccuracy:{0:.4f}'.format(accuracy_score(y_test, y_pred)), fontsize=16)
    plt.ylabel('True label', fontsize = 12)
    plt.xlabel('Predicted label', fontsize = 12)
    plt.savefig("figure/shallow_nn.png")
    plt.show()


NN()


# ### LSTM


def LSTM_model():
    print("Results for LSTM")
    
    top_words = 100
    embedding_vecor_length = 8
    max_review_length = x_train_tfidf.shape.[1]
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(16))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train_tfidf, y_train.ravel(), validation_data=(x_test_tfidf, y_test), 
              epochs=1, batch_size=64)   



#LSTM_model()

