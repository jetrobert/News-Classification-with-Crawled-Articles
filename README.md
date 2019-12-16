### News Classification with Crawled Articles

#### Intro

* Text classification is one of the widely used Natural Language Processing (NLP) tasks which allow algorithms to learn and understand human languages. It is a process of assigning labels or categories to the text according to their contents. Text classifiers can be used to organize, structure and categorize various corpora by feeding the text as inputs and analysing the contents.

#### Data collection

* The first step of news classification is to collect data. For this project, we have crawled total 2336 news articles with 6 categories (Business, Arts and Entertainment, Politics, Science and Health, Sports, Technology) from various news websites such as CNN, The Economist and New York Post. Among the whole news data, 21.% of them are sports articles, 17.6% are Politics news, 11.8% are entertainment and arts articles, 18.4% are science and health articles, 18.6% are technology articles and 12.0% are business articles.

#### Data Pre-processing

* After the data collection, data pre-processing is the next step required so that the dataset could be free from interference of abundant information and meaningless words and symbols. For every single article, we remove punctuation, condense various spaces to one space. Then we set all words to lowercase and remove English stop words. The cleaning process can reduce the size of the raw dataset and improve performances of machine learning models. Furthermore, our models will takes less time to be trained.

#### Feature Extraction

* To choose relevant and high effective features in this project, we focus on using Term Frequency Inverse Document Frequency (TF- IDF) to extract features from original news dataset. These features are numerical vectors, which can be applied to a machine learning model. TF-IDF process consist of two parts: the first part computes the normalized Term Frequency (TF) and the second part computes Inverse Document Frequency (IDF), which is the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term existed. 

* In this project, we combined the all the news from one category as one document. Therefore we totally have 6 documents (each news category has one), then we use TF-IDF to represent the text features for training set and test set. The total number of unique words of our news articles is over 43,000. We use the first 10,000 most frequent words for our TF- IDF approach. Therefore, the dimension of our dataset is (2336, 10000).

#### Data Partition

* Once the cleaned feature vectors to represent our news articles are prepared, we split them randomly into training set and test set with a ratio of 80:20 for each news category. Thus each model trains with the same random 80% (1866 instances) of the data, the remaining 20% (470 instances) are used as the test data for model evaluation.

#### Models

* Logistic Regression
* Naïve Bayes
* Support Vector Machine (SVM)
* Random Forest
* Xtereme Gradient Boosting (XGBoost)
* Shallow Neural Network

#### Results

* The optimal accuracy is 0.891 obtained by shallow neural network, the worst accuracy result is 0.785 gained by Naïve Bayes model. The accuracy of logistic regression is 0.881, the SVM model gains 0.874 accuracy. The accuracy of Random Forest and XGBoost is 0.823 and 0.857 respectively.
