import requests
import re
from bs4 import BeautifulSoup
from nltk.classify import NaiveBayesClassifier
from nltk import word_tokenize
from bahasa.stemmer import Stemmer
import pandas
import nltk
from matplotlib import pyplot as plt


# function
# ambil list kata-kata dari file
def getList(FileName):
    words = []
    fp = open(FileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        words.append(word)
        line = fp.readline()
    fp.close()
    return words


# buang stopword tapi tokenize dulu
def removeStopwords(sentence):
    sw = getList('data/feature_list/stopwordsID.txt')
    # print(sentence)
    filtered = []
    for kata in sentence:
        if kata not in sw:
            filtered.append(kata)
    # print(filtered)
    return filtered


# stem / kata dasar
def stem(text):
    stemmer = Stemmer()
    return stemmer.stem(text)


# bersihin artikel dari journalotaku
def articlefromurl(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # Ganti html IDs kalau artikel nya beda id
    articleIds = "#AdAsia"
    articleDirty = str(soup.select(articleIds))
    cleanTag = re.compile('<.+?>')
    cleanSemiColon = re.compile('.*;')
    cleanScriptBegin = re.compile('#.*{')
    cleanScriptEnd = re.compile('}')
    articleClean = re.sub(cleanTag, '', articleDirty)
    sentence = re.split("\r|\n", articleClean)
    finishArticle = []
    for s in sentence:
        sub = re.sub(' +', ' ', re.sub(cleanScriptEnd, '', re.sub(cleanScriptBegin, '', re.sub(cleanSemiColon, '', s))))
        if sub != '[' and sub != "]" and sub != '' and sub != ' ':
            finishArticle.append(sub)
    return finishArticle


def wordFeatures(words):
    filtered = removeStopwords(word_tokenize(stem(words)))
    return {word: True for word in filtered}


# training model
def buildModel():
    dataset = pandas.read_csv('data/feature_list/dataset_opini_film.csv')
    total = dataset['Text'].count()
    positive_reviews = dataset[dataset['Sentiment'] == "positive"]['Text']
    negative_reviews = dataset[dataset['Sentiment'] == "negative"]['Text']
    negative_features = [(wordFeatures(stem(f)), "negative") for f in negative_reviews]
    positive_features = [(wordFeatures(stem(f)), "positive") for f in positive_reviews]
    split = 80
    trainPercentage = (split*2 / total) * 100
    testPercentage = (total - split*2) / total * 100
    sentiment_classifier = NaiveBayesClassifier.train(positive_features[:split] + negative_features[:split])
    print("Model trained:\nTrain data: " + str(trainPercentage) + "% data \nTest data: " + str(testPercentage) + "% data\nAkurasi: " + str(nltk.classify.util.accuracy(sentiment_classifier, positive_features[split:] + negative_features[split:])*100) + "% akurat\n")
    return sentiment_classifier


def pieChart(pos, neg):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    label = ['Positif', 'Negatif']
    value = [pos, neg]
    ax.pie(value, labels=label, autopct='%1.2f%%')
    plt.show()


def nilaiArtikel(url):
    article = articlefromurl(url)
    model = buildModel()
    negPts = 0
    posPts = 0
    for sentence in article:
        # print(sentence)
        nilai = model.classify(wordFeatures(sentence))
        if nilai == "negative":
            negPts = negPts+1
        if nilai == "positive":
            posPts = posPts+1
        # print(nilai)
    return [posPts, negPts]


# Ambil html page dari url
url = "http://jurnalotaku.com/2021/01/13/review-kimetsu-no-yaiba-the-movie-mugen-train/"
# Nge-Judge Sentimen Artikel
pts = nilaiArtikel(url)
print("Artikel mengandung:\n" + str(pts[0]) + " kalimat Positif (" + str(pts[0]/(pts[0]+pts[1])*100) + "%)\n" + str(pts[1]) + " kalimat Negatif (" + str(pts[1]/(pts[0]+pts[1])*100) + "%)")
pieChart(pts[0], pts[1])