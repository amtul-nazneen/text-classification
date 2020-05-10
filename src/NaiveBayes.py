import os
import re
from collections import Counter
from math import log

# ----------------------------- Configurable Parameters -----------------------------
spamTrainDir = "train/spam"
hamTrainDir = "train/ham"
spamTestDir = "test/spam"
hamTestDir = "test/ham"
stopWordsFile = "stop_words_list.txt"
uniqueWords = []

# ----------------------------- Naive Bayes Fetching Stop Words -----------------------------
def fetchAllStopWords(stopWordsFile):
    file = open(stopWordsFile)
    stopWords = file.read().strip().split()
    return stopWords

# ----------------------------- Naive Bayes Fetching Test Data -----------------------------
def fetchTestData(dir):
    files = os.listdir(dir)
    map = {}
    vocab = []
    for file in files:
        current = open(dir + "/" + file, encoding="ISO-8859-1")
        words = current.read()
        allWords = words.strip().split()
        map[file] = allWords
        vocab.extend(allWords)
    return vocab, map

# ----------------------------- Naive Bayes Training and Accuracy -----------------------------
def naiveBayesTrain(countSpamDocs, countHamDocs, trainingSpamVocab, trainingHamVocab):
    global uniqueWords
    allSpamMap = Counter(trainingSpamVocab)
    allHamDict = Counter(trainingHamVocab)
    uniqueWords = list(set(allSpamMap) | set(allHamDict))
    totalUniqueWords = len(uniqueWords)
    # Spam
    likelihoodSpam = {}
    totalSpamWords = len(trainingSpamVocab)
    for term in uniqueWords:
        count = 0
        if term in allSpamMap:
            count = allSpamMap[term]
        probabilitySpam = (float)((count + 1) / (totalSpamWords + totalUniqueWords))
        likelihoodSpam[term] = probabilitySpam
    # Ham
    likelihoodHam = {}
    totalHamWords = len(trainingHamVocab)
    for term in uniqueWords:
        count = 0
        if term in allHamDict:
            count = allHamDict[term]
        probabilityHam = (float)((count + 1) / (totalHamWords + totalUniqueWords))
        likelihoodHam[term] = probabilityHam

    priorSpam = (float)(countSpamDocs / (countSpamDocs + countHamDocs))
    priorHam = (float)(countHamDocs / (countSpamDocs + countHamDocs))

    return priorSpam, priorHam, likelihoodSpam, likelihoodHam

def naiveBayesAccuracy(priorSpam, priorHam, likelihoodSpam, likelihoodHam, testingSpamMap, testingHamMap):
    global uniqueWords
    spamHamMap = [testingSpamMap, testingHamMap]
    count = 0
    for i in range(len(spamHamMap)):
        for j in spamHamMap[i]:
            spamProbTemp = log(priorSpam)
            hamProbTemp = log(priorHam)
            for term in spamHamMap[i][j]:
                if term in uniqueWords:
                    spamProbTemp = spamProbTemp + log(likelihoodSpam[term])
                    hamProbTemp = hamProbTemp + log(likelihoodHam[term])
            if (spamProbTemp >= hamProbTemp and i == 0):
                count += 1
            elif (spamProbTemp <= hamProbTemp and i == 1):
                count += 1
    return (float)(count / (len(testingSpamMap) + len(testingHamMap))) * 100

# ----------------------------- Naive Bayes Excluding Stop Words -----------------------------
def NBWithoutStopWords():
    trainingSpamVoc, trainingSpamMap = naiveBayesWithoutStopWords(spamTrainDir, stopWordsFile)
    trainingHamVoc, trainingHamMap = naiveBayesWithoutStopWords(hamTrainDir, stopWordsFile)
    testingSpamVoc, testingSpamMap = fetchTestData(spamTestDir)
    testingHamVoc, testingHamMap = fetchTestData(hamTestDir)
    priorSpam, priorHam, likelihoodSpam, likelihoodHam = naiveBayesTrain(len(trainingSpamMap), len(trainingHamMap),
                                                                         trainingSpamVoc,
                                                                         trainingHamVoc)
    accuracyWithoutStopWords = naiveBayesAccuracy(priorSpam, priorHam, likelihoodSpam, likelihoodHam, testingSpamMap,
                                                  testingHamMap)
    return accuracyWithoutStopWords

def naiveBayesWithoutStopWords(inputDir, stopWordsFilePath):
    files = os.listdir(inputDir)
    map = {}
    vocab = []
    stopWords = fetchAllStopWords(stopWordsFilePath)
    for f in files:
        file = open(inputDir + "/" + f, encoding="ISO-8859-1")
        words = file.read()
        words = re.sub('[^a-zA-Z]', ' ', words)
        allWords = words.strip().split()
        result = []
        for word in allWords:
            if (word not in stopWords):
                result.append(word)
        map[f] = result
        vocab.extend(result)
    return vocab, map

# ----------------------------- Naive Bayes Including Stop Words -----------------------------
def NBWithStopWords():
    trainingSpamVoc, trainingSpamMap = naiveBayesWithStopWords(spamTrainDir)
    trainingHamVoc, trainingHamMap = naiveBayesWithStopWords(hamTrainDir)
    testingSpamVoc, testingSpamMap = fetchTestData(spamTestDir)
    testingHamVoc, testingHamMap = fetchTestData(hamTestDir)
    priorSpam, priorHam, likelihoodSpam, likelihoodHam = naiveBayesTrain(len(trainingSpamMap), len(trainingHamMap),
                                                                         trainingSpamVoc,
                                                                         trainingHamVoc)
    accuracyWithStopWords = naiveBayesAccuracy(priorSpam, priorHam, likelihoodSpam, likelihoodHam, testingSpamMap,
                                               testingHamMap)
    return accuracyWithStopWords

def naiveBayesWithStopWords(dir):
    files = os.listdir(dir)
    map = {}
    vocab = []
    for file in files:
        current = open(dir + "/" + file, encoding="ISO-8859-1")
        words = current.read()
        allWords = words.strip().split()
        map[file] = allWords
        vocab.extend(allWords)
    return vocab, map

# ----------------------------- Naive Bayes Main()  -----------------------------
if __name__ == "__main__":
    print("************* ************** Begin of NaiveBayes ************* ************** ")

    print("Calculating Without Stop Words")
    accuracyWithoutStopWords = NBWithoutStopWords()
    print("Accuracy:", accuracyWithoutStopWords)
    print("Calculating With Stop Words")
    accuracyWithStopWords = NBWithStopWords()
    print("Accuracy:", accuracyWithStopWords)

    print("************* ************** End of NaiveBayes ************* ************** ")