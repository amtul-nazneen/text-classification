from numpy import *
import os
import re

# -----------------------------Logistic Regression Configurable Parameters -----------------------------
spamTrainDir = "train/spam/"
hamTrainDir = "train/ham"
spamTestDir = "test/spam"
hamTestDir = "test/ham"
stopWordsFile = "stop_words_list.txt"
lambdaVal = 0.1
totalIterations = 100
learningRate = 0.1

# ----------------------------- Logistic Regression Fetching Stop Words -----------------------------
def fetchAllStopWords(stopWordsFile):
    file = open(stopWordsFile)
    stopWords = file.read().strip().split()
    return stopWords

# ----------------------------- Logistic Regression Fetching Test Data -----------------------------
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

# ----------------------------- Logistic Regression Training and Accuracy -----------------------------
def logisticRegTrain(train, list, lambdaVar):
    matrix = mat(train)
    p, q = shape(matrix)
    labelMatrix = mat(list).transpose()
    weight = zeros((q, 1))
    for iteration in range(totalIterations):
        predict_condProb = 1.0 / (1 + exp(-matrix * weight))
        error = labelMatrix - predict_condProb
        weight = weight + learningRate * matrix.transpose() * error - learningRate * lambdaVar * weight
    return weight

def logisticRegAccuracy(weight, test, testingSpamMapLength, testingHamMapLength):
    matrix = mat(test)
    result = matrix * weight
    val = 0
    allMapLength = testingSpamMapLength + testingHamMapLength
    for i in range(testingSpamMapLength):
        if (result[i][0] < 0.0):
            val += 1
    i = 0
    for i in range(testingSpamMapLength + 1, allMapLength):
        if (result[i][0] > 0.0):
            val += 1
    return (float)(val / allMapLength) * 100

def vector(allUniqueWords, map):
    featureResult = []
    for file in map:
        row = [0] * (len(allUniqueWords))
        for i in allUniqueWords:
            if (i in map[file]):
                row[allUniqueWords.index(i)] = 1
        row.insert(0, 1)
        featureResult.append(row)
    return featureResult

# ----------------------------- Logistic Regression Excluding Stop Words -----------------------------
def LRWithoutStopWords():
    trainingSpamVoc, trainingSpamMap = logisticRegWithoutStopWords(spamTrainDir, stopWordsFile)
    trainingHamVocab, trainingHamMap = logisticRegWithoutStopWords(hamTrainDir, stopWordsFile)
    testingSpamVocab, testingSpamDict = fetchTestData(spamTestDir)
    testingHamVocab, testingHamDict = fetchTestData(hamTestDir)
    uniqueWords = list(set(trainingSpamVoc) | set(trainingHamVocab))
    allTrainingMap = trainingSpamMap.copy()
    allTrainingMap.update(trainingHamMap)
    allTestingMap = testingSpamDict.copy()
    allTestingMap.update(testingHamDict)
    captureList = []
    for counter in range(len(trainingSpamMap)):
        captureList.append(0)
    for counter in range(len(trainingHamMap)):
        captureList.append(1)
    trainingFeatures = vector(uniqueWords, allTrainingMap)
    testingFeatures = vector(uniqueWords, allTestingMap)
    weight = logisticRegTrain(trainingFeatures, captureList, lambdaVal)
    accuracyWithoutStopWords = logisticRegAccuracy(weight, testingFeatures, len(testingSpamDict), len(testingHamDict))
    return accuracyWithoutStopWords

def logisticRegWithoutStopWords(dir, stopWordsFile):
    files = os.listdir(dir)
    map = {}
    vocab = []
    stopWords = fetchAllStopWords(stopWordsFile)
    for file in files:
        current = open(dir + "/" + file, encoding="ISO-8859-1")
        words = current.read()
        words = re.sub('[^a-zA-Z]', ' ', words)
        allWords = words.strip().split()
        result = []
        for word in allWords:
            if (word not in stopWords):
                result.append(word)
        map[file] = result
        vocab.extend(result)
    return vocab, map
# ----------------------------- Logistic Regression Including Stop Words -----------------------------
def LRWithStopWords():
    trainingSpamVoc, trainingSpamMap = logisticRegWithStopWords(spamTrainDir)
    trainingHamVocab, trainingHamMap = logisticRegWithStopWords(hamTrainDir)
    testingSpamVocab, testingSpamDict = fetchTestData(spamTestDir)
    testingHamVocab, testingHamDict = fetchTestData(hamTestDir)
    uniqueWords = list(set(trainingSpamVoc) | set(trainingHamVocab))
    allTrainingMap = trainingSpamMap.copy()
    allTrainingMap.update(trainingHamMap)
    allTestingMap = testingSpamDict.copy()
    allTestingMap.update(testingHamDict)
    captureList = []
    for counter in range(len(trainingSpamMap)):
        captureList.append(0)
    for counter in range(len(trainingHamMap)):
        captureList.append(1)
    trainingFeatures = vector(uniqueWords, allTrainingMap)
    testingFeatures = vector(uniqueWords, allTestingMap)
    weight = logisticRegTrain(trainingFeatures, captureList, lambdaVal)
    accuracyWithStopWords = logisticRegAccuracy(weight, testingFeatures, len(testingSpamDict), len(testingHamDict))
    return accuracyWithStopWords

def logisticRegWithStopWords(dir):
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

# ----------------------------- Logistic Regression Main()  -----------------------------
if __name__ == "__main__":
    print("************* ************** Begin of Logistic Regression ************* ************** ")
    print("Lambda:" ,lambdaVal , ", Learning Rate:", learningRate , ", Iterations:",totalIterations)
    print("Calculating Without Stop Words")
    accuracyWithoutStopWords = LRWithoutStopWords()
    print("Accuracy: ", accuracyWithoutStopWords)
    print("Calculating With Stop Words")
    accuracyWithStopWords = LRWithStopWords()
    print("Accuracy: ", accuracyWithStopWords)

    print("************* ************** End of Logistic Regression ************* ************** ")
