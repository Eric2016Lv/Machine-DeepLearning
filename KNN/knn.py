# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:11:28 2018
E-mail: Eric2014_Lv@sjtu.edu.cn
@author: DidiLv
Python Version: 3.5
"""

from numpy import *
from os import listdir
import operator
import matplotlib
import matplotlib.pyplot as plt



def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # inX: array; 
    # dataSet: 2-dimension dataSet; 
    # labels: dataSet labels; 
    # k: classify number
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat * diffMat
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort() # arg ascent
    classCount = {}
    for i in range(k): # k neighborhood after sorted distance
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # 0 is the value by default
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
#    print(sortedClassCount)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() # 去除首尾空格
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0) # return min value rowwise: vector
    maxVals = dataSet.max(0) # return max value rowwise: vector
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = int(normMat.shape[0])
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], 
                                     datingLabels[numTestVecs:m], 4)
                                     
        print("The classifier came back with: %d, the real answer is: %d" 
              %(classifierResult, datingLabels[i]))
              
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("The total error rate is %f" %(errorCount/float(numTestVecs)))
   

def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: %s" %(resultList[classifierResult - 1]))

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
        return returnVect
    
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     # take off ".txt"
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     # take off ".txt"
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 300)
        print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n The total number of errors is: %d" % errorCount)
    print("\n The total error rate is: %f" % (errorCount/float(mTest)))
        

def main():
    group, labels = createDataSet()
    inX = array([0, 0])
    k = 3
    label_class = classify0(inX, group, labels, k)
    print("Project1: classify0--> ", label_class)
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print("Project2: plot dataTestSet2 ")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    plt.show()
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print("Project3: autoNorm --> ", normMat, ranges, minVals)
    print("Project4: datingClassTest -->")
    datingClassTest()
    print("Project5: classifyPerson -->")
    classifyPerson()
    print("Project6: handwritingClassTest -->")
    handwritingClassTest()
    
    
if __name__ == "__main__":
    main()


    
    
    
    