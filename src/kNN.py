import math
import pandas as pd
import matplotlib.pyplot as plt

groupsDict = {}

def vectorSize():
    file = open("train-set.csv")
    return len(file.readline().split(";"))-1

def checkAccuracy(k):
    file = open("test-set.csv")
    i = 0
    count = 0
    list = findAllkNN(k)
    for line in file.readlines():
        if line.strip().split(";")[-1] == list[i]:
            count += 1
        i += 1
    return {"formatted" : str(k) + ". accuracy: " + str((count/len(list))*100) + "%", "numeric" : count/len(list)}


def makeGroups():
    file = open("train-set.csv")
    while True:
        line = file.readline()
        if not line:
            break
        strSplit = line.strip().split(";")
        conc = ";".join(strSplit[:-1])
        if strSplit[-1] not in groupsDict:
            groupsDict[strSplit[-1]] = []
        groupsDict[strSplit[-1]].append(conc)

def calculateAllDistances():
   file = open("test-set.csv")
   data = file.read()
   lines = data.split("\n")

   allDistances = []
   for line in lines:
      vec1 = line.split(";")
      allDistances.append((calculateDistances(vec1[:-1]), line))
   return allDistances

def calculateDistances(vec1):
    distances = []
    for groupName, groupVecs in groupsDict.items():
        for vec in groupVecs:
            vec2 = vec.split(";")
            dist = vectorDistance(vec1, vec2)
            distances.append((dist, groupName))
    return distances

def findAllkNN(k):
    resList = []
    for e in calculateAllDistances():
        findkNN(k, e[0], e[1], resList)
    return resList

def findkNN(k, distances, vec, res=[]):
    distances.sort()
    kNearest = distances[:k]
    # print(kNearest)
    counts = {}
    for t in kNearest:
        if t[1] not in counts:
            counts[t[1]] = 1
        else:
            counts[t[1]] += 1
    # print(counts)
    res.append(max(counts, key=counts.get))
    print("Vector: " + vec + " belongs to group: " + max(counts, key=counts.get))
    # If multiple items are maximal, the function returns the first one encountered.
    # This is consistent with other sort-stability preserving tools such as
    # sorted(iterable, key=keyfunc, reverse=True)[0]
    # and heapq.nlargest(1, iterable, key=keyfunc).

def vectorDistance(vec1, vec2):
    sum = 0
    for i in range(len(vec1)):
        sum += (float(vec1[i]) - float(vec2[i]))**2
    return math.sqrt(sum)

def userInputVector():
    print("Enter " + str(vectorSize()) + " float numbers separeted by spaces and an integer k at the end.\nType X to exit.")
    while True:
        userInput = input()
        inputVector = userInput.split(" ")

        if userInput == "X":
            break

        if len(inputVector) != vectorSize()+1:
            print("Invalid input!\nPlease enter " + str(vectorSize()) + " float numbers separeted by spaces and an integer k at the end.\nType X to exit.")
        else:
            floatInputVector = list(map(float, inputVector[:-1]))
            distance = calculateDistances(floatInputVector)
            formattedVector = ";".join(inputVector[:-1])
            findkNN(int(floatInputVector[-1]), distance, formattedVector)
            print("Enter " + str(vectorSize()) + " float numbers separeted by spaces and an integer k at the end.\nType X to exit.")

def plotAccuracyVsK(topK):
    accuracyList = []
    for k in range(1, topK+1):
        accuracy = checkAccuracy(k)["numeric"]
        accuracyList.append((k, accuracy))

    df = pd.DataFrame(accuracyList, columns=["k", "Accuracy"])
    # print(df)
    plt.plot(df["k"], df["Accuracy"])
    plt.grid(True)
    plt.title("Accuracy/k")
    plt.xlabel("k")
    plt.ylabel("Accuracy %")
    plt.show()