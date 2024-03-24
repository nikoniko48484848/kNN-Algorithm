import kNN

k = 50
kNN.makeGroups()
kNN.calculateAllDistances()
kNN.findAllkNN(k)
# for k in range(1, 51):
#     print(kNN.checkAccuracy(k)["formatted])
print(kNN.checkAccuracy(k)["formatted"])
kNN.userInputVector()
kNN.plotAccuracyVsK(k)