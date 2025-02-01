from random import seed, randint as randomInt
from time import time
from math import sqrt

def generateRandomArrayN(N):
    bias = 100
    tempList = [randomInt(-bias, bias) for i in range(N)]
    
    return tempList;

def returnResults(resultTuple):
    print("""Μεγαλύτερο συνεχόμενο άθροισμα: {}
Δείκτης αρχικού στοιχείου: {}
Δείκτης τελικού στοιχείου: {}\n""".format(resultTuple[0], resultTuple[1], resultTuple[2]))

    return;

def maxSubArray_On3(ls):
    lsLen = len(ls)
    maxSum = 0
    (start, end) = (-1, -1)
    for i in range(lsLen):
        for j in range(i, lsLen):
            tempSum = 0
            for o in range(i, j + 1):
                tempSum += ls[o]
                if tempSum > maxSum:
                    maxSum = tempSum
                    (start, end) = (i, j)

    return maxSum, start, end;

def maxSubArray_On2(ls):
    lsLen = len(ls)
    maxSum = 0
    (start, end) = (-1, -1)
    for i in range(lsLen):
        tempSum = 0
        for j in range(i, lsLen):
            tempSum += ls[j]
            if tempSum > maxSum:
                maxSum = tempSum
                (start, end) = (i, j)

    return maxSum, start, end;

def maxSubArray_On(ls):
    #Kadane's Algorithm
    lsLen = len(ls)
    maxSum = 0
    tempSum = 0
    tempStart = 0
    (start, end) = (-1, -1)
    for i in range(lsLen):
        tempSum += ls[i]
        if tempSum < 0:
            tempSum = 0
            if i + 1 < lsLen and ls[i + 1] >= 0:
                tempStart = i + 1
        elif tempSum > maxSum:
            maxSum = tempSum
            (start, end) = (tempStart, i)

    return maxSum, start, end;

def maxSubArray_Onlogn(ls):
    return maxSubArray_Onlogn_innerWorkingFunction1(ls, 0, len(ls) - 1)

def maxSubArray_Onlogn_innerWorkingFunction1(ls, lsStartPoint, lsEndPoint):
    #<<Διαίρει και βασίλευε>>
    if lsStartPoint == lsEndPoint:
        return ls[lsStartPoint], lsStartPoint, lsEndPoint;

    lsMidPoint = (lsStartPoint + lsEndPoint)//2
    resultFirstHalf = maxSubArray_Onlogn_innerWorkingFunction1(ls, lsStartPoint, lsMidPoint)
    resultSecondHalf = maxSubArray_Onlogn_innerWorkingFunction1(ls, lsMidPoint + 1, lsEndPoint)
    resultCombo = maxSubArray_Onlogn_innerWorkingFunction2(ls, lsStartPoint, lsMidPoint, lsEndPoint)

    if resultFirstHalf[0] >= resultSecondHalf[0] and resultFirstHalf[0] >= resultCombo[0]:
        result = resultFirstHalf
    elif resultSecondHalf[0] >= resultFirstHalf[0] and resultSecondHalf[0] >= resultCombo[0]:
        result = resultSecondHalf
    else:
        result = resultCombo
    
    return result;

def maxSubArray_Onlogn_innerWorkingFunction2(ls, lsStartPoint, lsMidPoint, lsEndPoint):
    tempSum = 0
    (maxSumLeft, maxSumRight) = (0, 0)
    (startIndex, endIndex) = (-1, -1)
    for i in range(lsMidPoint, lsStartPoint - 1, -1):
        tempSum += ls[i]
        if tempSum > maxSumLeft:
            maxSumLeft = tempSum
            startIndex = i
    
    tempSum = 0
    for i in range(lsMidPoint + 1, lsEndPoint + 1):
        tempSum += ls[i]
        if tempSum > maxSumRight:
            maxSumRight = tempSum
            endIndex = i
    
    maxComboSum = (maxSumLeft + maxSumRight, startIndex, endIndex)
    maxSumLeft = (maxSumLeft, startIndex, lsMidPoint)
    maxSumRight = (maxSumRight, lsMidPoint + 1, endIndex)
    if maxSumLeft[0] >= maxComboSum[0] and maxSumLeft[0] >= maxSumRight[0]:
        result = maxSumLeft
    elif maxComboSum[0] >= maxSumLeft[0] and maxComboSum[0] >= maxSumRight[0]:
        result = maxComboSum
    else:
        result = maxSumRight
    
    return result;

def testFunctionSpeed(func, ls, lsLen, printVariable = True):
    start = time()
    result = func(ls)
    stop = time()
    executionTime = stop - start
    if printVariable == True:
        print("Function's name: {}\nList's length: {}\nExecution time: {}\n".format(func.__name__, lsLen, executionTime))

    return func.__name__, executionTime, lsLen, result;

def main():
    seed(91588426223578521246982)
    # 1ο ερώτημα
    print("- Ερώτημα 1ο -")
    allAlgorithms = ((maxSubArray_On3, (100, 500)),
                     (maxSubArray_On2, (1000, 8000)),
                     (maxSubArray_Onlogn, (500000, 700000)),
                     (maxSubArray_On, (10000000, 20000000)))
    
    algorithmsExecutionInfo = {"maxSubArray_On3": 3 * [None],
                               "maxSubArray_On2": 3 * [None],
                               "maxSubArray_Onlogn": 3 * [None],
                               "maxSubArray_On": 3 * [None]} #Λεξικό ώστε να αποθηκευτούν οι χρόνοι εκτέλεσης για χρήση στο 2ο ερώτημα
    for algorithm in allAlgorithms:
        for i in range(3):
            lsLen = randomInt(algorithm[1][0], algorithm[1][1])
            ls = generateRandomArrayN(lsLen)
            result = testFunctionSpeed(algorithm[0], ls, lsLen)
            algorithmsExecutionInfo[result[0]][i] = result[1:]

    # 2ο ερώτημα
    print("- Ερώτημα 2ο -")
    estimatedLsLen3secTime_On = int((3 * algorithmsExecutionInfo["maxSubArray_On"][0][1]) / algorithmsExecutionInfo["maxSubArray_On"][0][0])
    testFunctionSpeed(maxSubArray_On, generateRandomArrayN(estimatedLsLen3secTime_On), estimatedLsLen3secTime_On) #Έλεγχος της παραπάνω εκτίμησης για το μεγέθος της λίστας

    estimatedLsLen3secTime_On2 = int(sqrt(((algorithmsExecutionInfo["maxSubArray_On2"][2][1]**2) * 3) / algorithmsExecutionInfo["maxSubArray_On2"][2][0]))
    testFunctionSpeed(maxSubArray_On2, generateRandomArrayN(estimatedLsLen3secTime_On2), estimatedLsLen3secTime_On2) #Έλεγχος της παραπάνω εκτίμησης για το μεγέθος της λίστας

    estimatedLsLen3secTime_On3 = int(pow(((algorithmsExecutionInfo["maxSubArray_On3"][0][1]**3) * 3) / algorithmsExecutionInfo["maxSubArray_On3"][0][0], 1/3))
    testFunctionSpeed(maxSubArray_On3, generateRandomArrayN(estimatedLsLen3secTime_On3), estimatedLsLen3secTime_On3) #Έλεγχος της παραπάνω εκτίμησης για το μεγέθος της λίστας

    (leftBound, rightBound) = (estimatedLsLen3secTime_On2, estimatedLsLen3secTime_On//10)
    executionTime = 0
    targetTime = 3
    while not ((targetTime - 0.2 <= executionTime) and (executionTime <= targetTime + 0.2)):
        estimatedLsLen3secTime_Onlogn = (leftBound + rightBound)//2
        executionTime = testFunctionSpeed(maxSubArray_Onlogn, generateRandomArrayN(estimatedLsLen3secTime_Onlogn), estimatedLsLen3secTime_Onlogn, printVariable = False)[1]
        if executionTime > targetTime:
            rightBound = estimatedLsLen3secTime_Onlogn
        else:
            leftBound = estimatedLsLen3secTime_Onlogn
    testFunctionSpeed(maxSubArray_Onlogn, generateRandomArrayN(estimatedLsLen3secTime_Onlogn), estimatedLsLen3secTime_Onlogn) #Έλεγχος της παραπάνω εκτίμησης για το μεγέθος της λίστας

    print("- ΤΕΛΟΣ ΠΡΟΓΡΑΜΜΑΤΟΣ -")

    return;

if __name__ == "__main__":
    main()
