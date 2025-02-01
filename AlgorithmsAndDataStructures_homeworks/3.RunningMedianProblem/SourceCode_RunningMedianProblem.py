from random import seed, choice, uniform as randomFloat, randint as randomInt
from minHeap import MinHeap as MH
from time import time

def medianFromMeas(NtM, N):
    #NtM: N temperature Measurements & N: Πλήθος μετρήσεων
    largeHeap = MH([]) #min Heap
    smallHeap = MH([]) #Max Heap
    
    resultList = [None] * 2
    resultIndex = 0
    bias = 10
    
    for i in range(bias):
        key = str(NtM[i][1][0]) + str(NtM[i][1][1])
        value = NtM[i][0]

        if smallHeap.isInMinHeap(key):
            smallHeap.changeKey((key, -value))
        else:
            smallHeap.insert((key, -value)) #push(smallHeap, -temp)
            if largeHeap.isInMinHeap(key):
                largeHeap.deleteKey((key, value)) #Άρα, το σημείο μπορεί να υπάρχει μόνο σε έναν από τους 2 heap!
        
        (smallHeap_maxKey, smallHeap_maxValue) = smallHeap.extractMin()
        largeHeap.insert((smallHeap_maxKey, -smallHeap_maxValue)) #push(largeHeap, -smallHeap_max)
        
        if largeHeap.size > smallHeap.size:
            (largeHeap_minKey, largeHeap_minValue) = largeHeap.extractMin()
            smallHeap.insert((largeHeap_minKey, -largeHeap_minValue)) #push(smallHeap, -largeHeap_min)

    for i in range(bias, N - bias):
        key = str(NtM[i][1][0]) + str(NtM[i][1][1])
        value = NtM[i][0]

        if largeHeap.size != smallHeap.size:
            median = -smallHeap.getMin()[1]
        else:
            median = round(((largeHeap.getMin()[1] - smallHeap.getMin()[1]) / 2), 2)
        if (i == ((N - 1) // 2)) or (i == (N - 1 - bias)):
            resultList[resultIndex] = median
            resultIndex += 1
        ############################################################################# value > median
        if value > median and largeHeap.size > smallHeap.size:
            (largeHeap_minKey, largeHeap_minValue) = largeHeap.extractMin()
            smallHeap.insert((largeHeap_minKey, -largeHeap_minValue))
            if largeHeap.isInMinHeap(key):
                largeHeap.changeKey((key, value))
            else:
                largeHeap.insert((key, value))
                if smallHeap.isInMinHeap(key):
                    smallHeap.deleteKey((key, -value))
        elif value >= median and largeHeap.size <= smallHeap.size:
            if largeHeap.isInMinHeap(key):
                largeHeap.changeKey((key, value))
            else:
                largeHeap.insert((key, value))
                if smallHeap.isInMinHeap(key):
                    smallHeap.deleteKey((key, -value))
        ############################################################################# value < median
        elif value <= median and largeHeap.size >= smallHeap.size:
            if smallHeap.isInMinHeap(key):
                smallHeap.changeKey((key, -value))
            else:
                smallHeap.insert((key, -value))
                if largeHeap.isInMinHeap(key):
                    largeHeap.deleteKey((key, value))
        elif value < median and largeHeap.size < smallHeap.size:
            (smallHeap_maxKey, smallHeap_maxValue) = smallHeap.extractMin()
            largeHeap.insert((smallHeap_maxKey, -smallHeap_maxValue))
            if smallHeap.isInMinHeap(key):
                smallHeap.changeKey((key, -value))
            else:
                smallHeap.insert((key, -value))
                if largeHeap.isInMinHeap(key):
                    largeHeap.deleteKey((key, value))
        ############################################################################# largeHeap.size == smallHeap.size
        else:
            if largeHeap.isInMinHeap(key):
                largeHeap.changeKey((key, value))
            else:
                largeHeap.insert((key, value))
                if smallHeap.isInMinHeap(key):
                    smallHeap.deleteKey((key, -value))

    return (resultList, (largeHeap.size, smallHeap.size));

def main():
    seed(19816132165161654)
    N_pair = (500_000, 1_000_000) #Πλήθος μετρήσεων

    for N in N_pair:
        rMP = [(randomInt(0, 999), randomInt(0, 999)) for _ in range(100_000)] #100000 random Measured Points
        NtM = [(round(randomFloat(-10, 90), 2), choice(rMP)) for _ in range(N)] #NtM: N temperature Measurements

        start = time()
        (mediansResults, heapsSize) = medianFromMeas(NtM, N)
        stop = time()
        executionTime = stop - start

        print("Χρόνος εκτέλεσης για N = {} μετρήσεις: {} sec".format(N, executionTime))
        print("Διάμεση τιμή της θερμοκρασίας, μετά την δημιουργία των μισών[Ν/2] μετρήσεων: {}".format(mediansResults[0]))
        print("Διάμεση τιμή της θερμοκρασίας, μετά την δημιουργία όλων των μετρήσεων: {}".format(mediansResults[1]))
        print("Τελικό μέγεθος της δομής δεδομένων Large Heap[min]: {}".format(heapsSize[0]))
        print("Τελικό μέγεθος της δομής δεδομένων small Heap[Max]: {}\n".format(heapsSize[1]))

    return;

if __name__ == "__main__":
    main()
