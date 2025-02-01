from math import sqrt

class MyHashTable:
    def __init__(self, startSize = 101, maxLoadFactor = 0.7):
        self.itemsCount = 0
        self.size = startSize
        self.maxLoadFactor = maxLoadFactor
        self.innerWorkingList = [None] * self.size

        return;

    def __innerWorkingHashFunction(self, itemKey):
        return self.__polyAccumHashCode(itemKey) % self.size;

    def __polyAccumHashCode(self, word):
        z = 33
        code = 0
        for c in word:
            code = z * code + ord(c) # % 2**31
        return code

    def __innerWorkingRehashing(self):
        oldSize = self.size
        oldList = self.innerWorkingList
        self.size = self.__findPrime(oldSize)
        self.innerWorkingList = [None] * self.size
        for i in range(oldSize):
            if oldList[i] != None:
                self.insertValue(oldList[i][0], oldList[i][1], True)

        return;
    
    def __findPrime(self, number):
        #Sieve of Eratosthenes
        """However, there is a conjecture in number theory called the Bertrand’s Postulate
        (also known as Bertrand’s Conjecture) which states that for any integer n>1,
        there always exists at least one prime number p such that n<p<2n.
        This means that for any given number, you can find a prime number
        less than twice that number."""
        upperSearchLimit = 4 * number
        truthPrimeList = [True] * upperSearchLimit
        for i in range(2, round(sqrt(upperSearchLimit)) + 1):
            if truthPrimeList[i] == True:
                for j in range(i*i, upperSearchLimit, i):
                    truthPrimeList[j] = False
        
        for i in range(2, upperSearchLimit):
            if truthPrimeList[i] and i > 2 * number:
                prime = i
                break

        return prime;

    def __calculateLoadFactor(self):
        return self.itemsCount / self.size;

    def insertValue(self, itemKey, itemValue, innerCall = False):
        index = self.__innerWorkingHashFunction(itemKey)
        while self.innerWorkingList[index] != None:
            index = (index + 1) % self.size
        self.innerWorkingList[index] = (itemKey, itemValue)
        if innerCall == False:
            self.itemsCount += 1

        if self.__calculateLoadFactor() > self.maxLoadFactor:
            self.__innerWorkingRehashing()

        return;
    
    def getValue(self, itemKey):
        index = self.__innerWorkingHashFunction(itemKey)
        while self.innerWorkingList[index] != None:
            if self.innerWorkingList[index][0] == itemKey:
                return self.innerWorkingList[index][1];
            index = (index + 1) % self.size
        
        return None;
    
    def values(self):
        values = [None] * self.itemsCount
        i = 0
        for pair in self.innerWorkingList:
            if pair != None:
                values[i] = pair[1]
                i += 1

        return values;

    def keyExists(self, itemKey):
        index = self.__innerWorkingHashFunction(itemKey)
        while self.innerWorkingList[index] != None:
            if self.innerWorkingList[index][0] == itemKey:
                return True;
            index = (index + 1) % self.size
        
        return False;
