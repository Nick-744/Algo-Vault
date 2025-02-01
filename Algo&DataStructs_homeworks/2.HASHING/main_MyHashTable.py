from MyHashTable import MyHashTable as HT
from random import randint, choice, seed
from time import time

class Card:
    def __init__(self, cardID):
        self.cardID = cardID
        self.charge = 0
        self.transactionsNumber = 0
        self.week = [0] * 6

        return;
    
    def printCardInfo(self):
        print("Κάρτα: ID = {} | χρέωση = {:5d} | κινήσεις = {:3d} | εβδομάδα = {}".format(self.cardID,\
            self.charge, self.transactionsNumber, " ".join(str(self.week[day]) for day in range(6))))
        
        return;

    def makeTransaction(self, charge, day, printTransactionStatement = False):
        self.transactionsNumber += 1
        self.charge += charge
        self.week[day] += 1
        if printTransactionStatement == True:
            print("Συναλλαγή από την κάρτα με ID: {} | Χρέωση = {:5d} | Μέρα = {}".format(self.cardID, charge, day))

        return;

def makeCardID():
    return "{}-{}-{}-{}".format(randint(1000, 9999), randint(1000, 9999),\
        randint(1000, 9999), randint(1000, 9999));

def runEconomy(cardsList, num):
    ht = HT()
    for i in range(num):
        cardID = choice(cardsList)
        charge = randint(5, 500)
        day = randint(0, 5)
        if ht.keyExists(cardID):
            ht.getValue(cardID).makeTransaction(charge, day)
        else:
            ht.insertValue(cardID, Card(cardID))
            ht.getValue(cardID).makeTransaction(charge, day)

    return ht;

def printAllCardsInfo(ht):
    maxCharge = -1
    maxChargeCard = None
    minCharge = float("inf")
    minChargeCard = None
    maxTransactionsNumber = -1
    maxTransactionsNumberCard = None

    for card in ht.values():
        if maxCharge < card.charge:
            maxCharge = card.charge
            maxChargeCard = card
        if minCharge > card.charge:
            minCharge = card.charge
            minChargeCard = card
        if maxTransactionsNumber < card.transactionsNumber:
            maxTransactionsNumber = card.transactionsNumber
            maxTransactionsNumberCard = card

    totalDayTransactions = [0] * 6
    for card in ht.values():
        for day in range(6):
            totalDayTransactions[day] += card.week[day]

    minTransactions = min(totalDayTransactions)
    minTotalDayTransactions = totalDayTransactions.index(minTransactions)

    print("- Μέγιστο κόστος συναλλαγών = {}:".format(maxCharge))
    maxChargeCard.printCardInfo()
    print("- Ελάχιστο κόστος συναλλαγών = {}:".format(minCharge))
    minChargeCard.printCardInfo()
    print("- Μέγιστος αριθμός συναλλαγών = {}:".format(maxTransactionsNumber))
    maxTransactionsNumberCard.printCardInfo()
    print("- Ημέρα ελάχιστων συναλλαγών: {} | Ελάχιστες συναλλαγές: {}"\
        .format(minTotalDayTransactions, minTransactions))

    return;

def main():
    seed(1681861564189646161)

    cardsNum = 20_000
    iterationsNumList = [1_000_000, 2_000_000, 5_000_000]
    cardsList = [None] * cardsNum
    for i in range(cardsNum):
        cardsList[i] = makeCardID()

    for i in iterationsNumList:
        print("- Εκτέλεση αλγορίθμου για {} χρεώσεις -\n".format(i))
        startTime = time();

        ht = runEconomy(cardsList, i)
        printAllCardsInfo(ht)

        endTime = time();
        print("Χρόνος εκτέλεσης: {}\n".format(endTime - startTime))
    
    print("-- ΤΕΛΟΣ ΠΡΟΓΡΑΜΜΑΤΟΣ --")

    return;

if __name__ == "__main__":
    main()
