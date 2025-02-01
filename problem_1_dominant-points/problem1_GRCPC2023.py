from time import time

class Alloy:
    def __init__(self, identification: str, flexibility: float, conductivity: float):
        self.ident = identification
        self.flex = flexibility
        self.con = conductivity

        return;

def maxAlloyFlexOrCon(ls: list[Alloy]) -> tuple:
    (maxFlex, maxCon) = (ls[0], ls[0])

    for i in range(1, len(ls)):
        if ls[i].flex > maxFlex.flex:
            maxFlex = ls[i]
        if ls[i].con > maxCon.con:
            maxCon = ls[i]

    return (maxFlex, maxCon);

def loadInput(txtFileName: str) -> list[Alloy]:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()
    txtInput = text.split("\n")
    
    listLen = int(txtInput[0])
    ls = [None] * listLen
    for i in range(1, listLen + 1):
        idFlexCon = txtInput[i].split(" ")
        ls[i - 1] = Alloy(idFlexCon[0], float(idFlexCon[1]), float(idFlexCon[2]))

    return ls;

def checkAnswer(txtFileName: str, answer: str) -> bool:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()

    passOrFail = True
    if text != answer:
        passOrFail = False

    return passOrFail;

def solution(ls: list[Alloy]) -> str:
    solution = set()
    
    while ls:
        (maxFlex, maxCon) = maxAlloyFlexOrCon(ls)
        solution.add(maxFlex.ident)
        solution.add(maxCon.ident)

        lenLs = len(ls)
        interestingAlloys = []
        for i in range(lenLs):
            if (ls[i].flex >= maxCon.flex) and (ls[i].con >= maxFlex.con) and (ls[i] != maxFlex) and (ls[i] != maxCon):
                interestingAlloys.append(ls[i])
                                         
        ls = interestingAlloys

    solution = list(solution)
    solution.sort()
    solutionStr = " ".join(solution)

    return solutionStr;

def main():
    startTime = time()
    
    inputLen = 24
    inputLs = [None] * inputLen
    for i in range(inputLen):
        i_str = str(i)
        inputLs[i] = "input_{}_alloy".format(i_str if (len(i_str) >= 2) else "0{}".format(i_str))

    for testInput in inputLs:
        ls = loadInput("{}.in".format(testInput))
        mySolution = solution(ls)
        result = checkAnswer("{}.ans".format(testInput), mySolution)
        print("{}: {}".format(testInput, result))

    endTime = time()
    print("\nTime: {}".format(endTime - startTime))

    return;

if __name__ == "__main__":
    main()
