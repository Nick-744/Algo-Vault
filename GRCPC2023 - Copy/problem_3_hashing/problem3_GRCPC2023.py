from time import time

def loadInput(txtFileName: str) -> tuple:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()
    txtInput = text.split("\n")

    setA = set(txtInput[1].split(" ")) # HASHMAP ΦΕΙΔΑΣΣΣ
    setB = txtInput[3].split(" ")

    return (setA, setB);

def checkAnswer(txtFileName: str, answer: str) -> bool:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()

    passOrFail = True
    if text != answer:
        passOrFail = False

    return passOrFail;

def solution(setAsetB: tuple) -> str:
    (setA, setB) = setAsetB

    for course in setB:
        if course not in setA:
            return "0";

    return "1";

def main():
    startTime = time()

    inputLen = 18 + 1
    inputLs = []
    for i in range(inputLen):
        if i == 2:
            continue;
        i_str = str(i)
        inputLs.append("input_{}_courses".format(i_str if (len(i_str) >= 2) else "0{}".format(i_str)))

    for testInput in inputLs:
        setAsetB = loadInput("{}.in".format(testInput))
        mySolution = solution(setAsetB)
        result = checkAnswer("{}.ans".format(testInput), mySolution)
        print("{}: {}".format(testInput, result))

    endTime = time()
    print("\nTime: {}".format(endTime - startTime))

    return;

if __name__ == "__main__":
    main()
