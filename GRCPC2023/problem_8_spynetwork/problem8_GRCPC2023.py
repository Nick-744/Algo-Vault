from functools import cache
from time import time

def loadInput(txtFileName: str) -> tuple:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()
    txtInput = text.split("\n")
    txtInput.pop() # ΠΩΣ ΘΑ ΤΟ ΞΕΡΑΜΕ ΕΚΕΙ;;;

    (n, wordsPositions, keyWords, paragraph) = txtInput # n: USELESS!!!
    wordsPositions = [(int(x) - 1) for x in wordsPositions.split(" ")]
    keyWords = keyWords.split(" ")
    paragraph = paragraph.split(" ")

    return (wordsPositions, keyWords, paragraph);

def checkAnswer(txtFileName: str, answer: str) -> bool:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()

    passOrFail = True
    if text != answer:
        passOrFail = False

    return passOrFail;

@cache
def levenshtein(s: str, t: str) -> int:
    if s == "":
        return len(t);
    if t == "":
        return len(s);

    p1 = levenshtein(s, t[:-1]) + 1
    p2 = levenshtein(s[:-1], t) + 1

    p3 = levenshtein(s[:-1], t[:-1])
    if s[-1] != t[-1]:
        p3 += 1

    return min(p1, p2, p3);

def solution(n_w_k_p: tuple) -> str:
    (wordsPositions, keyWords, paragraph) = n_w_k_p
    solution = ""

    for (i, j) in enumerate(wordsPositions):
        solution += f"{str(levenshtein(keyWords[i], paragraph[j]))} "

    return solution[:-1];

def main():
    startTime = time()
    
    inputLen = 12
    inputLs = [None] * inputLen
    for i in range(inputLen):
        i_str = str(i + 1)
        inputLs[i] = "input_{}_spynetwork".format(i_str if (len(i_str) >= 2) else "0{}".format(i_str))

    for testInput in inputLs:
        w_k_p = loadInput("{}.in".format(testInput))
        mySolution = solution(w_k_p)
        result = checkAnswer("{}.ans".format(testInput), mySolution)
        print("{}: {}".format(testInput, result))

    endTime = time()
    print("\nTime: {}".format(endTime - startTime))

    return;

if __name__ == "__main__":
    main()
