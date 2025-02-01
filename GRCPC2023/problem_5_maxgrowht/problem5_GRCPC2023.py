from collections import deque
from time import time

def loadInput(txtFileName: str) -> tuple:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()
    txtInput = text.split("\n")
    
    n = int(txtInput[0]) # The number of available temperature measurements
    m = int(txtInput[1]) # The maximum time interval in days
    
    ls = [float(x) for x in txtInput[2].split(" ")]

    return (n, m, ls);

def checkAnswer(txtFileName: str, answer: str) -> bool:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()

    passOrFail = True
    if "{:.6f}".format(float(text)) != answer: # Ensure that your output has an absolute or relative error of at most 10^-6??????
        passOrFail = False

    return passOrFail;

def solution(n_m_ls: tuple) -> str:
    q = deque() # Temperature Index Q (Monotonic)
    (n, m, ls) = n_m_ls
    solution = 0
    l = 0

    for r in range(n):
        while q and ls[q[-1]] >= ls[r]:
            q.pop()
        q.append(r)

        if l > q[0]:
            q.popleft()

        if r >= (m - 1):
            l += 1

        increase = ls[q[-1]] - ls[q[0]]
        if increase > solution:
            solution = increase

    return "{:.6f}".format(solution); # Ensure that your output has an absolute or relative error of at most 10^-6??????

def main():
    startTime = time()
    
    inputLen = 8
    inputLs = []
    for i in range(0, inputLen, 2):
        if i == 4:
            i = 3
        i_str = str(i + 1)
        inputLs.append("input_{}_tempe".format(i_str))

    for testInput in inputLs:
        n_m_ls = loadInput("{}.in".format(testInput))
        mySolution = solution(n_m_ls)
        result = checkAnswer("{}.ans".format(testInput), mySolution)
        print("{}: {}".format(testInput, result))

    endTime = time()
    print("\nTime: {}".format(endTime - startTime))

    return;

if __name__ == "__main__":
    main()
