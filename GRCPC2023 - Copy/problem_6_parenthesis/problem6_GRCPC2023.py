from collections import deque
from time import time

def loadInput(txtFileName: str) -> tuple:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()
    txtInput = text.split("\n")
    
    logLen = int(txtInput[0])
    log = txtInput[1]

    return (logLen, log);

def checkAnswer(txtFileName: str, answer: int) -> bool:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = int(file.read())

    passOrFail = True
    if text != answer:
        passOrFail = False

    return passOrFail;

def solution(len_log: tuple) -> int:
    logLen = len_log[0] # USELESS!!!
    log = len_log[1]

    stack = deque()
    for container in log:
        ASCII_value = ord(container)
        if ASCII_value >= 97: # ord('Z') = 90 & ord('a') = 97
            stack.append(ASCII_value)
        else:
            if stack and ((ASCII_value + 32) == stack.pop()):
                continue;
            else:
                return 0;

    if stack:
        return 0;

    return 1;

def main():
    startTime = time()

    inputLen = 10
    inputLs = [None] * inputLen
    for i in range(inputLen):
        i_str = str(i)
        inputLs[i] = "input_0{}_containterlogs".format(i_str)

    for testInput in inputLs:
        len_log = loadInput("{}.in".format(testInput))
        mySolution = solution(len_log)
        result = checkAnswer("{}.ans".format(testInput), mySolution)
        print("{}: {}".format(testInput, result))

    endTime = time()
    print("\nTime: {}".format(endTime - startTime))

    return;

if __name__ == "__main__":
    main()
