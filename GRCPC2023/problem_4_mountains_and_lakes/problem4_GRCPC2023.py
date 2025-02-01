from itertools import groupby
from time import time

def loadInput(txtFileName: str) -> list:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()
    txtInput = text.split("\n")
    while txtInput[-1] == '':
        txtInput.pop()
    
    (n, ls) = txtInput # n: UseLess...
    ls = list(map(int, ls.split()))
    # Tuples use less memory and are faster to access than to lists!!!
    # ΤΕΛΙΚΑ ΗΘΕΛΕ LIST ΓΙΑ ΝΑ ΤΟ ΣΠΑΣΩ ΣΤΑ 0!!!

    return ls;

def checkAnswer(txtFileName: str, answer: str) -> bool:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = int(file.read())

    passOrFail = True
    if text != answer:
        passOrFail = False

    return passOrFail;

def solution(ls: list) -> int:
    def trapHelper(height: list, start: int, end: int, step: int):
        s = start  # s: Start Pointer | e: End Pointer
        trappedWater = 0
        currentTrapped = 0
        zeroFlag = False
        for e in range(start, end, step):
            if height[e] == 0:
                zeroFlag = True
            
            if height[s] <= height[e]:
                if not zeroFlag:
                    trappedWater += currentTrapped
                zeroFlag = False
                currentTrapped = 0
                s = e
            else:
                currentTrapped += height[s] - height[e]

        return trappedWater;

    heights_ls = [list(g) for k, g in groupby(ls, lambda x: x == 0) if not k]
    
    trappedWater = 0
    for height in heights_ls:
        maxHeightIndex = 0
        for i in range(1, len(height)):
            if height[i] > height[maxHeightIndex]:
                maxHeightIndex = i

        trappedWaterLeft = trapHelper(height, 0, maxHeightIndex + 1, 1)
        trappedWaterRight = trapHelper(height, len(height) - 1, maxHeightIndex - 1, -1)
        trappedWater += trappedWaterLeft + trappedWaterRight

    return trappedWater;

def main():
    startTime = time()
    
    inputLen = 22
    inputLs = [None] * inputLen
    for i in range(inputLen):
        i_str = str(i + 1)
        inputLs[i] = "input_{}".format(i_str if (len(i_str) >= 2) else "0{}".format(i_str))

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
