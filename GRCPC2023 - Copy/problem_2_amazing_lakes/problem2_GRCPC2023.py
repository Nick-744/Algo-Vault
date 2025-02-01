from collections import deque
from time import time

def loadInput(txtFileName: str) -> tuple:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()
    txtInput = text.split("\n")
    
    dimensions = txtInput[0].split(" ")
    NandM = (int(dimensions[0]), int(dimensions[1]))
    
    ls = [[0] * NandM[1] for _ in range(NandM[0])] 
    for i in range(NandM[0]):
        for j in range(NandM[1]):
            ls[i][j] = int(txtInput[i + 1][j])

    return (ls, NandM);

def checkAnswer(txtFileName: str, answer: str) -> bool:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()

    passOrFail = True
    if text != answer:
        passOrFail = False

    return passOrFail;

def solution(lsAndDimensions: tuple) -> str:
    (ls, (n, m)) = lsAndDimensions

    visited = set()
    lakesCount = 0
    lakesAreas = []

    # BFSSSSSSSSSSSSSSSSSS!!!
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            if (i, j) in visited or ls[i][j] == 0:
                continue;
            
            visited.add((i, j))
            lakesCount += 1
            area = 0
            
            queue = deque([(i, j)])
            while queue:
                current = queue.popleft()
                area += 1

                neighbors = [(current[0] + 1, current[1]),
                             (current[0] - 1, current[1]),
                             (current[0], current[1] - 1),
                             (current[0], current[1] + 1)]
                for (k, o) in neighbors:
                    if ((k, o) not in visited) and (ls[k][o] == 1):
                        visited.add((k, o))
                        queue.append((k, o))

            lakesAreas.append(area)

    lakesAreas.sort()
    solution = "{}\n{}\n".format(lakesCount, " ".join(map(str, lakesAreas)))

    return solution;

def main():
    startTime = time()
    
    inputLen = 20
    inputLs = [None] * inputLen
    for i in range(inputLen):
        i_str = str(i + 1)
        inputLs[i] = "input_{}".format(i_str if (len(i_str) >= 2) else "0{}".format(i_str))

    for testInput in inputLs:
        lsAndDimensions = loadInput("{}.in".format(testInput))
        mySolution = solution(lsAndDimensions)
        result = checkAnswer("{}.ans".format(testInput), mySolution)
        print("{}: {}".format(testInput, result))

    endTime = time()
    print("\nTime: {}".format(endTime - startTime))

    return;

if __name__ == "__main__":
    main()
