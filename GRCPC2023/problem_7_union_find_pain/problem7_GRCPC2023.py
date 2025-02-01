from time import time

def loadInput(txtFileName: str) -> tuple:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()
    txtInput = text.split("\n")

    (n, m) = txtInput[0].split(" ")
    (n, m) = (int(n), int(m))
    ls = [None] * m
    for k in range(m):
        (node1, node2) = txtInput[k + 1].split(" ")
        (node1, node2) = (int(node1), int(node2))
        if node2 > 0:
            ls[k] = (node1, node2, 1)
        else:
            ls[k] = (node1, -node2, -1)
            # u, v, weight

    return (n, m, ls);

def checkAnswer(txtFileName: str, answer: int) -> bool:
    with open(txtFileName, mode = "r", encoding = "utf8") as file:
        text = file.read()

    passOrFail = True
    if text != answer:
        passOrFail = False

    return passOrFail;

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)
        """ Union by *rank* ensures that the smaller tree is
        always added under the root of the larger tree! """
        self.weight = [1] * (n + 1)

        return;

    def find(self, u: int) -> int:
        if self.parent[u] != u:
            originalParent = self.parent[u]
            self.parent[u] = self.find(self.parent[u])
            self.weight[u] *= self.weight[originalParent]

        return self.parent[u];

    def union(self, u: int, v: int, weight: int):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            # Union by rank => find operation becomes O(log(n)) time!!!
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
                self.weight[root_v] = self.weight[u] * weight * self.weight[v]
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
                self.weight[root_u] = self.weight[v] * weight * self.weight[u]
            else:
                self.parent[root_v] = root_u
                self.weight[root_v] = self.weight[u] * weight * self.weight[v]
                self.rank[root_u] += 1

        return;

    def connected(self, u: int, v: int) -> int:
        if self.find(u) == self.find(v):
            return self.weight[u] * self.weight[v];

        return 0;

def solution(n_m_ls: tuple) -> str:
    n, m, ls = n_m_ls
    solution = ""

    uf = UnionFind(n)
    for (u, v, weight) in ls:
        alreadyConnectedWeight = uf.connected(u, v)
        if alreadyConnectedWeight == weight:
            solution += "E\n"
        elif alreadyConnectedWeight == 0:
            solution += "N\n"
        else:
            solution += "C\n"
            break;

        uf.union(u, v, weight)

    return solution;

def main():
    inputLen = 20
    inputLs = []
    inputLs = [None] * inputLen
    for i in range(inputLen):
        i_str = str(i + 1)
        inputLs[i] = "input_{}".format(i_str if (len(i_str) >= 2) else "0{}".format(i_str))

    for testInput in inputLs:
        startTime = time()

        n_m_ls = loadInput("{}.in".format(testInput))
        mySolution = solution(n_m_ls)
        result = checkAnswer("{}.ans".format(testInput), mySolution)
        print("{}: {}".format(testInput, result))

        endTime = time()
        print("Time: {}\n".format(endTime - startTime))

    return;

if __name__ == "__main__":
    main()
