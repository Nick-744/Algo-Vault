'''
MinHeap class for storing key-value items
-----------------------------------------
Operations on MinHeap
---------------------
1) getMin(): 
    It returns the root element of min Heap. 
    Time Complexity is O(1).

2) extractMin(): 
    Removes the minimum element from MinHeap. 
    Time Complexity is O(logn) 
    as this operation needs to maintain the heap 
    property by calling heapify() after removing root.

3) insert(): 
    Inserting a new key takes O(logn) time. 
    We add a new key at the end of the tree. 
    If new key is greater than its parent, 
        then we don’t need to do anything. 
    Otherwise, we need to traverse up 
        to fix the violated heap property (up-heap).

4) decreaseKey(): 
    Decreases value of key. 
    Time complexity is O(logn). 
     
     [5]
        \
        [10]    -> e.g 9 or 1
        /   \
      [15]  [16]  

    If the decreased key value of a node 
        is greater than the parent of the node, 
        then we don’t need to do anything. 
    Otherwise, we need to traverse up 
        to fix the violated heap property (up-heap).

5) increaseKey(): 
    Increases value of key. 
    Time complexity is O(logn). 

     [5]
        \
        [10]    -> e.g 12 or 20
        /   \
      [15]  [16]  

    If the increased key value of a node 
        is less than the children of the node, 
        then we don’t need to do anything. 
    Otherwise, we need to traverse down 
        to fix the violated heap property (down-heap).

6) changeKey(): 
    Decreases or Increases value of key. 

7) delete(): 
    Deleting a key takes O(logn) time. 
    Replace the key to be deleted with 
        minimum infinite by calling decreaseKey(). 
    then as the minus infinite is root, 
        call extractMin() to remove the key. 

8) isInMinHeap(v)
    check if exists an item with key = v
    Time complexity is O(1) 
'''
class MinHeap():
    def __init__(self, arr):
        self.array = []         # tuple (key, value)
        self.pos = {}           # position of key in array
        self.size = len(arr)    # number of elements in min heap

        for i, item in enumerate(arr):
            self.array.append((item[0], item[1]))
            self.pos[item[0]] = i

        for i in range(self.size // 2, -1, -1):
            self.heapify(i)


    # display items of heap
    def display(self):
        print('array =', end=' ')
        for i in range(self.size):
            print(f'({self.array[i][0]} : {self.array[i][1]})', end=' ')
        print()


    # is heap empty ?
    def isEmpty(self):
        return self.size == 0


    # i is the array index
    # modify array so that i roots a heap, down-heap
    def heapify(self, i):

        smallest = i
        le = 2 * i + 1
        ri = 2 * i + 2

        if le < self.size and self.array[le][1] < self.array[smallest][1]:
            smallest = le
        if ri < self.size and self.array[ri][1] < self.array[smallest][1]:
            smallest = ri

        if smallest != i:
            # update pos
            self.pos[self.array[smallest][0]] = i
            self.pos[self.array[i][0]] = smallest

            # swap
            self.array[smallest], self.array[i] = \
                self.array[i], self.array[smallest]

            self.heapify(smallest)


    # return the min element of the heap
    def getMin(self):
        if self.size == 0:
            return None

        return self.array[0]


    # return and remove the min element of the heap
    def extractMin(self):
        if self.size == 0:
            return None

        root = self.array[0]
        lastNode = self.array[self.size - 1]

        self.array[0] = lastNode

        # update pos
        self.pos[lastNode[0]] = 0
        del self.pos[root[0]]

        self.size -= 1
        self.heapify(0)

        return root


    # item (key, value) to insert
    # modify array to include item
    def insert(self, item):
        # insert an item at the end with big value
        if self.size < len(self.array):
            self.array[self.size] = (item[0], 10**80)
        else:
            self.array.append((item[0], 10**80))

        self.pos[item[0]] = self.size
        self.size += 1
        self.decreaseKey(item)


    # decrease value of item (key, value)
    # with value smaller than current
    def decreaseKey(self, item):
        i = self.pos[item[0]]
        val = item[1]

        # new value must be smaller than current
        if self.array[i][1] <= val:
            return

        self.array[i] = item

        # check if is smaller than parent
        p = (i - 1) // 2
        while p >= 0 and self.array[i][1] < self.array[p][1]:
            # update pos
            self.pos[self.array[i][0]] = p
            self.pos[self.array[p][0]] = i

            # swap
            self.array[p], self.array[i] = self.array[i], self.array[p]

            i = p
            p = (i - 1) // 2

    # ---
    # increase value of item (key, value),
    # with value greater than current
    def increaseKey(self, item):
        i = self.pos[item[0]]
        val = item[1]

        # new value must be greater than current
        if self.array[i][1] >= val:
            return

        self.array[i] = item

        # check children
        self.heapify(i)


    # change value of item (key, value),
    # call decreaseKey or increaseKey
    def changeKey(self, item):
        i = self.pos[item[0]]
        val = item[1]

        if val < self.array[i][1]:
            self.decreaseKey(item)

        if val > self.array[i][1]:
            self.increaseKey(item)


    # check if exists an item with key = v
    def isInMinHeap(self, v):
        if self.pos.get(v) == None:
            return False
        if self.pos[v] < self.size:
            return True
        return False


    # get the value of item with key = key
    def getValueMinHeap(self, key):
        if self.pos.get(key) == None:
            return None
        if self.pos[key] < self.size:
            return self.array[self.pos[key]][1]
        return None


    # delete item 
    # reduce value of item to minus infinite 
    # and call extractMin() 
    def deleteKey(self, item): 
        self.decreaseKey((item[0], float('-inf'))) 
        self.extractMin()


# --- test ---
if __name__ == '__main__':
    arr = [('a', 50), ('b', 30), ('c', 10), ('d', 20), ('e', 40), ('f', 80)]

    h = MinHeap(arr)
    h.display()

    h.decreaseKey(('b', 2))
    h.display()
    h.increaseKey(('b', 30))
    h.display()

    h.extractMin()
    h.display()
    h.extractMin()
    h.display()

    h.insert(('x', 5))
    h.display()

    print('min=', h.getMin())

    h.deleteKey(('e', 0))
    h.display()

    # print(h.isInMinHeap('a'))
    # print(h.isInMinHeap('c'))
    # print(h.isInMinHeap('x'))

    # h2 = MinHeap([])
    # for i in range(1024, 0, -1):
    #     h2.insert((i, i**2))
    # print(h2.getMin())
