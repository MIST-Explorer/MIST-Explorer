import numpy as np

base =np.array([[0,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]])

super =np.array([[1,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0]])

in_bounds = lambda arr, i, j: i >= 0 and i < len(arr) and j >= 0 and j < len(arr)

def pprint(arr):
    
    for row in (arr):
        print(row)

def bleed_sub(arr):

    copy = []

    for row in (arr):
        sub = []
        for col in ((row)):
            sub.append(col)
        copy.append(sub)

    for row in range(len(copy)):
        # print(len(arr))
        for col in range((len(copy[0]))):
            # print(len(arr))
            # print(row, col)

            val = arr[row][col]
            if val == 0: 
                continue

            for i, j in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                # print(row, col, i, j, in_bounds(arr, row + i, col + j))
                if in_bounds(arr, row + i, col + j):
                    copy[row + i][col + j] = (val + arr[row + i][col + j]) / 2
    return copy
                
def bleed(arr, i):
    for count in range(i):
        arr = bleed_sub(arr)

    return arr

super_bled = bleed(super, 4)
base_bled = bleed(base, 4)



print(np.sum(np.abs(np.subtract(super_bled, base_bled))))
print(np.sum(np.abs(np.subtract(super_bled[0:9], base_bled[1:10]))))
print(np.sum(np.abs(np.subtract(super_bled[0:8], base_bled[1:10][1:10]))))


def cut_cols(arr, n, is_front):
    copy = []

    for row in arr:
        sub = []
        for col,i in enumerate(row):
            if  n > i:
                sub.append(col)
        copy.append(sub)

pprint(cut_cols(base_bled[1:10], 1, True))
# i, j = (0, 1)
# row = 0
# col = 0

# print(super[row + i][col + j])
# super[row + i][col + j] = (1 + super[row + i][col + j]) / 2
# pprint(super)