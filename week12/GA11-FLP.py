# 最佳解 y = 851700

import matplotlib.pyplot
import numpy as np

# ==== 參數設定(與問題相關) ====

NUM_PRODUCT = 8            # === Step 1. 設定成4個產品 ===
NUM_MACHINE = 15

productVol = [300, 400, 200, 500, 100, 150, 250, 450]

routes = [
    [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 8, 9, 10]],
    [[1, 2, 3, 5, 7, 8, 10, 11], [3, 4, 5, 6, 7, 8, 9, 10]],
    [[1, 4, 10, 11], [1, 4, 10, 12, 13, 14], [1, 4, 9, 12, 13, 14]],
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
    [[4, 5, 6, 7, 8, 14, 15], [4, 5, 6, 7, 12, 13, 14, 15],
        [4, 5, 7, 9, 10, 11, 12, 13, 14, 15]],
    [[2, 3, 4, 5, 6], [2, 3, 4, 5, 7, 8], [2, 3, 4, 5, 7, 10]],
    [[4, 5, 7, 8, 10, 11, 12], [4, 5, 8, 9, 10, 12, 14],
        [4, 5, 8, 10, 11, 12, 13, 15]],
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]]
]
route = [[[station - 1 for station in route]
          for route in product] for product in routes]

print(route)

dist = [
    [0, 1, 11, 4, 9, 2, 8, 5, 12, 7, 8, 3, 10, 4, 8, 1],
    [1, 0, 6, 1, 9, 10, 3, 11, 6, 8, 3, 4, 12, 5, 9, 11],
    [11, 6, 0, 2, 7, 12, 5, 9, 4, 3, 7, 6, 11, 7, 1, 5],
    [4, 1, 2, 0, 11, 3, 12, 6, 9, 2, 8, 6, 3, 13, 4, 9],
    [9, 9, 7, 11, 0, 2, 7, 3, 4, 10, 4, 9, 8, 1, 7, 5],
    [2, 10, 12, 3, 2, 0, 10, 12, 55, 3, 7, 3, 9, 10, 6, 8],
    [8, 3, 5, 12, 7, 10, 0, 1, 9, 13, 5, 7, 3, 9, 12, 4],
    [5, 11, 9, 6, 3, 12, 1, 0, 5, 8, 1, 10, 4, 12, 6, 3],
    [12, 6, 4, 9, 4, 55, 9, 5, 0, 9, 3, 8, 7, 12, 5, 7],
    [7, 8, 3, 2, 10, 3, 13, 8, 9, 0, 4, 9, 12, 3, 7, 6],
    [8, 3, 7, 8, 4, 7, 5, 1, 3, 4, 0, 7, 11, 2, 9, 8],
    [3, 4, 6, 6, 9, 3, 7, 10, 8, 9, 7, 0, 4, 8, 3, 9],
    [10, 12, 11, 3, 8, 9, 3, 4, 7, 12, 11, 4, 0, 1, 9, 6],
    [4, 5, 7, 13, 1, 10, 9, 12, 12, 3, 2, 8, 1, 0, 12, 11],
    [8, 9, 1, 4, 7, 6, 12, 6, 5, 7, 9, 3, 9, 12, 0, 10],
    [1, 11, 5, 9, 5, 8, 4, 3, 7, 6, 8, 9, 6, 11, 10, 0]
]

# ==== 參數設定(與演算法相關) ====

NUM_ITERATION = 500		# 世代數(迴圈數)

NUM_CHROME = 300				# 染色體個數
NUM_BIT = NUM_MACHINE + NUM_PRODUCT  # 染色體長度    # ==== Step 1. 設定bit個數 ====

Pc = 0.95    					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.2   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)

NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # 突變的次數

np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ====


def initPop():             # 初始化群體
    p1 = []
    p2 = []

    for i in range(NUM_CHROME):
        # ==== Step 2-1. 產生0~4的隨機排列 ====
        p1.append(np.random.permutation(NUM_MACHINE))

        # ==== Step 2-2. 產生 NUM_PRODUCT 個上下界的路徑選擇 ====
        p2.append([np.random.randint(len(route[j]))
                  for j in range(NUM_PRODUCT)])

    return p1, p2


def fitFunc(x1, x2):       # 適應度函數    # ==== Step 4. 改適應度函數 ====
    totalCost = 0

    for i in range(NUM_PRODUCT):
        myRoute = route[i][x2[i]]
        pre_j = myRoute[0]
        tmpDist = 0
        for j in range(1, len(myRoute)):
            tmpDist += dist[x1[pre_j]][x1[myRoute[j]]]
            pre_j = myRoute[j]
        totalCost += tmpDist * productVol[i]

    return -totalCost           # 因為是最小化問題


def evaluatePop(p1, p2):        # 評估群體之適應度
    return [fitFunc(p1[i], p2[i]) for i in range(len(p1))]


def selection(p1, p2, p_fit):   # 用二元競爭式選擇法來挑父母
    a1 = []
    a2 = []

    for i in range(NUM_PARENT):
        [j, k] = np.random.choice(NUM_CHROME, 2, replace=False)  # 任選兩個index
        if p_fit[j] > p_fit[k]:                      # 擇優
            a1.append(p1[j].copy())
            a2.append(p2[j].copy())
        else:
            a1.append(p1[k].copy())
            a2.append(p2[k].copy())

    return a1, a2


def crossover_uniform(p1, p2):           # 用均勻交配來繁衍子代
    a1 = []
    a2 = []

    for i in range(NUM_CROSSOVER):
        mask1 = np.random.randint(2, size=NUM_MACHINE)
        mask2 = np.random.randint(2, size=NUM_PRODUCT)
        [j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index

        child1, child2 = p1[j].copy(), p1[k].copy()
        remain1, remain2 = list(p1[j].copy()), list(
            p1[k].copy())     # 存還沒被用掉的城市

        for m in range(NUM_MACHINE):
            if mask1[m] == 1:
                remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
                remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]

        t = 0
        for m in range(NUM_MACHINE):
            if mask1[m] == 0:
                child1[m] = remain2[t]
                child2[m] = remain1[t]
                t += 1

        a1.append(child1)
        a1.append(child2)

        # === for p2 ===
        child3, child4 = p2[j].copy(), p2[k].copy()

        for m in range(NUM_PRODUCT):
            if mask2[m] == 1:
                child3[m] = p2[k][m]
                child4[m] = p2[j][m]

        a2.append(child3)
        a2.append(child4)

    return a1, a2


def mutation(p1, p2):	           # 突變
    for _ in range(NUM_MUTATION):
        row = np.random.randint(NUM_CROSSOVER_2)  # 任選一個染色體

        if np.random.randint(2) == 0:
            [j, k] = np.random.choice(NUM_MACHINE, 2, replace=False)  # 任選兩個基因

            p1[row][j], p1[row][k] = p1[row][k], p1[row][j]       # 此染色體的兩基因互換
        else:
            j = np.random.randint(NUM_PRODUCT)  # 任選1個基因
            p2[row][j] = np.random.randint(
                len(route[j]))  # ==== Step 3. 改好上下界 ====


def sortChrome(a1, a2, a_fit):	    # a的根據a_fit由大排到小
    # 產生 0, 1, 2, ..., |a|-1 的 list
    a_index = range(len(a1))

    # a_index 根據 a_fit 的大小由大到小連動的排序
    a_fit, a_index = zip(*sorted(zip(a_fit, a_index), reverse=True))

    # 根據 a_index 的次序來回傳 a，並把對應的 fit 回傳
    return [a1[i] for i in a_index], [a2[i] for i in a_index], a_fit


def replace(p1, p2, p_fit, a1, a2, a_fit):            # 適者生存
    b1 = np.concatenate((p1, a1), axis=0)               # 把本代 p 和子代 a 合併成 b
    b2 = np.concatenate((p2, a2), axis=0)               # 把本代 p 和子代 a 合併成 b
    b_fit = p_fit + a_fit                           # 把上述兩代的 fitness 合併成 b_fit

    b1, b2, b_fit = sortChrome(b1, b2, b_fit)                 # b 和 b_fit 連動的排序

    # 回傳 NUM_CHROME 個為新的一個世代
    return b1[:NUM_CHROME], b2[:NUM_CHROME], list(b_fit[:NUM_CHROME])


# ==== 主程式 ====

pop1, pop2 = initPop()             # 初始化 pop
pop_fit = evaluatePop(pop1, pop2)  # 算 pop 的 fit

best_outputs = []                           # 用此變數來紀錄每一個迴圈的最佳解 (new)
best_outputs.append(np.max(pop_fit))        # 存下初始群體的最佳解

mean_outputs = []                           # 用此變數來紀錄每一個迴圈的平均解 (new)
mean_outputs.append(np.average(pop_fit))        # 存下初始群體的最佳解

for i in range(NUM_ITERATION):
    parent1, parent2 = selection(pop1, pop2, pop_fit)            # 挑父母
    offspring1, offspring2 = crossover_uniform(parent1, parent2)       # 均勻交配
    mutation(offspring1, offspring2)                            # 突變
    offspring_fit = evaluatePop(offspring1, offspring2)      # 算子代的 fit
    pop1, pop2, pop_fit = replace(
        pop1, pop2, pop_fit, offspring1, offspring2, offspring_fit)    # 取代

    best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解

    if i != NUM_ITERATION-1:
        print('iteration %d: y = %d' % (i, -pop_fit[0]))     # fit 改負的
    else:
        print('iteration %d: x1 = %s, x2 = %s, y = %d' %
              (i, pop1[0], pop2[0], -pop_fit[0]))     # fit 改負的

# 畫圖
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.plot(mean_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
