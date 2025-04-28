# 答案 y < 800
import matplotlib.pyplot
import numpy as np
# import math

import csv

def load_data_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    mOrder = [[int(row[i]) for i in range(0, len(row), 2)] for row in data]  # Odd columns
    pTime = [[int(row[i]) for i in range(1, len(row), 2)] for row in data]  # Even columns

    return pTime, mOrder

# ==== 參數設定(與問題相關) ====

NUM_JOB = 20            # 工件個數   # === Step 1. 設定成 3 個工件 ===
NUM_MACHINE = 15                     # === Step 1. 設定 3 個機台

# === Step 2. 設定 processing time 矩陣(row: job ID; col: 對應其加工機台所用的加工時間) ===
# pTime = [
#     [3, 2, 2],
#     [2, 1, 4],
#     [4, 3, 0]
# ]

# === Step 2. 設定 job 的機台加工順序 ===
# mOrder = [
#     [0, 1, 2],
#     [0, 2, 1],
#     [1, 2, 0]
# ]

pTime, mOrder = load_data_from_csv('jobshop_data.csv')

print("pTime:")
print(pTime)
print("mOrder:")
print(mOrder)

# ==== 參數設定(與演算法相關) ====

NUM_ITERATION = 2000			# 世代數(迴圈數)

NUM_CHROME = 100				# 染色體個數
NUM_BIT = NUM_JOB * NUM_MACHINE		   # 染色體長度 # === Step 3-1. 編碼是 000111222 的排列 ===

Pc = 0.85    					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.03   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)

NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
# 突變的次數# === Step 3-2. NUM_BIT 要修改成 3 x 3 ===
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)

np.random.seed(1000000)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ====    # === Step 5. 設定適應度函數 ===


def initPop():             # 初始化群體
    p = []

    # === 編碼 000111222 的排列  ===
    for i in range(NUM_CHROME):
        a = []
        for j in range(NUM_JOB):
            for k in range(NUM_MACHINE):
                a.append(j)
        np.random.shuffle(a)

        p.append(a)

    return p


def fitFunc(x):            # 適應度函數
    # S[i][j] = Starting time of job i at machine j
    S = np.zeros((NUM_JOB, NUM_MACHINE))
    # C[i][j] = Completion time of job i at machine j
    C = np.zeros((NUM_JOB, NUM_MACHINE))

    # B[j] = Available time of machine j
    B = np.zeros(NUM_MACHINE, dtype=int)

    # opJob[i] = current operation ID of job i
    opJob = np.zeros(NUM_JOB, dtype=int)

    for i in range(NUM_BIT):
        m = mOrder[x[i]][opJob[x[i]]]
        if opJob[x[i]] != 0:
            S[x[i]][m] = max([B[m], C[x[i]][mOrder[x[i]][opJob[x[i]]-1]]])
        else:
            S[x[i]][m] = B[m]

        C[x[i]][m] = B[m] = S[x[i]][m] + pTime[x[i]][opJob[x[i]]]

        opJob[x[i]] += 1

    return -max(B)           # 因為是最小化問題


def evaluatePop(p):        # 評估群體之適應度
    return [fitFunc(p[i]) for i in range(len(p))]


def selection(p, p_fit):   # 用二元競爭式選擇法來挑父母
    a = []

    for i in range(NUM_PARENT):
        [j, k] = np.random.choice(NUM_CHROME, 2, replace=False)  # 任選兩個index
        if p_fit[j] > p_fit[k]:                      # 擇優
            a.append(p[j].copy())
        else:
            a.append(p[k].copy())

    return a


def crossover_one_point(p):           # 用單點交配來繁衍子代 (new)
    a = []

    for i in range(NUM_CROSSOVER):
        c = np.random.randint(1, NUM_BIT)      		  # 隨機找出單點(不包含0)
        [j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index

        child1, child2 = p[j].copy(), p[k].copy()
        remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市

        for m in range(NUM_BIT):
            if m < c:
                remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
                remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]

        t = 0
        for m in range(NUM_BIT):
            if m >= c:
                child1[m] = remain2[t]
                child2[m] = remain1[t]
                t += 1

        a.append(child1)
        a.append(child2)

    return a


def crossover_uniform(p):           # 用均勻交配來繁衍子代 (new)
    a = []

    for i in range(NUM_CROSSOVER):
        mask = np.random.randint(2, size=NUM_BIT)
        [j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index

        child1, child2 = p[j].copy(), p[k].copy()
        remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市

        for m in range(NUM_BIT):
            if mask[m] == 1:
                remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
                remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]

        t = 0
        for m in range(NUM_BIT):
            if mask[m] == 0:
                child1[m] = remain2[t]
                child2[m] = remain1[t]
                t += 1

        a.append(child1)
        a.append(child2)

    return a


def mutation(p):	           # 突變
    for _ in range(NUM_MUTATION):
        row = np.random.randint(NUM_CROSSOVER_2)  # 任選一個染色體
        [j, k] = np.random.choice(NUM_BIT, 2, replace=False)  # 任選兩個基因

        p[row][j], p[row][k] = p[row][k], p[row][j]       # 此染色體的兩基因互換


def sortChrome(a, a_fit):	    # a的根據a_fit由大排到小
    # 產生 0, 1, 2, ..., |a|-1 的 list
    a_index = range(len(a))

    # a_index 根據 a_fit 的大小由大到小連動的排序
    a_fit, a_index = zip(*sorted(zip(a_fit, a_index), reverse=True))

    # 根據 a_index 的次序來回傳 a，並把對應的 fit 回傳
    return [a[i] for i in a_index], a_fit


def replace(p, p_fit, a, a_fit):            # 適者生存
    b = np.concatenate((p, a), axis=0)               # 把本代 p 和子代 a 合併成 b
    b_fit = p_fit + a_fit                           # 把上述兩代的 fitness 合併成 b_fit

    b, b_fit = sortChrome(b, b_fit)                 # b 和 b_fit 連動的排序

    return b[:NUM_CHROME], list(b_fit[:NUM_CHROME])  # 回傳 NUM_CHROME 個為新的一個世代


# ==== 主程式 ====

pop = initPop()             # 初始化 pop
pop_fit = evaluatePop(pop)  # 算 pop 的 fit

best_outputs = []                           # 用此變數來紀錄每一個迴圈的最佳解 (new)
best_outputs.append(np.max(pop_fit))        # 存下初始群體的最佳解

mean_outputs = []                           # 用此變數來紀錄每一個迴圈的平均解 (new)
mean_outputs.append(np.average(pop_fit))        # 存下初始群體的最佳解

for i in range(NUM_ITERATION):
    parent = selection(pop, pop_fit)            # 挑父母
    offspring = crossover_one_point(parent)     # 單點交配
    mutation(offspring)                         # 突變
    offspring_fit = evaluatePop(offspring)      # 算子代的 fit
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # 取代

    best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解

    print('iteration %d: x = %s, y = %d' %
          (i, pop[0], -pop_fit[0]))     # fit 改負的

# 畫圖
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.plot(mean_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
