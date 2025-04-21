# 答案 y = 7038
import matplotlib.pyplot
import numpy as np
# import math


# ==== 參數設定(與問題相關) ====

NUM_JOB = 11            # 工件個數   # === Step 1. NUM_CITY 改成 NUM_JOB，設定成 11 個工件 ===
NUM_MACHINE = 5                     # === Step 1. 設定 5 個機台

# === Step 2. 改 cost 矩陣為 processing time 矩陣 ===
pTime = [
    [375,  12, 142, 245, 412],
    [632, 452, 758, 278, 398],
    [12, 876, 124, 534, 765],
    [460, 542, 523, 120, 499],
    [528, 101, 789, 124, 999],
    [796, 245, 632, 375, 123],
    [532, 230, 543, 896, 452],
    [14, 124, 214, 543, 785],
    [257, 527, 753, 210, 463],
    [896, 896, 214, 258, 259],
    [532, 302, 501, 765, 988]
]

# ==== 參數設定(與演算法相關) ====

NUM_ITERATION = 100			# 世代數(迴圈數)

NUM_CHROME = 100				# 染色體個數
NUM_BIT = NUM_JOB		   # 染色體長度# === Step 3. 編碼長度為 NUM_JOB ===

Pc = 0.9    					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.1   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)

NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # 突變的次數

np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ====    # === Step 4. 設定適應度函數 ===


def initPop():             # 初始化群體
    p = []

    for i in range(NUM_CHROME):
        p.append(np.random.permutation(NUM_BIT))  # 產生 0, 1, ..., NUM_BIT-1 的排列

    return p


def fitFunc(x):            # 適應度函數
    # S[i][j] = Starting time of job i at machine j
    S = np.zeros((NUM_JOB, NUM_MACHINE))
    # C[i][j] = Completion time of job i at machine j
    C = np.zeros((NUM_JOB, NUM_MACHINE))

    # Put code here

    return True  # Put code here


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
    offspring = crossover_uniform(parent)       # 均勻交配
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
