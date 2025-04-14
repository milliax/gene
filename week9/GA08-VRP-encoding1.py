# 答案 x = [ [0, 8, 6, 2, 5, 0],
#            [0, 7, 1, 4, 3, 0],
#            [0, 9, 10, 16, 14, 0],
#            [0, 12, 11, 15, 13, 0]
# 最佳解 y = 1552 (目前程式參數設定可得最佳解1552)

import numpy as np
# import math


# ==== 參數設定(與問題相關) ====

NUM_CITY = 16            # 城市個數    # ==== Step 1-1. 設定城市個數 ====

cost = [                              # ==== Step 1-2. 設定cost ====
        [ 0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662 ],
        [ 548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210 ],
        [ 776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754 ],
        [ 696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358 ],
        [ 582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244 ],
        [ 274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708 ],
        [ 502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480 ],
        [ 194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856 ],
        [ 308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514 ],
        [ 194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468 ],
        [ 536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354 ],
        [ 502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844 ],
        [ 388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730 ],
        [ 354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536 ],
        [ 468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194 ],
        [ 776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798 ],
        [ 662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0 ],
    ]

NUM_VEHICLE = 4                        # ==== Step 1-3. 設定車輛個數 ====

# ==== 參數設定(與演算法相關) ====        # ==== Step 4. 調參數 ====

NUM_ITERATION = 500			# 世代數(迴圈數)

NUM_CHROME = 100				# 染色體個數#
NUM_BIT = NUM_CITY + NUM_VEHICLE - 1			# 染色體長度    # ==== Step 1-4. 設定bit個數 ====

Pc = 0.9    					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.1   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)

NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # 突變的次數

np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ==== 
def initPop():             # 初始化群體
    p = []
    
    for i in range(NUM_CHROME) :        # ==== Step 2-1. 產生1~16的隨機排列 ====
        a = list(np.random.permutation(range(1, NUM_CITY+1))) # 產生 1, ..., NUM_CITY 的排列
        for j in range(NUM_VEHICLE-1):  # ==== Step 2-2. 產生用三個0插入到這個1~16的隨機排列 ====
            a.insert(np.random.randint(len(a)+1), 0)
        p.append(a)
        
        
    return p

def fitFunc(x):            # 適應度函數    # ==== Step 3. 改適應度函數 ====
    v = []      # 記所有車的路徑    [ [車1的路], [車2的路], ... ]
    v_dist = [] # 記所有車的路徑距離 [ 車1的路長度, 車2的路長度, ... ]
    
    route = []  # 記目前考慮的車的路 [ 經過城市1, 經過城市2, ... ]
    dist = 0    # 記目前考慮的車的路的長度
    
    pre_city = 0    # 車子前一個city所在位置為 depot (0)
    
    for i in range(NUM_BIT):
        if x[i] == 0 :  # 當 code 是 0 時(表這台車回到dept，此時要結算這台車的路和長度)
            v.append(route) # 把目前的路加入到 v
            v_dist.append(dist + cost[pre_city][0]) # 把目前路長度加上回到dept的長度，把此長度加到 v_dist
            
            route = []      # 清空route內經過的城市
            dist = 0        # 長度設為初始0
            pre_city = 0    # 車子前一個city所在位置在 dept (0)
            continue        # 跳執行 for loop 的下一迴圈
        
        route.append(x[i])  # 把 code 所代表的城市加入到目前車的路route
        dist += cost[pre_city][x[i]]    # 把前一城市至目前 code 所代表城市的距離加到目前車的路長度 dist
        pre_city = x[i]     # 車子前一個city所在位置變成是 code 所代表城市

    if route != [] :        # 若最後一台車還沒考慮的話
        v.append(route)     # 把目前的路加入到 v
        v_dist.append(dist + cost[pre_city][0]) # 把目前路長度加上回到dept的長度，把此長度加到 v_dist
    
    return -max(v_dist)           # 因為是最小化問題

def evaluatePop(p):        # 評估群體之適應度
    return [fitFunc(p[i]) for i in range(len(p))]

def selection(p, p_fit):   # 用二元競爭式選擇法來挑父母
	a = []

	for i in range(NUM_PARENT):
		[j, k] = np.random.choice(NUM_CHROME, 2, replace=False)  # 任選兩個index
		if p_fit[j] > p_fit[k] :                      # 擇優
			a.append(p[j].copy())
		else:
			a.append(p[k].copy())

	return a

def crossover_uniform(p):           # 用均勻交配來繁衍子代 (new)
	a = []

	for i in range(NUM_CROSSOVER) :
		mask = np.random.randint(2, size=NUM_BIT)

		[j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index
       
		child1, child2 = p[j].copy(), p[k].copy()
		remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市
       
		for m in range(NUM_BIT):
			if mask[m] == 1 :
				remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
				remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]
		
		t = 0
		for m in range(NUM_BIT):
			if mask[m] == 0 :
				child1[m] = remain2[t]
				child2[m] = remain1[t]
				t += 1
		
		a.append(child1)
		a.append(child2)

	return a

def mutation(p):	           # 突變
	for _ in range(NUM_MUTATION) :
		row = np.random.randint(NUM_CROSSOVER_2)  # 任選一個染色體
		[j, k] = np.random.choice(NUM_BIT, 2, replace=False)  # 任選兩個基因
      
		p[row][j], p[row][k] = p[row][k], p[row][j]       # 此染色體的兩基因互換

def sortChrome(a, a_fit):	    # a的根據a_fit由大排到小
    a_index = range(len(a))                         # 產生 0, 1, 2, ..., |a|-1 的 list
    
    a_fit, a_index = zip(*sorted(zip(a_fit,a_index), reverse=True)) # a_index 根據 a_fit 的大小由大到小連動的排序
   
    return [a[i] for i in a_index], a_fit           # 根據 a_index 的次序來回傳 a，並把對應的 fit 回傳

def replace(p, p_fit, a, a_fit):            # 適者生存
    b = np.concatenate((p,a), axis=0)               # 把本代 p 和子代 a 合併成 b
    b_fit = p_fit + a_fit                           # 把上述兩代的 fitness 合併成 b_fit
    
    b, b_fit = sortChrome(b, b_fit)                 # b 和 b_fit 連動的排序
    
    return b[:NUM_CHROME], list(b_fit[:NUM_CHROME]) # 回傳 NUM_CHROME 個為新的一個世代


# ==== 主程式 ====

pop = initPop()             # 初始化 pop
pop_fit = evaluatePop(pop)  # 算 pop 的 fit

best_outputs = []                           # 用此變數來紀錄每一個迴圈的最佳解 (new)
best_outputs.append(np.max(pop_fit))        # 存下初始群體的最佳解

mean_outputs = []                           # 用此變數來紀錄每一個迴圈的平均解 (new)
mean_outputs.append(np.average(pop_fit))        # 存下初始群體的最佳解

for i in range(NUM_ITERATION) :
    parent = selection(pop, pop_fit)            # 挑父母
    offspring = crossover_uniform(parent)       # 均勻交配
    mutation(offspring)                         # 突變
    offspring_fit = evaluatePop(offspring)      # 算子代的 fit
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # 取代

    best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解

    if i != NUM_ITERATION-1 :
        print('iteration %d: y = %d'	%(i, -pop_fit[0]))     # fit 改負的
    else:
        print('iteration %d: x = %s, y = %d'	%(i, pop[0], -pop_fit[0]))     # fit 改負的

# 畫圖
import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.plot(mean_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()