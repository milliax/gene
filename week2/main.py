dept = ['工工系', '運管系', '管科系', '資財系', '資管所', 
'科管所', '經管所', 'GMBA', 'EMBA']

new_dept = []

for e in dept:
    if(e.endswith("系")):
        new_dept.append(f"{e}(所)")
    elif(e.endswith("MBA")):
        new_dept.append(f"{e}學程")
    else:
        new_dept.append(e)

for i in range(len(new_dept)):
    print(f"{i}: {new_dept[i]}")