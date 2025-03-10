uinput = []

for i in range(5):
    t = input("")
    uinput.append(t)


open("output.txt", "w").write("\n".join(uinput))