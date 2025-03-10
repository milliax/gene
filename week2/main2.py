uinput = input("")

toOutput = ""

for c in uinput:
    if(c.islower()):
        toOutput+=c.upper()
    elif(c.isupper()):
        toOutput+=c.lower()
    else:
        toOutput+=c

print(toOutput)