import random

def random10int():
    return [random.randint(0, 50) for _ in range(10)]

def squareSum(params: list):
    return sum([x**2 for x in params])


randomInts = random10int()

print(randomInts)
print(squareSum(randomInts))