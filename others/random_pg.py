import random

num = random.randrange(1, 11)  # 1부터 10까지!! 의 랜덤 숫자값이 num값이 되는 거임
print("randrange(1,11) 값 : ", num)

choices = "aBcD1543"
lists = ["left", "right"]
Choose = random.choice(choices)
print("choice(choices) 값 : ", Choose, "입니다.")
print("choice(lists) 값 : ", lists, "입니다.")
