import math

x = [1, 2, 3, 4]
x.append(5)
print(len(x))
print(x[1])
print(x[-1])
print(x[0:3])
print(x[:3])
print(x[1:])
# This is a comment
zeros = [0, 0, 0]
zeros_one = zeros[:]
zeros_one.append(1)
print(zeros, id(zeros))
print(zeros_one, id(zeros_one))
print(type(zeros))
print(10/3)
print(10//3)
print(10 % 3)
print(10**3)
x = 1
x *= 2
# python built in function
# python 3 math module

# x = input("x: ")
# int(x)
# float(x)
# str(x)
# bool(x)  # Falsy
# y = int(x)+1

age = 22
if age >= 18:
    print("adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

x = 4
if x > 1:
    pass
else:
    pass

message = "eligible" if age >= 18 else "not eligible"

for x in "Jaspar":
    print(x)

for x in [1, 2, 3, 4, 5]:
    print(x)

for x in range(5):
    print(x)
print("")
for x in range(5, 10):
    print(x)
print("")
for x in range(2, 10, 2):
    print(x)
print("")
print(range(5))

# range() takes up very little memory
names = ["AJaspar", "Travis"]
for name in names:
    if name.startswith("J"):
        print("Found")
        break
else:
    print("Not found")


# def increment(number: int, by: int = 1) -> tuple:  # default values
#     return (number, number + by)


def increment(number, by):  # default values
    return number + by


# print(increment(2, by=3))  # keyword argument
print(increment(2, 3))  # keyword argument


def multiply(*list):  # args, list is a tuple
    print(list)


multiply(2, 3, 4, 5)


def save_user(**user):
    print(user)


save_user(id=1, name="admin")

message = "a"


def greet():
    message = "b"


greet()
print(message)

# F9 adds breakpoint
# F5 starts debuging session
# F10 steps forward
# F11 goes into function
# shift F11 steps out of function

# alt up down moves a line
# shift+alt+down duplicate lines
# shift F6 renames all of a variables

coordinates = (1, 2, 3)
x, y, z = coordinates

print(3*int(False))

x = False

