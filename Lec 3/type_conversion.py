x = 10          #integer
y = 10.2        #float
z = "Hello"     #string

print(x, type(x))
print(y, type(y))
print(z, type(z))

#implicit type conversion
x = x*y


print(x,type(x))
print(y,type(y))
print(z,type(z))

#exploit type conversion
age = input("What is your age? ")
print(age,type(age))
age = int(age)
print(age,type(age))

#name
name = input("What is your name? ")
print(name,type(name))
name = int(name)
print(name,type(name))

