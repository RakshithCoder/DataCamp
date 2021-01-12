# Python
## Why learn Python?
* Easy to learn, because of easy syntax
* Many libraries which makes work easy because you don't have to re-write them yourself.
* Web dev
* Game dev
* App dev
* AI
* Data Science
* ML

## Two ways to run python:
* Interactive Shell
    * Run each command and recieve the output right after
* Python Script
    * Write all the code, name the file and give it a .py extension.
    * After you run, it will display the output in the shell or terminal.

# Basic Concepts:
## Assuming that you're running all these in a python script 
1) Comments:
* It is used to make your code more descriptive
* It is used because you might forget why you did some things the way you did
* Two types of comments:
    * Single line comment(Short comment like why certain variable name is used)
    * Multi line comment(Really long descriptive concept like explaining how or why a certain formula is being used) 
```python
# This is a single line comment
'''
This is a multi line comment
Ik this looks completely useless rn
and you'd say you could just use many single line
comments to do the same 
but you'll figure out how useful it 
is in the long run 
'''
```
2) Exponent:
```python
# ** is used for exponentiation
print(2**3)
# 2 power 3 is 8
```

3) Modulo:
```python
# % is used for modulo
print(3%1)
# 3 mod 1 is 3
```

4) Printing to the terminal:
```python
# print('Hello reader!')
```

5) Variables:
* containers to store data
* why would you store data?- So that you can use it again and again
```python
# integers
int_data = 5

# float
float_data = 5.0

# strings
string_data = 'hello'
string_data_one = "hello"

# boolean
is_true = True
```

6) Type of a variable:
* Sometimes you'd want to know what type the variable is, so you'd use type
```python
print(type(int_data))
# <class 'int'> is the output
```

7) Type Conversion(Casting):
* Knowing the type is good, but it won't be of any help if you're not able to progress with your coding when there are type mismatches.
* That's why we need type casting or conversion to get our data into the desired format
```python
string_data = '10'
int_data = int(string_data) # type casting
'''
other type castings possible
str()
list()
dict()
float()
bool()
```

8) String Concatenation:
* Sometimes you'd want to embedded some numbers inside string, you can't do that directly since integers and strings are different types completely.
* You could typecast first store it in a variable and then concatenate those variables in that string or just concatenate and type cast on the fly
```python
savings = 100
result = 100 * 1.10 ** 7

print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")
# I started with $100 and now have $194.87171000000012. Awesome!
```

# Lists:
## Why lists?
All those other datatypes can only store a single value.
With Lists you can store a collection of data(strings, integers, floats and also other lists). For eg: It could be a list of all people in the house

## Concepts:
1) List creation:
```python
# list creation
family_members = ['Appa','Amma','Muchla','Nanu']
```

2) Empty list creation:
```python
# empty list creation
family = []
```

3) Printing type of a list:
```python
print(type(family_members))
# <class 'float'>
```

4) List inside a list(sub-list):
```python
family=[["Dad",185],"Mom",200]]
```

5) Indexing a list:
* You index a list to get a specific element from a list
* Indexing starts from 0
* If you supply an invalid index, an **IndexError: list index out of range** is returned.
```python
family_members = ['Appa','Amma','Muchla','Nanu']
print(family_members[0])
# Appa

# Reverse indexing in a list
print(family_members[-1])
# Nanu

print(family_members[4])
# IndexError: list index out of range
```

6) Slicing a list:
* You'd slice a list to obtain a subset of a list:
* Say, you have a list of 5 people, you are only interested in the first 2 ppl. Then you'd use slicing.
```python
ppl=['James','John','Mary','Kuri','Mari']
print(ppl[0:2])
# James, John

# printing the full list
print(ppl[0:])

# Slicing from back of the list
print(ppl[-1:-3])
# Mari, Kuri, Mary 
```







