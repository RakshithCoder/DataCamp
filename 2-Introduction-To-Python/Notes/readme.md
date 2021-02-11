---
modified: 2021-01-30T22:13:44+05:30
---

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
family=[["Dad",185],["Mom",200]]
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

7) Lists of lists:
```python
x = [["a", "b", "c"],
     ["d", "e", "f"],
     ["g", "h", "i"]]
x[2][0]  # g
x[2][:2]  # g, h
```

8) Manipulating list elements:
* Sometimes you'd want to change only a specific element in a list
```python
my_list=[1,2,3]
my_list[2]=5
print(my_list)
# 1,2,5
```
* Changing an entire slice of elements
```python
my_list=[1,2,3]
my_list[0:2]=[2,4]
print(my_list)
# 2,4,3
```
* Adding an element:
```python
my_list=[1,2,3]
my_list+1.9
print(my_list)
# [1,2,3,1.9]
```
* Adding a list:
```python
my_list=[1,2,3]
my_list+["me",1.9]
print(my_list)
# [1,2,3,me,1.9]
```
* Removing an element:
```python
my_list=[1,2,3]
del(my_list[1])
print(my_list)
# [1,3]
```

# Functions:
## Why functions?
Basically, we create functions for code reusability.
What I mean by that is write code once, which you know for sure that you'll use it many more times, so that you won't have to write it again and again.

## Concepts:
We could define our own function and use for the purpose stated. In this case since, we made our own function it is like a **white box**.
Or, in other cases, we could use(call) other functions created by others(built-in functions), that time it will be a **black box**
we won't know and most time we would not care how it's made, we'd only focus on using it in our application.

1) Type of:
* Sometimes we'd want to know the type of maybe a variable or a data structure, to help in debugging and understanding our code.
```python
a=5
print(type(a))
# <class 'int'>
```

2) Maximum element:
* Sometimes we'd want to know the maximum element in a list or a tuple.
```python
heights=[170,158,140]
print(max(heights))
# 170
```

3) Rounding an integer:
```python
num=1.68
print(round(num))  # 2
print(round(num,1))  # 1.7 (rounded to 1 decimal place)
```

4) Help:
* Help function is used to for us to get to know a function better(what arguments must be passed, what arguments are optional, what a function returns)
* After help is run, if something like this comes up,
complex(real[, imag]) then it means real is a required argument and imag is an optional argument.
```python
print(help(round))
```

5) int(),str() and all those which we saw earlier in Type casting are also functions

6) Length of a list:
* To find how many elements are present in a list
```python
some_list = [1,2,3,4,4,5,6,6,7,7,8,8,8]
print(len(some_list))
# 13
```

7) Sorting a data structure:
```python
age_list = [10,6,11,5]
sorted_list=sorted(age_list)
print(sorted_list, reverse=True)
# [11,10,6,5]
```

# Methods:
* Various data structures like str, list, etc are actually classes and they contain functions inside them. They are called methods.

## List methods:
1) Index:
* To get the index of a specific element in a list.
* **Note:** Index method is also present in strings
```python
my_list = [1,2,3,4,5]
print(my_list.index(2))
# 1
```

2) Count:
* To count the occurence of a specific element.
```python
my_list = [1,2,3,4,5]
print(my_list.count(2))
# 1
```

3) Append:
* To add an element to the end of a list.
```python
my_list = [1,2,3]
print(my_list.append(4))
# [1,2,3,4]
```

## String methods:
1) Capitalize:
* To capitalize the first element in the string.
```python
my_string = 'raks'
print(my_string.capitalize())
# 'Raks'
```

2) Replace:
* Replace a certain character/s with a different set.
```python
my_string = 'liz'
print(my_string.replace('z', "sa"))
# 'lisa'
```

3) Upper:
* Converts all the characters to uppercase.
```python
my_string = 'liz'
print(my_string.upper())
# 'LIZ'
```

# Packages:
## What are packages?
* It is directory of python **scripts.**
* Each script is called a **module.**
* These modules consists of **methods** designed to solve a particular problem.
* A package would look something like this...
    * package/
        * mod1.py
        * mod2.py

Eg: Numpy-to efficiently work with arrays, Matplotlib for data viz and scikit-learn for ml.

## How do use them?
* Not all of these packages are by default in python.
* You have to install them.

## How to install them?
* Steps:
1) Search for pip:
    http://pip.readthedocs.org/en/stable/installing/
2) Download pip:
    get-pip.py 
3) Open terminal:
    **python get-pip.py**

## Everything is setup now, note that this only has to be done once.After this, just install the specific package you need.

Eg: pip install numpy

## Import the installed package:
* To use a the package, it has to be imported in your script.
```python
import numpy
numpy.array([1,2,3])
```

## You could import the package and rename it for simplicity:
```python
import numpy as np
np.array([1,2,3])
```

## Specific import:
* Sometimes you would only want to import a specific module from a package.
```python
from numpy import array
array([1,2,3])
```

## Note:
* It is always best to not use specific import when the code is huge. Because somewhere in between you'd see the name of the imported module and get confused if it is from a specific package or a user-defined one.
```python
import numpy as np
np.array([1,2,3])
```
* This is way more specific, because we are explicitly specifiying from where we are using array.

# Numpy:
* Lists are good. But, as data gets bigger and bigger lists are more time consuming. So we use numpy(numeric python library) which is faster and more effecient than list.

## Installation:
* Numpy is not available by default. We have to install it. 
* **pip install numpy**

## Using it:
```python
import numpy as np
```

## How is it more efficient(Advantage)?
```python
# List example
height = [1.73, 1.68, 1.71, 1.89, 1.79]
weight = [65.4, 59.2, 63.6, 88.4, 68.7]
print(weight / height ** 2)
# TypeError: unsupported operand type(s) for **: 'list' and 'int'

# Numpy example:
import numpy as np
np_height = np.array(height)
print(np_height)
# array([ 1.73,  1.68,  1.71,  1.89,  1.79])

np_weight = np.array(weight)
print(np_weight)
# array([ 65.4,  59.2,  63.6,  88.4,  68.7])

bmi = np_weight / np_height ** 2
print(bmi)
# array([ 21.852,  20.975,  21.75 ,  24.747,  21.441])
```

## Disadvantage:
* All the elements in the a numpy array must be of a single type.
```python
np.array([1.0, "is", True])
# array(['1.0', 'is', 'True'], dtype='<U32')
# Everything is converted to a string in this case.
# It is called "type coercion"
```

## Other behaviours:
```python
python_list = [1, 2, 3]
numpy_array = np.array([1, 2, 3])
# or numpy_array = np.array(python_list)

python_list + python_list
# [1, 2, 3, 1, 2, 3]

numpy_array + numpy_array
# [2, 4, 6]

python_list * 3
# [1, 2, 3, 1, 2 ,3, 1, 2 ,3]

numpy_array *3
# [3, 6, 9]
```

## Note:
* Indexing is similar to lists.
```python
bmi[bmi>23]
# array([24.54])
```

## Example of boolean indexing:
```python
import numpy as np

# Calculate the BMI: bmi
np_height_m = np.array(height_in) * 0.0254
np_weight_kg = np.array(weight_lb) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light = np.array(bmi < 21)

# Print out light
print(light)

print(type(light))
# numpy.ndarray

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])
```

## Attributes of numpy:
* shape:
    ```python
    print(light.shape)
    ```

## Numpy 2D array:
```python
np_2d = np.array([[1.73,1.68,1.71,1.89,1.79],
                 [65.4,59.2,63.6,88.4,68.7]])

print(np_2d[0][1])
# both are the same 
print(np_2d[0,2])  # row,col

# suppose you want height and weight of 2nd and 3rd person.
print(np_2d[:,1:3])

# suppose you want only weight of all people
print(np_2d[1,:])
```

## Other operations:
```python
import numpy as np
np_mat = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])
np_mat * 2
'''
np.array([[1, 2],
         [3, 4],
         [5, 6]])
'''
np_mat + np.array([10, 10])
'''
array([[11, 12],
       [13, 14],
       [15, 16]])
'''
np_mat + np_mat
'''
array([[ 2,  4],
       [ 6,  8],
       [10, 12]])
'''
```

## Numpy basic statistics:
* Mean:
```python
import numpy as np
np.mean(np_city[:,0])  # considering all rows, and the first column, then taking the mean of those.
```
* Median
```python
np.median(np_city[:,0])
```
* Standard deviation:
```python
np.std(np_city[:,0])
```
*Correlation
```python
np.corrcoef(np_city[:,0], np_city[:,0])  # To find correlation b/w first and second column.
```

## Note:
* Basically why numpy is faster than list?
A list is a compound datatype, it might contain other data types.
But a numpy array would contain only a single datatype throughout.

