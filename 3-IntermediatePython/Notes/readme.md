# Data Visualization:
## What is data visualization?
* Giving data some meaning through graphs.

## Why data visualization?
* You can better understand data and get insights into the data.

## Various viz ways:
* Matplotlib:
```python
import matplotlib.pyplot as plt
year = [...]
population = [...]
plt.plot(year, pop)  # plot doesn't show right after this, we need to call show
plt.show()
```
    ## Note:
        * plot() function tells what to plot and how to.
        * show() function actually displays it.

* Scatterplot:
```python
import matplotlib.pyplot as plt
year = [...]
population = [...]
plt.scatter(year, pop)
plt.show()
```

## Difference b/w scatter and plot:
scatter | plot
------- | -------
scatter only displays the data points doesn't connect it | plot not only displays all the data points, but also connects it. 

## Histogram:
* To check the distribution of the variables.
* The entire line can be divided into equal chunks called bins
* Then you add the variables to those bins.(bins=10 by default)
* For each bin, a bar is presented.
* You use matplotlib for building histograms
```python
import matplotlib.pyplot as plt
values = [0,0.6,1.4,1.6,2.2,2.5,2.6,3.2,3.5,3.9,4.2,6]
plt.hist(values, bins=3)
plt.show()
plt.clf()  # clear figure
```

## Choosing the right graph:
* You're a professor teaching Data Science with Python, and you want to visually assess if the grades on your exam follow a particular distribution. Which plot do you use? - Histogram
* You're a professor in Data Analytics with Python, and you want to visually assess if longer answers on exam questions lead to higher grades. Which plot do you use? - Scatter, if you use a line plot for this, the points will be all over the place.

## Customization:
* We can use a variety of things for customizing our graphs:
    * axis labels
    * title
    * ticks
```python
import matplotlib.pyplot as plt
year = [1950, 1951, 1952, ..., 2100] 
pop = [2.538, 2.57, 2.62, ..., 10.85]
# Add more data
year = [1800, 1850, 1900] + year
pop = [1.0, 1.262, 1.650] + pop
plt.plot(year, pop)
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('World Population Projections')
plt.yticks([0, 2, 4, 6, 8, 10], 
          ['0', '2B', '4B', '6B', '8B', '10B'])
plt.show()
```

## More customization:
* size as argument(s) for scatter plot
```python
# Import numpy as np
import numpy as np

# Store pop as a numpy array: np_pop
np_pop = np.array(pop)

# Double np_pop
np_pop = np_pop * 2

# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

# Display the plot
plt.show()
```
* color and opacity as argument for(c, alpha) for scatter plot
```python
# Specify c and alpha inside plt.scatter()
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c=col, alpha=0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Show the plot
plt.show()
```
* grid as a method for plt
```python
# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()
```

## Dictionary:
* Key value pairs
* You could do the same with two lists, take the index from one list and use that as the index for the other list.
* But, it's just too much useless code.
* Also, dictionaries are faster compared to lists, cause lists store index wise, so data is in linear fashion, but when it comes to dictionaries, it is randomly allocated and accessing the value is based on the key.
* Creating a dictionary:
    ```python
    pop_list = {"India":7,"China":10,"Japan":3}
    ```
* Accessing a dictionary element from the key:
    ```python
    print(pop_list['India'])
    #7
    ```
* Retrieve all keys from a dictionary:
    ```python
    print(pop_list.keys())
    print(type(pop_list.keys())
    #dict_keys(['germany', 'norway', 'spain', 'france'])
    #<class 'dict_keys'>
    ```
* Dictionary keys must be unique, even if you add multiple keys, with same or different values, it will only consider the latest one:
    ```python
    world = {"afghanistan":30.55, "albania":2.77, "algeria":39.21, "albania":2.81}
    print(world)
    #{'afganistan':30.55,'albania':2.81,'algeria':39.21}
    ```
* Keys have to be immutable objects like bools, int, string:
    ```python
    random_shit = {1:"one", "two":2, True:"3"}
    ```
* Adding data to a dic: If key exists the value is updated, otherwise key and value are added.
    ```python
    random_shit[3] = 2000
    ```
* Deleting a key:
    ```python
    del(random_shit[True])
    ```

# Pandas:
## What is pandas?
* Representing data in rows and columns format(excel, relational databases)
 
## Why would you use pandas over numpy when you could do the same shit with numpy?
* See in most cases, when it comes to excel or relational databases, each column would be of different datatypes, so have to nest each numpy array because w.k.t each numpy array can have only one datatype. We'd have to create a lot of nested arrays. This is not an efficient way.

## Creating a pandas dataframe(2d data structure) from a dictionary:
```python
import pandas as pd
random_shit = {1:"one", "two":2, True:"3"}
brics = pd.DataFrame(random_shit)
# It will have default indices.
```

## You can give your own indices:
```python
brics.index = ['BR', 'RU', 'JP', 'IN', 'US']
```

## Creating a pandas dataframe from a csv file:
```python
brics = pd.read_csv("bjbj.csv", index_col=0)  #remove the initial column, which would contain only column index like 0,1,2,... and make the first column the zeroth index
```

## Column access:
```python
# specifying the column name
brics["country"]
# returns the entire column
type(brics["country"])
# pandas.core.series.Series (series is a 1d labelled array)
type(brics[["country"]])
# pandas.core.frame.DataFrame
print(brics[["country","capital"]]) 
```

## Row access:
```python
brics[1:4]
```

## Label based indexing:
* Main advantage of using loc is because, you can do both row and column based indexing at the same time. Check example 3 and 4 below.
```python
brics.loc["RU"]  # series
brics.loc[["Ru"]]  # DataFrame
# If you want to do something like, np.array(row,column)
brics.loc[["RU", "IN", "CH"], ["country", "capital"]]
# If you want all the rows and only specific columns
brics.loc[:, ["country", "capital"]]
```

## Integer position based indexing:
```python
brics.iloc[1]
brics.iloc[[1]]
brics.iloc[[0,1,2],[0,1]]  # np.array(row,column)
brics.iloc[;, [0,1]]
```