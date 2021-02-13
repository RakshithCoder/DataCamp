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