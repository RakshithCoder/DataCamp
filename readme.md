# DataCamp Notes
## What is **Machine Learning?**
* Set of tools to make **inferences,  predictions and find out patterns** from data.
## What Machine Learning **can do?**
* **Predict** future events
  * Will it rain tomorrow?
    * Yes
* **Infer** causes of events behaviour
    * Why does it rain?
        * Time of the year, temperature.
* **Patterns**
    * Different types of weather conditions.
        * Summer, winter.
## How does **ml work?**
* Learns patterns from existing data and applies it to new data.
* Like you can tell it what spam emails look like, later on it will detect new emails which are spam.
## Difference b/w **Data Science, AI and Machine Learning**
* AI is trying to make computers think like humans. Like:
    * Game playing algorithms
    * Self driving cars

* Data Science is making discoveries or trying to find meaning from data. Like:
    * Recommendation Systems
    * Forecasting

* Machine Learning is a subset of AI, which enables it to automatically improve itself through experience. 

* Deep Learning is a subset of AI, which specifically is concentrated on making machines think like humans.

![Image of Terms](https://miro.medium.com/max/410/1*4Wl_EOjivZu1qmdUBol7WQ.png)

## What is a **machine learning model?**
* Representation of a real-world process based on data.
* If we make a model based on historical traffic data, we can enter a future data into the model to predict how heavy traffic will be there tomorrow.
* Output can even be a probability of an outcome. Like probability of a tweet being fake.

## Three **types** of ML:
* Reinforcement Learning
* Supervised Learning
* Unsupervised Learning

## **Reinforcement Learning:**
* It is used for deciding sequential actions, like a robot deciding its path or its next move in a game.
* Reinforcement Learning uses game theory.

## Background concepts to be known before supervised and unsupervised learning:
* We know from before that ml **learns** patterns from existing data and applies it to new data.
* This existing data is called **training data**
* When a model is being built and learning from training data, that concept is called **training a model**
    * Depending on the size of the data, it can take nanoseconds to weeks.

## **Supervised Learning:**
* Let's say we want to train a model to predict whether or not a person has heart disease.
* We have existing of ppl who've had chest pains and have tested postive for heart disease.
* Some terminologies that are present here are:
    * Target variable
    * Labels
    * Observations
    * Features
* The way we can relate these terminologies to the example above is:
    * The way a target variable works is, the actual thing which we need to predict is taken as a target variable. 
    Target variable is the heart disease.
    * The values for the target variable are the labels i.e true or false if a person has a heart disease.
    Labels can be in numbers or in categories.
    * Observations are basically the multiple rows of data for eg: multiple patient records.
    The more observations or rows you have, the better it is.
    * Features and observations go hand in hand.
    Features are the columns for those observations.
    Like age, sex, cholesterol level, etc.
    Features are the ones, that help us to predict our target.
    * In machine learning we can analyze multiple features at once and determine the relationship b/w them.
    * We input labels and features as data to train our model.
    * Once the model is trained, we can give the model new input i.e labels and observations, in our case a new patient/patients and it will predict if that patient has heart disease or not.
* So in a nutshell, in supervised learning, our training data is labelled, that is we had previous patients who had heart disease, based on labels t or f.

## Unsupervised Learning:
* We don't have labels in ul, only features.
* We use this when we want to divide the data into groups based on similiarity.
* In this case, we'd want to use ul to understand the different groups of patients like one category would be, patients with high cholesterol and other group to be high sugar levels. 
* So, we can use ul to identify the different groups of ppl based on the features and probably give them better treatment.
* To train the model, we only need to provide observations and features.
* Once, the model is trained, it, we give it new observations, i.e new patient in this case and it predict that that specific patient falls into which category. 
* So in a nutshell, in ul, we don't really have labels. 


  


