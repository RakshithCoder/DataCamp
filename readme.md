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

## **Background concepts to be known before supervised and unsupervised learning:**
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

## **Unsupervised Learning:**
* We don't have labels in ul, only features.
* We use this when we want to divide the data into groups based on similiarity.
* In this case, we'd want to use ul to understand the different groups of patients like one category would be, patients with high cholesterol and other group to be high sugar levels. 
* So, we can use ul to identify the different groups of ppl based on the features and probably give them better treatment.
* To train the model, we only need to provide observations and features.
* Once, the model is trained, it, we give it new observations, i.e new patient in this case and it predict that that specific patient falls into which category. 
* So in a nutshell, in ul, we don't really have labels. 

## **MachineLearning Workflow:**
* First, we extract features from raw data.
* Split the dataset into training and testing.
* Train the model.
* The model learns only from the training data.
* And it tries to evaluate what it's learnt from the testing data.
* The testing data is unseen, i.e it has never seen that data before.
* If the evaluation is okay and is giving good accuracy, we save the model, otherwise we tune the model and re-train the model.(adding or removing some of the features)

## **More info on Supervised and Unsupervised learning:**
* Supervised Learning
    1. Classification
    2. Regression

    Classification | Regression
    -------------- | ----------
    assigning a **category** | assigning a **continous** variable
    Like will this customer stop his subscription - Yes, No | How much will this stock be worth?
    What flower is that? - Rose, Tulip, Carnation | How tall will this child be as an adult?
    Only a few specific values | Any no. of values with a finite/infinite time interval.
    SVM | LR

* Unsupervised Learning
    1. Clustering
    2. Association
    3. Anomaly Detection

    **Clustering:** 
    Clustering is similar to classification, but in classification it groups the data based on the labels, like categories must be made based on only those labels. 
    But in clustering, since it is an unsupervised learning, and there is no label, it groups based on **similarities**, what kind of similarities the algo finds and groups depends, for eg: It could make two groups dog and cat or it could cluster based on color like black, red, green, etc.
    Eg: K-means. Here we have to mention the no. of clusters beforehand. Take a dataset with unknown flowers, we only have flower names and length. We obviously can't do classification cause we don't how many classes are there, we can just analyze the data manually and decide on the no. of clusters with k-means. 
    
    **Association:**
    Association involves finding **relationships b/w the observations or finding events that occur together.** Eg: ppl who buy jam are likely to buy butter too. "ppl who bought this also bought this" in amazon.

    **Anomaly Detection:**
    Anomaly Detection involves **finding the odd one out.** The odd one out is called an outlier. Eg: Finding out which devices fail faster, which patients resist a fatal disease. 

## Evaluating Performance:
* This is to see how our model is performing on unseen data.
    * Supervised Learning:
        * Evaluating Classification:
            * Accuracy = No. of correctly predicted observations/total no. of observations
                * Sometimes when the data is imbalanced, the model will predict the value of the majority class for all predictions and achieve a high accuracy.
                * So, accuracy is not always the best measure.
            * Confusion matrix:
                ![Image of CM](https://miro.medium.com/max/625/1*fxiTNIgOyvAombPJx5KGeA.png)
                * TP: A true positive is an outcome where the model correctly predicts the correct class and in reality it is also belongs to the correct class.
                Eg: Umpire gives a batsman **NOT OUT** when he is **NOT OUT** 
                * TN: A true negative is an outcome where the model correctly predicts that it is an incorrect class and in reality it is also belongs to the incorrect class.
                Eg: Umpire gives a batsman **OUT** when he is **OUT**  
                * FN: A false negative is when the model predicts a class as negative, but in reality it is actually positive. 
                Eg: Umpire gives a batsman **NOT OUT** when he is **OUT**
                * FP: A false positive is when the model predicts a class as positive, but in reality it is actually negative.
                Eg: Umpire gives a batsman **OUT** when he is **NOT OUT**
            * Sensitivity = How many wrong predictions did the model classify as correct.
            i.e TP/TP+FN
        * Evaluating Regression:
            * Distance b/w the points and the predicted line. Eg:RMS error. 
    * Unsupervised Learning:
        * Since, ul doesn't have predicted variables, therefore there is no correct output to compare to.
        * It depends on the problem and how well the results match your initial objective.

## Improving performance:
* Once, if you've evaluated the performance of your model, if the performance is not satisfactory, how to improve the model performance.
    * Dimensionality reduction:
        * Reducing the no. of features(dimensions) in your data.
        * You might think, more no. of features mean better predictions. Most of the time, some of the features might be irrelevant.
        * Eg: If we are trying to predict how long it will take us to go to office, time of the day and weather are imp features, but how many bottles of water we drank last year is not imp.
        * Some features might be highly correlated and carry similar information. We could keep only one feature and still have most information. For example, height and shoe size are highly correlated. Tall people are very likely to have a large shoe size. We could keep only the height without losing much additional information. We could also collapse multiple features into just one underlying feature. If we have two features, height and weight, we can calculate one Body Mass Index feature instead.
    * Hyperparameter tuning:
        * A hyperparameter is a parameter whose value is used to control the learning process.
        * This inturn, affects the model performance.
        * Eg: Consider an SVM, changing the kernel from linear to polynomial.
        * Eg: In a deep learning model, batch size, dropout, epochs, etc.
        * Eg: For a guitar, we change the settings based on the genre, genre is the dataset and guitar settings are the hyperparameters.
    * Ensemble methods:
        * Ensemble methods are techniques that create multiple models and then combine them to produce improved results.
            * Classification:
                * Imagine we have three different models A,B,C, we use voting. If model A and model B predicts a student is accepted and model c says the student is not. We accept the most common and the model is accepted since we had two models which predicted properly.
            * Regression:
                * We use averaging. model A-8, B-5, C-4.
                * Avg is 5.67.




        



