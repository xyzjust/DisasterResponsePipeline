# Disaster Response Pipeline Project

### Motivation

When a disater occurs, it is vital to be able to deliver aid to the people in need as fast as possible. Modern technology has allowed us to send messages with ease, whilst this is an advantage in many ways, an organisation in charge of deliverying help is often bombarded with messages as a result. This project aims to use machine learning to analyse the messages recieved and label them with the appropriate labels so the corresponding teams can act fast in deliverying resources.


## Results

This project that builds a **web app** and a ML pipeline to train a **supervised multilabel classifier** to classify text messages into appropriate categories (e.g. food, shelter etc). The web app is built using Flask and Boostrap. The outcome is a web app where a message can be inputted and based on the message, the approriate categories will be highlighted.

#### Example of the final web app
![alt text](https://raw.githubusercontent.com/xyzjust/DisasterResponsePipeline/master/example.png)


## Instructions for setup
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database <br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves <br>
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command **in the app's directory** to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Dependencies/Packages used
- xgboost  :  **1.3.0.post0**
- scikit-learn  :  **0.23.2**
- pandas  :  **1.1.3**
- numpy  :  **1.19.2**
- nltk  :  **3.5**



## A Note on imbalanced data
the training data is imbalanced in a way such that for certain categories, there are more negative results than positives ones in the order of 10^3, the training model here has tried to resolve this in some ways by introducing a weight when training the model. However, the method was rudimentary and can be improved upon.
