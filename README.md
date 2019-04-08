# Disaster Response Pipeline Project

### Project description:
The goal of this project is to design a web app that takes a message and classifies the category of that message. In order to
achieve that, a dataset of several of thousands of labeled text messages where loaded, cleaned and saved in a SQL database using the ETL pipeline approach.
The saved data is used to train and test a multi-output random forest classifier. The trained model is used as backend of the web app which provides a user interface to enter a text message
that is to be classified.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
