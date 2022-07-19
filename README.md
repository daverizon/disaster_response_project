# disaster_response_project
This code is aimed at taking message inputs from company Figure Eight Inc., and classifying them into pre-determined categories


## Project motivation
Messages that are provided by Figure Eight Inc. are pre-labeled tweets and text messages from real life disaster.  It was required to create an ETL pipeline for the messages, and then use a Machine Learning Pipeline to build a supervised learning model in order to correcly classify the messages.

## Libraries used
pandas - Pandas is a standard library for analyzing and wrangling data from python

sqlalchemy - SQLAlchemy is a Python library used to connect and interact with various different types of databases

sklearn - This package is a standard package in Python for machine learning and is used to build a Linear Regression model in this project

nltk - This is a Python library that contains libraries and programs for statistical language processing, and is one of the most powerful NLP libraries avialable

plotly - This is a Python library which inclues data analytics and visualization tools

flask - This is a web framework that includes a Python model to develop web applications easily


## Files in the repository
This repository has the following structure as outlined in the project details
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

## How to run
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Go to `app` directory: `cd app`
3. Run your web app: `python run.py`
4. Click the `PREVIEW` button to open the homepage

## Summary of results
The result of this project is a model which classifies messages into the presubscribed categories given.  The pickle file of the model is located in models > classifier.pkl

Performance of the classification is shown below, where for each category the precision, recall, f1-score, and support are summarized.  Below all the categories is the overall average/total of all categories

Category | Precision | Recall | F1-Score | Support
--- | --- | --- | --- | ---
related | 0.85 | 0.94 | 0.89 | 6034
request | 0.82 |0.55 | 0.66 | 1355
offer | 0.00 | 0.00 | 0.00 | 30
aid_related | 0.78 | 0.67 | 0.72 | 3260
medical_help | 0.63 | 0.24 | 0.35 | 626
medical_products | 0.69 | 0.29 | 0.41 | 397
search_and_rescue | 0.69 | 0.17 | 0.27 | 225
security | 0.33 | 0.02 | 0.04 | 158
military | 0.59 | 0.28 | 0.38 | 268
water | 0.74 | 0.57 | 0.65 | 513
food | 0.81 | 0.67 | 0.73 | 853
shelter | 0.76 | 0.55 | 0.64 | 709
clothing | 0.70 | 0.40 | 0.51 | 124
money | 0.65 | 0.23 | 0.34 | 186
missing_people | 0.42 | 0.05 | 0.10 | 92
refugees | 0.70 | 0.24 | 0.35 | 292
death | 0.72 | 0.43 | 0.54 | 378
other_aid | 0.55 | 0.15 | 0.24 | 1017
infrastructure_related | 0.39 | 0.05 | 0.09 | 530
transport | 0.72 | 0.20 | 0.32 | 350
buildings | 0.67 | 0.34 | 0.45 | 397
electricity | 0.46 | 0.23 | 0.31 | 157
tools | 0.00 | 0.00 | 0.00 | 43
hospitals | 0.29 | 0.04 | 0.08 | 90
shops | 0.00 | 0.00 | 0.00 | 48
aid_centers | 0.50 | 0.01 | 0.02 | 94
other_infrastructure | 0.38 | 0.04 | 0.07 | 351
weather_related | 0.84 | 0.69 | 0.76 | 2208
floods | 0.88 | 0.56 | 0.68 | 655
storm | 0.73 | 0.57 | 0.64 | 705
fire | 0.62 | 0.21 | 0.32 | 85
earthquake | 0.89 | 0.79 | 0.83 | 728
cold | 0.72 | 0.34 | 0.47 | 175
other_weather | 0.62 | 0.15 | 0.24 | 392
direct_report | 0.76 | 0.45 | 0.56 | 1545
--- | --- | --- | ---
avg / total | 0.75 | 0.59 | 0.64 | 25070

## Acknowledgements
I'd like the thank Verizon Communications who paid for this course and enabled me to further my learning in this deep field of Data Science.  I'd also like to thank Udacity for hosting the course in a format that accomodates my learning style most-efficiently
