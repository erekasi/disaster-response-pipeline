# disaster-response-pipeline
 Classification model for disaster messages and a web app for model deployment.

 ### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    # 'cd app' in the Terminal
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files
There are three folders in this repository:
1. data:
        - disaster_messages.csv: one of the datasets to train the model
        - disaster_categories.csv: the other dataset to train the model
        - process_data.py: script to load, clean and transform the above datasets
        - DisasterResponse.db: SQLite database that was created by running process_data.py as specified in the Instructions of the current README

2. models:
        - train_classifier.py: script to train a model to classify messages (with word tokenization and TFIDF)
        - classifier.pkl: pickle file created by saving the model as specified in the Instructions of the current README

3. app:
        - run.py: script to run the webapp that applies the aforementioned classifier model to new messages thus determines the predicted category of a new message typed in the app
        - templates (folder): html files to design the template of the webapp
            * go.html
            * master.html