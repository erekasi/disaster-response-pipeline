# Disaster Response Pipeline
 Classification model for disaster messages and a web app for model deployment.

 ### Table of contents

[1) Project motivation](#Motivation)

[2) Project description](#Description)

[3) Files](#Files)

[4) Instructions](#Instructions)

---

 ### 1) Project motivation<a name="Motivation"></a>
When a disaster occurs it is critical that those affected receive sufficient and relevant help as soon as possible. Affected people often call for help by themselves and disasters are also covered in the news and social media. Any of those sort of request for help, either direct or indirect, has to reach competent organizations which are competent to help. Many of such organizations are specialized though. Therefor it is critical that a message reaches the right organization without wasting time to minimize latency in the provision of help. For such situations, the incoming flood of requests for help should be processed and forwarded to the right helper, possibly automatically in order to act fast and even in case of limited human resources to process such information. The current project is a simple prototype for a solution that enables the automated categorisation of a message that is potentially a request for help in case of a disaster. Based on such kind of categorization (a more elaborated one compared to the solution presented here) an automated workflow to forward a message to the competent help providing organization could increase the efficiency and thus also the speed of help provision to those in need.

---

### 2) Project description<a name="Description"></a>
The project cosists of three phases:

**1. Data processing** <br>
    *Figure Eight* provided a set of disaster messages, which a message classification model could be trained on. The data was available in two datasets: messages and categories. The processing of these required the following steps:
    - load and merge datasets into a pandas dataframe
    - clean by splitting categories into separate features, creating dummy variables from the category values
    - convert string datatypes to numeric values
    - remove duplicates.

   At the end of this phase, data was saved into a SQLite database.
    
**2. Model training** <br>
    After loading the already processed data into a pandas dataframe, features were separated into predictor (X = messages) and target (Y = category variables) datasets. Both datasets were split into training and test sets.<br>
    A machine learning pipeline was set up that consisted of the following:
    - word tokenization - to split messages into words, normalizes these to lower case, skim these from *stop words* and punctuation, and lemmatize words
    - count vectorization - to count the number of occurences of each term (here: word) in each document (here: message)
    - TF-IDF transformation - to divide term frequency by document frequency, i.e. multiple by inverse document frequency (idf)
    - classification - to learn which categories a message belong to, by appliying a random forest classifier in a multi-output classifier (as there are many categories for which prediction per message needed to be made).

Grid-search was applied to tune one of the hyperparameters of the model (max depth of an estimator in a random forest model).<br>
Finally, the model was evaluated by precision, recall, F1-score and accuracy  on the test set.<br>
Considering the purpose of the classification, to find whether a message was requesting some specific help related to a disaster, recall might be the most relevant evaluation metric for the model. The model performed not extremely well as the average recall (with macro average) was low:<br>
    **Average recall = 0.5108**

**3. Setting up a web application** <br>
    The model saved from the training was loaded to be run in the webapp. Two visualizations were added to provide an overview on the nature of the training data set. A web page was set up to handle user query and display model results.
 
 ---

### 3) Files<a name="Files"></a>
There are three folders in this repository:
1. **data**:
    - `disaster_messages.csv`: one of the datasets to train the model
    - `disaster_categories.csv`: the other dataset to train the model
    - `process_data.py`: script to load, clean and transform the above datasets
    - `DisasterResponse.db`: SQLite database that was created by running `process_data.py` as specified in the Instructions of the current `README.md`

2. **models**:
    - `train_classifier.py`: script to train a model to classify messages (with word tokenization and TFIDF)
    - `classifier.pkl`: pickle file created by saving the model as specified in the Instructions of the current `README.md`

3. **app**:
    - `run.py`: script to run the webapp that applies the aforementioned classifier model to new messages thus determines the predicted category of a new message typed in the app
    - templates (folder): html files to design the template of the webapp
        - `go.html`
        - `master.html`

---

 ### 4) Instructions<a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database<br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves<br>
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run your web app.<br>
    ('cd app' in the Terminal)<br>
    `python run.py`


3. Go to http://0.0.0.0:3001/