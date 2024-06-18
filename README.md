# Disaster Response Pipeline Project

<img width="1107" alt="Screenshot 2024-06-17 at 10 55 17 PM" src="https://github.com/apolanco3225/Data-Engineering-for-Data-Scientists/assets/16232171/c4756859-d779-4497-aa3e-0490b7e04331">

In this project, we will develop a model to classify messages transmitted during disaster events. The messages will be sorted into 36 pre-defined categories, including but not limited to Aid Related, Medical Help, and Search and Rescue. By accurately classifying these messages, we can ensure they are directed to the appropriate disaster response agencies. This project encompasses the creation of a basic ETL (Extract, Transform, Load) and Machine Learning pipeline to streamline the classification process. Given that a single message can fall into multiple categories, this is a multi-label classification task. We will utilize a dataset provided by Figure Eight, which includes real messages sent during past disaster scenarios.

Additionally, this project features a web application where users can input a message and receive classification results in real-time.


## File Description
~~~~~~~
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- README
~~~~~~~



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
