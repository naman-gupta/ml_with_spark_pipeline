# DataChallenge 2019

This project is implemented using Pyspark API. Minimal feature engineering is done : 
1. OneHotEncoding of categorical values.
2. OneHotEncoding of categorical values but available as Integers like `publisher_app`. However it is not included in the current pipelined model but has been coded.
3. The task is treated as Multi-class-classification problem where the trained model predicts most likely class i.e conversion_event.
4. Training phase trains a simple RandomForestClassifier and saves it in the current directory for prediction phase. This phase also reports the `training error`
4. Prediction phase reads the saved model and predicts the most likely (highest probable) conversion_event
5. Possible modification to improve the classification
    * Using Better feature representation scheme for categorical values like TF-IDF, Word2Vec
    * Vectors are sparse and have very high dimentionality. Methods like PCA, LDA could be used to reduce dimensions
    * Better selection of training algorithm.
    * Performing GridSearch for selecting best paramater
    
# Running Project
1. Clone this repository
     ```
        git clone https://github.com/naman-gupta/remerge_data_challenge.git 
    ```
2. Install `virtualenv` 
    ```
        sudo apt-get install python3-pip
        sudo pip3 install virtualenv 
    ```
2. Create and activate virtual enviroment
     ```
        cd remerge_data_challenge
        virtualenv .
        source bin/activate
     ```
3. Install required packages
     ```
    pip install -r requirements.txt
     ```
4. Training Phase
    ```
    ./lib/python3.6/site-packages/pyspark/bin/spark-submit predictions.py train <location_of_datachallange_2019 or training_files>
    ```
5. Prediction Phase
    ```
    ./lib/python3.6/site-packages/pyspark/bin/spark-submit predictions.py predict <location_of_test_folder>
    ```


