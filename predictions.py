from __future__ import print_function

import sys
import logging
import numpy as np
import argparse
import os
import re
import glob

from pyspark.sql import SparkSession
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

MAX_MEMORY = "10g"  # Maximum Memory for SPARK


# Check if resources are available
def checkResourceAvailability(location, logger, checkForPredictionModel = False):
    if not os.path.isdir(location):
        logger.error("Location {} does not exists".format(location))
        sys.exit(-1)

    if checkForPredictionModel:
        if not os.path.exists('PredictionModel.model'):
            logger.error("PredictionModel not available in the current directory. "
                         "Use <train> or copy the model to current directory")
            sys.exit(-1)

    if not os.path.isfile('{}/campaign_data.csv'.format(location)):
        logger.error("campaign_data.csv not available in {}".format(location))
        sys.exit(-1)

    if len(glob.glob('{}/data_*.csv'.format(location))) == 0:
        logger.error("data files not available in {}".format(location))
        sys.exit(-1)

# Adding label to conversion event
def func(event_type, conversion_event):
    if event_type == "click":
        return conversion_event + "_" + event_type
    return conversion_event + "_conversion"

# Get Pyspark Logger
def getLogger(spark):
    log4jLogger = spark.sparkContext._jvm.org.apache.log4j
    return log4jLogger.LogManager.getLogger(__name__)

# Combine click conversion data with campaign data to get labels
# and filtering out invalid or null values.
def read_merge_clean_dataset(campaign_data, click_conversion_data, sampling=False):
    if sampling:
        click_conversion_data = click_conversion_data.sample(False, .1, seed=0)

    # Adding conversion event labels from campaign data
    click_conversion_data = click_conversion_data.join(campaign_data, click_conversion_data.campaign == campaign_data.campaign_id)

    # Creating Labels ( Conversion or Click)
    # func_udf = udf(func, StringType())
    # df = df.withColumn('campaign', func_udf(df['event_type'], df['conversion_event']))

    # Selecting feature columns
    click_conversion_data = click_conversion_data.select(
        'country',
        'device_type',
        'os',
        'os_version',
        'partner',
        'publisher_app',
        'organization',
        'audience',
        'conversion_event',
        'ad_type',
        'ad_size')

    # Dropping null/ incomplete values
    click_conversion_data = click_conversion_data.dropna()

    return click_conversion_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, help="train/predict")
    parser.add_argument("location", type=str, help="location of data folder for train/predict")

    args = parser.parse_args()

    # Train or Predict
    action = args.action

    # Location of data or test folder
    location = args.location

    if action == "train":

        # Creating Spark Session
        spark = SparkSession\
                .builder\
                .master("local[2]") \
                .appName("train-conversion-model") \
                .config("spark.driver.memory", MAX_MEMORY) \
                .config("spark.driver.maxResultSize",MAX_MEMORY) \
                .getOrCreate()

        logger = getLogger(spark)

        logger.info("Spark Session Created {}".format(logger))

        checkResourceAvailability(location, logger, checkForPredictionModel=False)

        # Campaign Data
        campaign_data = spark.read.csv('{}/campaign_data.csv'.format(location), header=True, inferSchema=True)

        # Clicks and conversion data
        df = spark.read.csv('{}/data_*.csv'.format(location), header=True, inferSchema=True)

        # Taking Sample of data due to memory restrictions
        df = read_merge_clean_dataset(campaign_data=campaign_data,click_conversion_data=df, sampling=True)

        # Feature Engineering
        # 1 : Convert String features into OneHotEncodingFormat
        # 2 : Convert categorical Integer features to String and then into OneHotEncoded Vectors
        # 3 : Assemble all vectors to get single feature vector
        # 4 : Execution of the pipeline

        # Part 1 : Conversion of categorical string columns
        categoricalStringColumns = ['country', 'device_type', 'os', 'partner', 'organization', 'ad_size']

        stages = []  # stages in our Pipeline
        for categoricalCol in categoricalStringColumns:
            # Category Indexing with StringIndexer
            stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index", handleInvalid='keep')
            # Use OneHotEncoder to convert categorical variables into binary SparseVectors
            encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],
                                             outputCols=[categoricalCol + "classVec"])
            # Add stages.  These are not run here, but will run all at once later on
            stages += [stringIndexer, encoder]

        # Part 2 : Conversion of categorical integer/double columns (Not using for the moment)
        '''
        #categoricalIntegerColumns = ['publisher_app','audience','ad_type']
        oneHotEncoder = OneHotEncoderEstimator(inputCols=categoricalIntegerColumns, outputCols=[column + "classVec" for column in categoricalIntegerColumns]).setHandleInvalid("error")
        stages += [oneHotEncoder]

        for categoricalCol in categoricalIntegerColumns:
            # Use OneHotEncoder to convert categorical variables into binary SparseVectors
            encoder = OneHotEncoderEstimator(inputCols=[categoricalCol], outputCols=[categoricalCol + "classVec"])
            # Add stages.  These are not run here, but will run all at once later on.
            stages += [encoder]
        '''

        # Part 3 : Assembling features
        assemblerInputs = [c + "classVec" for c in categoricalStringColumns]
        assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
        stages += [assembler]

        # Part 4 : Convert Labels to index
        label_stringIdx = StringIndexer(inputCol="conversion_event", outputCol="label", handleInvalid='keep').fit(df)
        stages += [label_stringIdx]


        # Part 5 : Splitting dataset for training and testing
        train, test = df.randomSplit([0.7, 0.3])

        # Part 6 : Training a basic Random Forest Classifier
        rf = RandomForestClassifier(labelCol="label", featuresCol="features")

        labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=label_stringIdx.labels)

        pipeline = Pipeline().setStages(stages + [rf, labelConverter])

        clf = pipeline.fit(train)

        clf.write().overwrite().save("PredictionModel.model")

        logger.info("Prediction Model saved in current directory")
        # Generating Predictions
        predictions = clf.transform(test)

        predictions.select('conversion_event', 'label', 'rawPrediction', 'probability', 'prediction', 'predictedLabel').show(100)
        # Evaluation
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        logger.info("Test Error = %g" % (1.0 - accuracy))

        spark.stop()

    elif action == "predict":

        # Creating Spark Session
        spark = SparkSession \
                .builder \
                .master("local[2]") \
                .appName("train-conversion-model") \
                .config("spark.driver.memory", MAX_MEMORY) \
                .config("spark.driver.maxResultSize", MAX_MEMORY) \
                .getOrCreate()

        logger = getLogger(spark)

        logger.info("Spark Session Created {}".format(logger))

        checkResourceAvailability(location, logger, checkForPredictionModel=True)

        # Read Conversion Data
        campaign_data = spark.read.csv('{}/campaign_data.csv'.format(location), header=True, inferSchema=True)

        # Clicks and conversion data
        test_data = spark.read.csv('{}/data_*.csv'.format(location), header=True, inferSchema=True)

        test_data = read_merge_clean_dataset(campaign_data=campaign_data, click_conversion_data=test_data)

        predictionModel = PipelineModel.load("PredictionModel.model")

        predictions = predictionModel.transform(test_data)

        logger.info('Predictions Complete')

        # Evaluating Predictions.
        #evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
        #                                              metricName="accuracy")
        #accuracy = evaluator.evaluate(predictions)

        #logger.info("Prediction accuracy {}".format(accuracy))

        predictions.select('conversion_event', 'predictedLabel').show(100)
    else:
        print("Unknown action, please select from train or predict")
        sys.exit(-1)