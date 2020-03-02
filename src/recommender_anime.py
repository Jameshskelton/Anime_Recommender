import numpy as np
import pandas as pd

import pyspark as ps
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, StringType, FloatType, DateType
import pyspark.sql.functions as F

class AnimeRecommender_loop1():
    """Loop 1 recommender
    Uses ALS out of the box"""
    def fit(self, train_df):
        self.als = ALS(rank=200,
              maxIter=18,
              regParam=0.1,
              userCol="user_id",
              itemCol="anime_id",
              ratingCol="my_score")

        self.loop1_model = self.als.fit(train_df)

    def transform(self, test_df):
        return(self.loop1_model.transform(test_df))


class AnimeRecommender_loop2():
    """Loop 2 recommender 
    Uses average per anime to fill up NaNs left by ALS"""
    def fit(self, training_df):
        self.avg_ratings = training_df.select('anime_id','my_score', 'stats_mean_score')\
                                      .groupBy('anime_id')\
                                      .agg(F.avg('my_score'))\
                                      .withColumnRenamed('avg(my_score)','avg_rating')

    def transform(self, requests_df):
        return(requests_df.join(self.avg_ratings, 'anime_id', 'left')\
                          .withColumnRenamed('avg_rating','prediction'))


class AnimeRecommender(ALS):
    """Aggregate the results from loop1 and loop2 """
    def __init__(self):
        self.spark = ps.sql.SparkSession.builder \
                    .master("local[4]") \
                    .appName("df lecture") \
                    .getOrCreate()
        self.sc = self.spark.sparkContext  # for the pre-2.0 sparkContext
        self.ar1 = AnimeRecommender_loop1()
        self.ar2 = AnimeRecommender_loop2()


    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.
        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'
        Returns
        -------
        self : object
            Returns self.
        """
        

        self.training = ratings
        self.training.persist()

        self.ar1.fit(self.training)
        self.ar2.fit(self.training)

        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.
        Then fills NaN values with avg. rating of the anime
        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'
        Returns
        -------
        dataframe : a *pandas* dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """

        self.requests = requests

        # GET FIRST ROUND PRED
        pred_loop1 = self.ar1.transform(self.requests).withColumnRenamed('prediction','prediction_loop1')
        # option to show        
        # print(pred_loop1.show())

        # GET AVERAGES OF VALS TO FILLNA FOR MISSING VALUES AFTER PRED_LOOP1
        pred_loop2 = self.ar2.transform(pred_loop1).withColumnRenamed('prediction','prediction_loop2')
        #option to show
        #  print(pred_loop2.show())

        # FILLNA USING PRED_LOOP2. OPTION TO FILLNA WITH USER AVERAGE IF NEEDED

        results_loop2 = pred_loop2.withColumn('prediction',F.when(F.isnan('prediction_loop1'),F.col('prediction_loop2')).otherwise(F.col('prediction_loop1')))
        
        # if you want to see how many Nan's remain, free up 3 lines below

        # print(results_loop2.count())
        # print(results_loop2.dropna(subset=('prediction_loop1')).count())
        # results_loop2 = pred_loop2.withColumn('prediction',F.when(F.isnan('prediction_loop1'),F.col('prediction_loop2')).otherwise(F.col('prediction_loop1')))

        predictions = results_loop2.select('user_id', 'anime_id', 'my_score', 'prediction')\
                                   .withColumnRenamed('prediction','rating')
        
        
        # see how many nans lost here and drop them for testing

        # print(predictions.count())
        preds = predictions.dropna(subset=('rating'))
        # print(preds.count())
        return preds
    
    def evaluate(self, predictions):
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="my_score",predictionCol="rating")
        
        rmse = evaluator.evaluate(predictions)
        return print(f'the rmse is {rmse}')