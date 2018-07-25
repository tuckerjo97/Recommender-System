from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.recommendation import *
from pyspark.sql.functions import udf


def load_data(data_path):
    """
    Loads implicit data set into a RDD

    Arguments:
          data_path: path for where to find data set to import

    Return:
        Dataframe of cleaned and formatted data for lastfm dataset. Contains Columns (userid, traid, count)
        song_id_db: Dictionary of song id to song name
    """

    spark = SparkSession\
        .builder\
        .getOrCreate()

    schema = StructType([
        StructField("userid", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("artid", StringType(), True),
        StructField("artname", StringType(), True),
        StructField("traid", StringType(), True),
        StructField("traname", StringType(), True)])

    df_raw = spark.read.csv(data_path, header=False, schema=schema, sep="\t")
    userid_format_udf = udf(lambda x: x[5:], StringType())
    df_cleaned = df_raw.dropna()
    df_cleaned = df_cleaned.drop("timestamp", "artid", "artname")
    df_formatted = df_cleaned.withColumn("userid", userid_format_udf(df_cleaned.userid))
    df_formatted = df_formatted.withColumn("userid", df_formatted.userid.cast(IntegerType()))
    unique_songs = df_formatted.select("traname").distinct().collect()
    id_song_db = {}
    song_id_db = {}
    for num in range(len(unique_songs)):
        song_id_db[unique_songs[num]["traname"]] = num
        id_song_db[num] = unique_songs[num]["traname"]
    traid_format_udf = udf(lambda x: song_id_db[x], IntegerType())
    df_formatted = df_formatted.withColumn("traid", traid_format_udf(df_formatted.traname))
    df_with_count = df_formatted.groupBy("userid", "traid").count().sort("userid")

    print("Data successfully loaded and formatted from: {}".format(path))

    return df_with_count, song_id_db


def perform_ALS(dataframe, rank):
    """
    Performs implicit ALS

    Arguments:
        dataframe: Already formatted dataframe in the form of (User, Item, Count). Should be implicit data.
        rank: rank of matrices produced by ALS

    Return:
        model: model of ratings based off implicit data.
    """

    als = ALS(rank=rank, implicitPrefs=True, seed=42069, userCol="userid", itemCol="traid", ratingCol="count")
    model = als.fit(dataframe)

    return model


def recommend(model, item_db, users, numrecommend = 5):
    recommend_df = model.recommendForAllUsers(numrecommend).sort("userid")
    recommend_df.show()
    for user in users:
        items = recommend_df.filter(recommend_df.userid == user).select("recommendations").collect()
        user_recs = []
        for line in items:
            id = line["traid"]
            user_recs.append(item_db[id])
        print("User {} should listen to: {}".format(user, user_recs))

path = "lastfm-dataset-1K\\userid-timestamp-artid-artname-traid-traname.tsv"
lastfm_df, song_to_id_db = load_data(path)
als_model = perform_ALS(lastfm_df, 10)
recommend(als_model, song_to_id_db, [0,1,2])
