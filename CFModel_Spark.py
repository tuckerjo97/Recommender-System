from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.recommendation import *
from pyspark.sql.functions import udf, concat_ws


def load_music_data(data_path):
    """
    Loads implicit data set into a RDD. Need to change formatting functions for different datasets.

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
    df_cleaned = df_cleaned.filter(df_cleaned.traname != "[Untitled]")
    df_cleaned = df_cleaned.filter(df_cleaned.traname != "Untitled")
    df_formatted = df_cleaned.withColumn("userid", userid_format_udf(df_cleaned.userid))
    df_formatted = df_formatted.withColumn("userid", df_formatted.userid.cast(IntegerType()))
    df_formatted = df_formatted.withColumn("traname", concat_ws(" - ", df_formatted.traname, df_formatted.artname))
    df_formatted = df_formatted.drop("timestamp", "artid", "artname")
    unique_songs = df_formatted.select("traname").distinct().collect()

    id_song_db = {}
    song_id_db = {}
    for num in range(len(unique_songs)):
        song_id_db[unique_songs[num]["traname"]] = num
        id_song_db[num] = unique_songs[num]["traname"]
    traid_format_udf = udf(lambda x: song_id_db[x], IntegerType())
    df_formatted = df_formatted.withColumn("traid", traid_format_udf(df_formatted.traname))
    df_with_count = df_formatted.groupBy("userid", "traid").count().sort("userid")

    print("Data successfully loaded and formatted from: {}".format(data_path))

    return df_with_count, id_song_db


def load_movie_data(rating_data_path, movie_data_path):
    """
        Loads explicit movie data set into a RDD.

        Arguments:
              data_path: path for where to find data set to import

        Return:
            Dataframe of cleaned and formatted data for movieLens dataset. Contains Columns (userid, movieid, rating)
            movie_id_db: Dictionary of song id to song name
        """
    spark = SparkSession \
        .builder \
        .getOrCreate()

    rating_schema = StructType([
        StructField("userid", IntegerType(), True),
        StructField("movieid", IntegerType(), True),
        StructField("rating", DoubleType(), True)])

    movie_schema = StructType([
        StructField("movieid", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True)])

    rating_df = spark.read.csv(rating_data_path, header=True, schema=rating_schema)
    movie_df = spark.read.csv(movie_data_path, header=True, schema=movie_schema)

    movie_df = movie_df.drop('genres')
    movie_dict = movie_df.toPandas()
    movie_dict['new_movie_id'] = range(len(movie_dict))
    id_dict = dict(zip(movie_dict['movieid'], movie_dict['new_movie_id']))
    rating_udf = udf(lambda x: id_dict[x])
    count = 0
    rating_df = rating_df.withColumn('movieid', rating_udf(rating_df.movieid))  # This line is causing an EOFError. for some reason the values in movieid are not working, but the ones in user id do. It's not because i'm passing them into a dictionary to look up.
    rating_df = rating_df.withColumn('movieid', rating_df.movieid.cast(IntegerType()))
    id_to_movie_dict = dict(zip(movie_dict['new_movie_id'], movie_dict['title']))
    return rating_df, id_to_movie_dict

def fit_model(dataframe, rank, alg="ALS"):
    """
    Performs implicit ALS

    Arguments:
        dataframe: Already formatted dataframe in the form of (User, Item, Count). Should be implicit data.
        rank: rank of matrices produced by ALS
        alg: algorithm used in recommender system. Defaults to Alternating Least Squares. Could also be SVD or nearest
        neighbor

    Return:
        model: model of ratings based off implicit data.
    """

    als = ALS(rank=rank, implicitPrefs=True, seed=42069, userCol="userid", itemCol="movieid", ratingCol="rating")
    model = als.fit(dataframe)


    return model


def recommend(model, item_db, users, numrecommend = 5):
    """
    Prints numrecommend recommendations for each user in users
    Arguments:
        model: Trained model from fit_model
        item_db: Database of itemid to itemname
        users: either a single user id or a list of user ids
        numrecommend: the number of recommended items returned for each user
    """
    recommend_df = model.recommendForAllUsers(numrecommend).sort("userid")
    recommend_df.show()
    for user in users:
        items = recommend_df.filter(recommend_df.userid == user).select("recommendations").collect()[0][0]
        user_recs = []
        for line in items:
            id = line[0]
            user_recs.append(item_db[id])
        print("User {} should listen to: {}".format(user, user_recs))


if __name__ == "__main__":
    # path = "lastfm-dataset-1K\\userid-timestamp-artid-artname-traid-traname.tsv"
    # lastfm_df, song_to_id_db = load_music_data(path)
    # als_model = fit_model(lastfm_df, 10)
    # recommend(als_model, song_to_id_db, [1,2])

    rating_path = "ml-latest-small\\ratings.csv"
    movie_path = "ml-latest-small\\movies.csv"
    rating_df, movie_dict = load_movie_data(rating_path, movie_path)
    als_model = fit_model(rating_df, 10)
    recommend(als_model, movie_dict, [0])
