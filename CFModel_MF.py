import time
import math
import numpy as np
import pandas as pd
import pyspark as ps
from math import sqrt
from pyspark import SparkContext
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from pyspark.mllib.recommendation import ALS
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split, KFold


# Reads and formats data into training and testing matrices. Data argument should be a tuple in
# format:(data_loc, data_header, info_loc, info_header). NEED TO DO: create u.info parser and add back in matrix_info
def import_format_data(data):
    raw_data = pd.read_csv(data[0], names=data[1], sep="\t")
    raw_data = raw_data.drop(labels='timestamp', axis =1)

    raw_info_data = pd.read_csv(data[2], names=data[3], sep="|")
    raw_info_data = np.array(raw_info_data)
    info_data = raw_info_data[:,1]

    # matrix_info = pd.read_csv(info_loc, names=info_header, sep="|" )
    return raw_data, info_data


# Splits the raw data into training and testing data
def data_split(raw_data):
    num_users = raw_data.user_id.unique().shape[0]
    num_items = raw_data.item_id.unique().shape[0]

    train_data_raw, test_data_raw = train_test_split(raw_data, test_size=0.2, random_state=40)
    train_data_matrix = create_data_matrix(train_data_raw, num_users, num_items)
    test_data_matrix = create_data_matrix(test_data_raw, num_users, num_items)
    return train_data_matrix, test_data_matrix


def find_user_ratings(matrix):
    list = []
    for line in matrix:
        list.append(np.array(np.nonzero(line))[0])
    return list


# Performs matrix decomposition through ALS. Returns two matrices:(user, latent factor), and (item, latent factors).
# Matrix is Transposed and masked in order to work with Alternating Least Square k argument is the number of dimensions
# desired for latent space. i is the desired number of iterations for ALS
def matrix_decomposition(matrix, k):
    matrix = pd.DataFrame(matrix)
    sc = SparkContext(master="local[4]")
    sqlCtx = ps.SQLContext(sc)
    spark_df = sqlCtx.createDataFrame(matrix)
    model = ALS.train(spark_df, rank=k, seed=42069)
    x = model.userFeatures().toDF(['user_id', 'values']).select('values').toPandas()
    y = model.productFeatures().toDF(['item_id', 'values']).select('values').toPandas()

    user_latent = []
    temp = []
    for line in x.values:
        user_latent.append(np.array(line[0]))
    user_latent = np.array(user_latent)
    item_latent =[]
    for line in y.values:
        item_latent.append((np.array(line[0])))
    item_latent = np.array(item_latent)
    print(len(user_latent), len(item_latent))

    # matrix_formatted = sparse.csr_matrix(matrix.T)
    # model = AlternatingLeastSquares(factors=k, iterations=i)
    # model.fit(matrix_formatted)
    # user_latent = model.user_factors
    # item_latent = model.item_factors
    #
    #
    # test = np.dot(user_latent, item_latent.T)
    sc.stop()
    return user_latent, item_latent


# Creates data matrix to be used in factorization
def create_data_matrix(data, num_users, num_items):
    data_matrix = np.zeros((num_users, num_items))
    for line in data.itertuples():
        data_matrix[line[1] - 1, line[2] - 1] = line[3]
    return data_matrix


# Creates the distance matrices I.E: (User, User), (Item, Item)
def create_distance_matrix(x, y=None):
    if y:
        return euclidean_distances(x, y)
    else:
        return euclidean_distances(x, x)


# Creates a predictor of the user scores based off weighted sum formula. Need to transpose data_matrix if based on item
# distance
def create_predictor_matrix(data_matrix, dist_matrix):
    if data_matrix.shape[0] != dist_matrix.shape[0]:
        print("ERROR! Matrices have different dimensions:{} and {}".format(data_matrix.shape[0], dist_matrix.shape[0]))
        exit()
    num_users = data_matrix.shape[0]
    num_items = data_matrix.shape[1]
    pred_matrix = np.zeros((num_users, num_items))
    for i in range(num_users):
        for j in range(num_items):
            num_ratings = len(np.array(np.nonzero(data_matrix[:, j]))[0])
            pred_matrix[i, j] = (np.sum(data_matrix[:, j] / (dist_matrix[i, :] + 1)))/(num_ratings+1)
    return pred_matrix


# Predicts top n movies for a specific user. Uses user labels and item labels generated by kmeans, and gives boost to
# movies with the same labels as the user by adding that user's avg rating
def recommend(pred_matrix, user, user_ratings, item_info, user_labels, item_labels, num_recommend=5):
    avg_rating = np.mean(pred_matrix[user, :])
    matrix = np.copy(pred_matrix)
    matrix[user, item_labels == user_labels[user]] += avg_rating/2
    already_rated = user_ratings[user]
    matrix[user, already_rated] = -10000
    top_movies_index = np.argsort(matrix[user])[-num_recommend:][::-1]
    return item_info[top_movies_index]


def recommend_test(pred_matrix, user, user_ratings, item_info, num_recommend=5):
    already_rated = user_ratings[user]
    matrix = np.copy(pred_matrix)
    matrix[user, already_rated] = -10000
    top_movies_index = np.argsort(matrix[user])[-num_recommend:][::-1]
    return item_info[top_movies_index]


def top_rated(matrix, user, item_info, num_rated=5):
    user_rated=np.argsort(matrix[user])[-num_rated:][::-1]
    return item_info[user_rated]


#############################
## Clustering and Analysis ##
#############################


# Uses k-means algorithm to cluster users based off latent values
def k_means(latent_matrix):
    kmeans = KMeans(n_clusters=10, random_state=42069)
    kmeans.fit(latent_matrix)
    return kmeans.labels_, kmeans.cluster_centers_


# Finds what user cluster a group of items is closest to
def find_clusters(matrix, centroids):
    return np.argmin(cdist(matrix, centroids, 'sqeuclidean'), axis=1)


# Finds what movies a particular user has rated
def movies_rated_by_user(raw_data, item_info, user):
    user_data = raw_data[raw_data.user_id == user]
    user_movies = user_data.item_id
    return item_info[user_movies]


# Finds the most popular movies in a cluster by adding up the ratings for all users in a cluster and dividing each
# rating by the number of people who have seen the movie to get the avg rating per cluster
def popular_movies(data_matrix, cluster_labels, item_info):
    num_clusters = len(set(cluster_labels))
    most_pop = {}
    for cluster in range(num_clusters):
        cluster_matrix = data_matrix[cluster_labels == cluster, :]
        # num_in_cluster = len(data_matrix[cluster_labels==cluster])
        # cluster_scores = [np.sum(x) for x in cluster_matrix.T]
        # cluster_scores_avg = np.array(cluster_scores)/num_in_cluster
        # movies_index = np.argsort(cluster_scores_avg).tolist()
        # movies_index.reverse()
        # movies = item_info[movies_index]
        movie_rate_count = np.count_nonzero(cluster_matrix, axis=0)
        movie_total_rating = np.sum(cluster_matrix, axis=0)
        movie_avg_rating = movie_total_rating/movie_rate_count
        movie_indexes = np.argsort(movie_rate_count).tolist()[::-1][:10]
        most_pop["cluster {}".format(cluster + 1)] = item_info[movie_indexes]
    return most_pop


def median_ratings(data_matrix):
    num_ratings = np.count_nonzero(data_matrix, axis=1)
    return np.median(num_ratings)


###########################
###### Running model ######
###########################

def run_model(raw_data, item_data):
    data_matrix = create_data_matrix(raw_data, 943, 1682)
    user_latent, item_latent = matrix_decomposition(raw_data, 100, 7)
    user_dist_matrix = create_distance_matrix(user_latent)
    user_pred = create_predictor_matrix(data_matrix, user_dist_matrix)


##########################
#### Model Validation ####
##########################


def train_matrix(data, method="cv"):
    rmse = math.inf
    data_matrix = None
    best_k = 0
    for k in range(8,20):
            if method == "cv":
                temp_rmse = cross_validation(data, k, 3)
                print("for K ={}, the RMSE = {}".format(k, temp_rmse))
                if temp_rmse < rmse:
                    rmse = temp_rmse
                    best_k = k
    print("best K: {}\nRMSE: {}".format(best_k, rmse))


def cross_validation(data, k, splits):
    kf = KFold(n_splits=splits)
    rmse_avg = 0
    for train, test in kf.split(data):
        print("Train: ", train, " Test: ", test)

        train_matrix = create_data_matrix(data.iloc[test, :], 943, 1682)
        train_data = np.zeros(((943*1682),3))
        count = 0
        for x in range(943):
            for y in range(1682):
                train_data[count, 0] = x
                train_data[count, 1] = y
                train_data[count, 2] = train_matrix[x, y]
                count += 1
        test_matrix = create_data_matrix(data.iloc[test, :], 943, 1682)
        train_user_latent, train_item_latent = matrix_decomposition(train_data, k)

        rmse_avg += rmse(np.dot(train_user_latent, train_item_latent.T), test_matrix)
        print("RMSE is: ", rmse(np.dot(train_user_latent, train_item_latent.T), test_matrix))

    return rmse_avg/splits


# Code for rmse found from:
# https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html
# Modified slightly to add the .tolist()[0] because of weird formatting errors with the prediction matrix. Should
# probably look into how to fix that but idk
# def rmse_formatted(prediction, ground_truth):
#     x = np.array(ground_truth.nonzero())
#     prediction = prediction[ground_truth.nonzero()].flatten().tolist()[0]
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     return sqrt(mean_squared_error(prediction, ground_truth))

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


if __name__ == "__main__":
    movie_data_loc, movie_data_header = 'ml-100k/u.data', ['user_id', 'item_id', 'rating', 'timestamp']
    movie_info_loc, movie_info_header = 'ml-100k/u.item', ['item_id', 'movie_title', "lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol","lol"]

    movie_data = (movie_data_loc, movie_data_header, movie_info_loc, movie_info_header)

    raw_data, item_info = import_format_data(movie_data)
    # train_matrix(raw_data)
    # exit()

    data_matrix = create_data_matrix(raw_data, 943, 1682)

    user_rated = find_user_ratings(data_matrix)

    user_latent, item_latent = matrix_decomposition(raw_data, 10)

    print("The median number of ratings for the matrix is: {} \nAs percent of total movies: {}"
          .format(median_ratings(data_matrix), median_ratings(data_matrix) / data_matrix.shape[1]))
    user_labels, user_centers = k_means(user_latent)
    # test_0 = np.array(user_latent[labels == 0])
    # test_1 = np.array(user_latent[labels == 1])
    # test_2 = np.array(user_latent[labels == 2])
    # test_3 = np.array(user_latent[labels == 3])
    #
    # item_labels, item_centers = k_means(item_latent)
    # item_test_00 = np.array(item_latent[item_labels == 0])
    # item_test_01 = np.array(item_latent[item_labels == 1])
    # item_test_02 = np.array(item_latent[item_labels == 2])
    # item_test_03 = np.array(item_latent[item_labels == 3])
    item_labels = find_clusters(item_latent, user_centers)

    predicted_matrix = np.dot(user_latent, item_latent.T)
    print("\n", "The RMSE when multiplying latent matrices is: {}".format(rmse(predicted_matrix, data_matrix)))

    user_dist_matrix = create_distance_matrix(user_latent)
    item_dist_matrix = create_distance_matrix(item_latent)


    start_time = time.time()
    user_pred = create_predictor_matrix(data_matrix, user_dist_matrix)
    PRED_COPY = user_pred
    print("--- %s seconds ---" % (time.time() - start_time))

    print(recommend(user_pred, 1, user_rated, item_info, user_labels, item_labels))
    print(top_rated(data_matrix, 1, item_info))
    print("-------------------------------------------------------------")
    print(recommend_test(user_pred, 1, user_rated, item_info))
    print("\n")
    print(recommend(user_pred, 2, user_rated, item_info, user_labels, item_labels))
    print(top_rated(data_matrix, 2, item_info))
    print("-------------------------------------------------------------")
    print(recommend_test(user_pred, 2, user_rated, item_info))
    print("\n")
    print(recommend(user_pred, 3, user_rated, item_info, user_labels, item_labels))
    print(top_rated(data_matrix, 3, item_info))
    print("-------------------------------------------------------------")
    print(recommend_test(user_pred, 3, user_rated, item_info))