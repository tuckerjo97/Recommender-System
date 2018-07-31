import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.metrics import mean_squared_error
from math import sqrt


def try_svd(train_data_matrix, test_data_matrix, with_dist=True, recommend_user=-1, num_recs=5):
    print("Trying to do SVD")
    u, s_array, vh = np.linalg.svd(train_data_matrix, full_matrices=False)
    s_array = np.sqrt(s_array)
    s_diag = np.diag(s_array)
    # Uses built in svd method in numpy to calculate 3 matrices, USV^t. Note: V is already transposed

    k = 10
    # Number of desired latent dimensions

    s_latent = np.copy(s_diag)
    s_latent[s_latent < s_array[k]] = 0.0
    # eliminates all values with low variance, effectively turning columns of U and rows of VH to 0. Currently have
    # k = 10 in order to lower rmse.

    if with_dist:
        uk = np.dot(u, s_latent)
        uk = uk[:, :k]
        u_size = uk.shape[0]
        u_dist_matrix = np.zeros((u_size, u_size))
        dist_sum = 0
        for i in range(u_size):
            for j in range(u_size):
                u_dist_matrix[i, j] = sqrt(np.sum(((uk[i] - uk[j])**2)))
                dist_sum += u_dist_matrix[i, j]
        # Creates a matrix for the vector distances between every possible pair of users. Used to weight each users
        # rating for prediction

        dist_sum = dist_sum/2
        # Use the sum of all the distances to normalize the predicted scores

        x_pred = np.zeros(train_data_matrix.shape)
        for x in range(train_data_matrix.shape[0]):
            for y in range(train_data_matrix.shape[1]):
                item = train_data_matrix[:, y]
                item = np.sum(item / (u_dist_matrix[x] + 1))
                x_pred[x, y] = item/dist_sum
                # for each user, sums every rating by every user for an item, multiplied by the distance between user
                # vectors in latent space. Currently gets rmse of 3.7 after dividing each predicted item score by the
                # sum of user vector distances in order to normalize item score
    else:
        x_pred = np.dot(np.dot(u, s_latent), vh)

    if recommend_user >= 0 and recommend_user < train_data_matrix.shape[0]:
        return recommend(x_pred, train_data_matrix, recommend_user, num_recs)
        # if a specific user index is added, will return highest recommended item index for user

    x_rmse = rmse(x_pred, test_data_matrix)
    print('User-based CF MSE: ' + str(x_rmse))
    return x_rmse
    # Find where variance is roughly 95% and shrinks matrix U to K rows


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
    # Code for rmse found from:
    # https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html


def cross_validation(header, matrix):
    df = pd.read_csv(matrix, sep='\t', names=header)
    num_users = len(df.user_id.unique())
    num_items = len(df.item_id.unique())
    # Uses Matrix string to find data set, for MovieLens data would use 'ml-100k/u.data

    data_matrix = df.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)
    data_matrix = data_matrix.values
    # Convert from a matrix of individual user ratings for single items, to a (num_users, num_items) shape matrix of
    # every rating by a user for every item. in this case, size (943, 1682). Use df.values to convert from a pandas
    # data frame to a np.array, otherwise breaks rmse function

    k = 10
    k_matrices = np.array_split(data_matrix, k)
    # Splits data_matrix into k equal sized matrix. In this case, 21 matrices sized (41, 1682)

    counter = k
    avg_rmse = 0
    while counter > 0:
        train_matrix = np.concatenate(k_matrices[:k-1])
        test_matrix = k_matrices[k-1]
        avg_rmse += try_svd(train_matrix, test_matrix, True)
        kth_matrix = k_matrices.pop()
        k_matrices.insert(0, kth_matrix)
        counter -= 1
    # Concatenates the first k-1 matrices, using the kth matrix as the test matrix. Then rotates kth matrix to the front
    # of the list

    avg_rmse = avg_rmse/k
    print("AVG RMSE: " + str(avg_rmse))


#This fix is actually hot garbage and runs way slower
def cross_validation_fix(header, matrix):
    df = pd.read_csv(matrix, sep='\t', names=header)
    num_users = len(df.user_id.unique())
    num_items = len(df.item_id.unique())

    k = 7
    k_matrices = np.array_split(df, k)
    k_matrices_final = []
    for m in k_matrices:
        temp_matrix = np.zeros((num_users, num_items))
        for line in m.itertuples():
            temp_matrix[line[1] - 1, line[2] - 1] = line[3]
        k_matrices_final.append(temp_matrix)

    counter = k
    avg_rmse = 0
    while counter > 0:
        train_matrix = np.concatenate(k_matrices_final[:k - 1])
        test_matrix = k_matrices_final[k - 1]
        avg_rmse += try_svd(train_matrix, test_matrix)
        kth_matrix = k_matrices_final.pop()
        k_matrices_final.insert(0, kth_matrix)
        counter -= 1

    avg_rmse = avg_rmse / k
    print("AVG RMSE: " + str(avg_rmse))

def bootstrap(header, matrix):
    df = pd.read_csv(matrix, sep='\t', names=header)
    num_users = len(df.user_id.unique())
    num_items = len(df.item_id.unique())

    counter = 0
    avg_rmse = 0
    while counter < 4:
        train_data = df.sample(frac=0.8, replace=True)

        test_data = df[~df.index.isin(train_data.index)]
        # This line was found at:
        # https://stackoverflow.com/questions/28901683/pandas-get-rows-which-are-not-in-other-dataframe/28902170
        # For some reason am getting way to many entries for test matrix

        train_data_matrix = np.zeros((num_users, num_items))
        for line in train_data.itertuples():
            train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
        test_data_matrix = np.zeros((num_users, num_items))
        for line in test_data.itertuples():
            test_data_matrix[line[1] - 1, line[2] - 1] = line[3]
        avg_rmse += try_svd(train_data_matrix, test_data_matrix)
        counter += 1
    avg_rmse = avg_rmse/4
    print("AVG RMSE: " + str(avg_rmse))


def recommend(pred_matrix, actual_matrix, user_num, num_recs):
    print("Finding recommendations")
    user_pred = pred_matrix[user_num]
    user_actual = actual_matrix[user_num]
    user_actual_bool = user_actual.astype(bool)
    rec_list =[]
    while num_recs > 0:
        item_index = np.argmax(user_pred)
        if not user_actual_bool[item_index]:
            rec_list.append(item_index)
            num_recs -=1
            user_pred[item_index]=0
        else:
            user_pred[item_index] = 0
        if np.sum(user_pred) == 0:
            found = True
    return rec_list
    # returns -1 if all items were already rated


def parse_recommend(rec_list, movie_db):
    rec_string = []
    db_as_dict = pd.Series(movie_db.title.values, index=movie_db.new_movie_Id).to_dict()
    for rec in rec_list:
        rec_string.append(db_as_dict[rec])
    return rec_string


# Needed to make because movie ids were not sequential. regenerates movie ids in sequential order and matches up
# original ratings
def reformat_movies(ratings_db, movie_db):
    movie_db['new_movie_Id'] = range(len(movie_db))
    id_dict = dict(zip(movie_db['movieId'], movie_db['new_movie_Id']))
    rating_copy = ratings_db.copy()
    ratings_db['new_movie_Id'] = ratings_db['item_id'].map(id_dict)
    return ratings_db, movie_db


if __name__ == "__main__":
    header = ['user_id', 'item_id', 'rating', "timestamp"]
    matrix = 'ml-latest-small/ratings.csv'
    test = pd.read_csv(matrix, names= header, skiprows=1)

    movie_url = 'ml-latest-small/movies.csv'
    movie_db = pd.read_csv(movie_url)

    pd.set_option('max_colwidth', 1000000)
    pd.set_option('expand_frame_repr', True)
    pd.options.display.max_columns = 20
    movie_db = movie_db.drop("genres", axis=1)

    test, movie_db = reformat_movies(test, movie_db)
    # test = np.array(test.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0))
    # cross_validation(header, matrix)

    test_pivot = test.pivot_table(values="rating", index="user_id", columns="new_movie_Id", fill_value=0)
    test_pivot = test_pivot.values
    user = 1
    movie_list = try_svd(test_pivot, 1, True, user)
    print("recommended movies for user {} are: ".format(user) + str(parse_recommend(movie_list, movie_db)))



