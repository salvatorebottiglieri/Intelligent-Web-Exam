from math import e
import pandas as pd
from src.Utils import max_timestamp, minimum_timestamp, rating_time
import numpy as np
from typing import Dict, List
from src.WeightsComputation import compute_similarity
import math


def time_impact(user:int, item:int, dataset:pd.Dataframe)-> float:
    e * ( (minimum_timestamp(user, dataset) + rating_time(user, item, dataset)) / (max_timestamp(user, dataset) - minimum_timestamp(user, dataset)) )

def create_user_item_matrix():
    # Create a user-item matrix
    ratings = np.array([[3, 4, 0, 3, 5],
                    [4, 5, 3, 4, 0],
                    [0, 3, 4, 0, 4],
                    [5, 4, 3, 5, 4]])
    return ratings

def similarity_matrix(ratings):
    # Calculate the similarity matrix using cosine similarity
    similarity = np.zeros((ratings.shape[0], ratings.shape[0]))
    for i in range(ratings.shape[0]):
        for j in range(ratings.shape[0]):
            if i == j:
                similarity[i][j] = 1
            else:
                dot_product = np.dot(ratings[i], ratings[j])
                norm_i = np.linalg.norm(ratings[i])
                norm_j = np.linalg.norm(ratings[j])
                similarity[i][j] = dot_product / (norm_i * norm_j)
    return similarity

def find_most_similar_users(ratings, similarity):
    # Find the k most similar users for each user
    k = 2
    nearest_neighbors = []
    for i in range(ratings.shape[0]):
        nn = np.argsort(similarity[i])[::-1][:k]
        nearest_neighbors.append(nn)

    # Predict the user's rating for unrated items
    predictions = np.zeros((ratings.shape[0], ratings.shape[1]))
    for i in range(ratings.shape[0]):
        for j in range(ratings.shape[1]):
            if ratings[i][j] == 0:
                numerator = 0
                denominator = 0
                for n in nearest_neighbors[i]:
                    numerator += similarity[i][n] * ratings[n][j]
                    denominator += similarity[i][n]
                if denominator == 0:
                    predictions[i][j] = 0
                else:
                    predictions[i][j] = numerator / denominator
            else:
                predictions[i][j] = ratings[i][j]
    return nearest_neighbors, predictions

def generate_recommendations(ratings, predictions):
    # Generate recommendations for each user
    recommendations = []
    for i in range(ratings.shape[0]):
        rec = np.argsort(predictions[i])[::-1]
        recommendations.append(rec)

    # Print the recommendations for each user
    for i in range(ratings.shape[0]):
        print(f"User {i+1} recommendations: {recommendations[i]}")





def weighted_sum_of_others_ratings(user_item_graph, active_user, item):
    r_u = user_item_graph.get_average_rating(active_user)
    u = user_item_graph.get_similar_users(active_user)
    numerator = 0
    denominator = 0
    for v in u:
        if item in user_item_graph.get_rated_items(v):
            r_v = user_item_graph.get_average_rating(v)
            similarity = compute_similarity(user_item_graph, active_user, v)
            numerator += similarity * (user_item_graph.get_rating(v, item) - r_v)
            denominator += similarity
    if denominator == 0:
        return r_u
    else:
        return r_u + numerator / denominator


def weighted_sum_of_others_ratings_with_time(user_item_graph, active_user, item):
    r_u = user_item_graph.get_average_rating(active_user)
    u = user_item_graph.get_similar_users(active_user)
    numerator = 0
    denominator = 0
    for v in u:
        if item in user_item_graph.get_rated_items(v):
            r_v = user_item_graph.get_average_rating(v)
            similarity = compute_similarity(user_item_graph, active_user, v)
            time_impact_value = time_impact(user_item_graph, v, item)
            numerator += similarity * time_impact_value * (user_item_graph.get_rating(v, item) - r_v)
            denominator += similarity * time_impact_value
    if denominator == 0:
        return r_u
    else:
        return r_u + numerator / denominator
