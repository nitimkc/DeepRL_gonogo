
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse

import numpy as numpy


def process_session_data(session_data):
    """
    Process session data to generate variables for trials, time steps, and observation matrix.

    Parameters:
    session_data (pd.DataFrame): The session data table.

    Returns:
    tuple: A tuple containing N_TRIALS, N_TIMESTEPS, CHANGE_TIME, TARGET_TIME, and observation_matrix.
    """
    N_TRIALS = len(session_data)
    
    # N_TIMESTEPS varies each trial, make a numpy array of the num_time_bins column in session_data
    N_TIMESTEPS = session_data['num_time_bins'].to_numpy().astype(int)
    # Check the shape of N_TIMESTEPS
    print(f"N_TIMESTEPS shape: {N_TIMESTEPS.shape}")
    print(f"First few values of N_TIMESTEPS: {N_TIMESTEPS[:10]}")

    CHANGE_TIME = session_data['change_time_bin'].to_numpy().astype(int)
    TARGET_TIME = session_data['response_time_bin'].to_numpy()

    print(f"CHANGE_TIME shape: {CHANGE_TIME.shape}")
    print(f"TARGET_TIME shape: {TARGET_TIME.shape}")

    # Get unique values of initial_image_name and map them to one-hot encoding
    unique_images = session_data['initial_image_name'].unique()
    image_to_index = {image: index for index, image in enumerate(unique_images)}
    initial_images_one_hot = np.zeros((N_TRIALS, len(unique_images)), dtype=int)
    for i, image in enumerate(session_data['initial_image_name']):
        index = image_to_index[image]
        initial_images_one_hot[i, index] = 1

    # Do the same for change_image_name
    unique_change_images = session_data['change_image_name'].unique()
    image_to_index_change = {image: index for index, image in enumerate(unique_images)}
    change_images_one_hot = np.zeros((N_TRIALS, len(unique_change_images)), dtype=int)
    for i, image in enumerate(session_data['change_image_name']):
        index = image_to_index_change[image]
        change_images_one_hot[i, index] = 1

    print(initial_images_one_hot.shape)
    print(change_images_one_hot.shape)

    print("Initial Images One-Hot Encoding (first 5 rows):")
    print(initial_images_one_hot[:5])
    print("Change Images One-Hot Encoding (first 5 rows):")
    print(change_images_one_hot[:5])

    # Initialize the observation matrix
    observation_matrix = np.zeros((N_TRIALS, max(N_TIMESTEPS), len(unique_images)), dtype=int)

    for i in range(N_TRIALS):
        # Fill the initial images up to CHANGE_TIME
        if CHANGE_TIME[i] > 0:  # Ensure CHANGE_TIME is valid
            observation_matrix[i, :CHANGE_TIME[i], :] = initial_images_one_hot[i, :]
        # Fill the change images from CHANGE_TIME onwards
        if CHANGE_TIME[i] < N_TIMESTEPS[i]:  # Ensure CHANGE_TIME is within bounds
            observation_matrix[i, CHANGE_TIME[i]:N_TIMESTEPS[i], :] = change_images_one_hot[i, :]

    print(f"Observation matrix shape: {observation_matrix.shape}")
    print(observation_matrix[1, :20, :])  # Display a trial and first 10 time steps

    return N_TRIALS, N_TIMESTEPS, CHANGE_TIME, TARGET_TIME, observation_matrix


N_TRIALS, N_TIMESTEPS, CHANGE_TIME, TARGET_TIME, observation_matrix = process_session_data(training_data)