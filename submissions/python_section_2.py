#Q9
import pandas as pd
import numpy as np

def calculate_distance_matrix(df):
    
    # Create a unique list of IDs
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))

    # Create an empty distance matrix
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)

    # Fill the distance matrix with known distances
    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    # Set the diagonal to 0 (distance to self)
    np.fill_diagonal(distance_matrix.values, 0)

    # Calculate cumulative distances using Floyd-Warshall algorithm
    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix
#q10
def unroll_distance_matrix(distance_matrix):
    
    # Create an empty list to store the results
    results = []

    # Iterate through the distance matrix to unroll it
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:  # Exclude same id_start and id_end
                distance = distance_matrix.at[id_start, id_end]
                results.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the results list
    unrolled_df = pd.DataFrame(results)

    return unrolled_df

# Load the dataset from CSV
file_path = "dataset-2.csv"
df = pd.read_csv(file_path)

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(df)

# Display the distance matrix
print("Distance Matrix:")
print(distance_matrix)

# Unroll the distance matrix
unrolled_df = unroll_distance_matrix(distance_matrix)

# Display the unrolled DataFrame
print("Unrolled Distance DataFrame:")
print(unrolled_df)
#Q11
def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance for each id_start
    avg_distances = unrolled_df.groupby('id_start')['distance'].mean().reset_index()
    avg_distances.columns = ['id', 'avg_distance']

    # Get the average distance of the reference ID
    ref_avg_distance = avg_distances[avg_distances['id'] == reference_id]['avg_distance'].values[0]

    # Calculate the 10% threshold
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1

    # Find all IDs whose average distance is within the 10% threshold
    ids_within_threshold = avg_distances[(avg_distances['avg_distance'] >= lower_bound) & 
                                         (avg_distances['avg_distance'] <= upper_bound)]

    return ids_within_threshold

# Let's pick a reference ID from the dataset to use for this question
reference_id = 1001400

# Find the IDs within 10% of the reference ID's average distance
ids_within_threshold_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

# Display the result
ids_within_threshold_df
#Q12
def calculate_toll_rate(unrolled_df):
    """
    Calculate toll rates based on the unrolled DataFrame. 
    We assume the toll is a simple function of distance (e.g., $1 per km).

    Args:
        unrolled_df (pd.DataFrame): DataFrame with columns 'id_start', 'id_end', and 'distance'.

    Returns:
        pd.DataFrame: DataFrame with id_start, id_end, distance, and toll_rate columns.
    """
    # Assume a toll rate of $1 per km for simplicity
    unrolled_df['toll_rate'] = unrolled_df['distance'] * 1.0  # Adjust the multiplier as per actual toll rate logic
    
    return unrolled_df

# Calculate toll rates
toll_rate_df = calculate_toll_rate(unrolled_df)

# Display the first few rows with toll rates
print(toll_rate_df.head())
#Q13
import numpy as np

def calculate_time_based_toll_rates(unrolled_df):
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Create a mock 'time' column to simulate different times of day (for example, in 24-hour format)
    np.random.seed(0)  # For reproducible random times
    unrolled_df['time'] = np.random.randint(0, 24, size=len(unrolled_df))  # Random hours of the day

    # Define toll rates based on time intervals
    def get_toll_rate_by_time(time):
        if 7 <= time <= 9 or 17 <= time <= 19:
            return 1.5  # Peak hours (7-9 AM, 5-7 PM) have a higher toll rate
        elif 9 < time <= 16:
            return 1.0  # Midday (9 AM - 4 PM) has a normal toll rate
        else:
            return 0.8  # Off-peak hours (before 7 AM or after 7 PM)

    # Apply the time-based toll rate
    unrolled_df['time_based_toll_rate'] = unrolled_df['time'].apply(get_toll_rate_by_time) * unrolled_df['distance']

    return unrolled_df

# Calculate time-based toll rates
time_based_toll_df = calculate_time_based_toll_rates(unrolled_df)

# Display the first few rows with time-based toll rates
print(time_based_toll_df[['id_start', 'id_end', 'distance', 'time', 'time_based_toll_rate']].head())
