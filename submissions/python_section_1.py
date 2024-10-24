from typing import Dict, List

import pandas as pd
#Q1
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    for i in range(0, len(lst), n):
        end = min(i + n, len(lst))
        
        for j in range((end - i) // 2):
            lst[i + j], lst[end - 1 - j] = lst[end - 1 - j], lst[i + j]

    return lst  
#2
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    length_dict = {}
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    sorted_dict = dict(sorted(length_dict.items()))
    return sorted_dict

#Q3
def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    items = {}
    def flatten(current_dict: Dict, parent_key: str = ''):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                flatten(value, new_key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flatten(item, f"{new_key}[{i}]")
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = value
    
    flatten(nested_dict)
    return items



#Q4
def unique_permutations(nums: List[int]) -> List[List[int]]:
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] not in seen:
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]  
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]  
    result = []
    nums.sort()  
    backtrack(0)
    return result
#Q5
import re
def find_all_dates(text: str) -> List[str]:
    patterns = [
        r'\b(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})\b',  
        r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b', 
        r'\b(\d{4})\.(0[1-9]|1[0-2])\.(0[1-9]|[12][0-9]|3[01])\b'   
    ]
    
    dates = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                if len(match) == 3:
                    if '-' in pattern:
                        dates.append(f"{match[0]}-{match[1]}-{match[2]}")
                    elif '/' in pattern:
                        dates.append(f"{match[0]}/{match[1]}/{match[2]}")
                    elif '.' in pattern:
                        dates.append(f"{match[0]}.{match[1]}.{match[2]}")
    
    return dates
#Q6
import pandas as pd
import polyline
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371000  
    return c * r

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    distances = [0]  
    for i in range(1, len(df)):
        distance = haversine(df.latitude[i-1], df.longitude[i-1], df.latitude[i], df.longitude[i])
        distances.append(distance)
    df['distance'] = distances
    return df

#Q7
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    result_mat = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i]) - rotated[i][j]
            col_sum = sum(rotated[k][j] for k in range(n)) - rotated[i][j]
            result_mat[i][j] = row_sum + col_sum
    return result_mat

#8
import pandas as pd
df = pd.read_csv("dataset-2.csv")
def time_check(df: pd.DataFrame) -> pd.Series:
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')
    grouped = df.groupby(['id', 'id_2'])
    def check_completeness(group):
        if group['start_datetime'].isnull().any() or group['end_datetime'].isnull().any():
            return False 
        unique_days = group['start_datetime'].dt.date.unique()
        day_count = len(unique_days)
        full_24_hour = (
            group['start_datetime'].min().normalize() == group['start_datetime'].dt.date[0] and
            group['end_datetime'].max().normalize() == group['end_datetime'].dt.date[0]
        )
        
        return full_24_hour and (day_count == 7)
    completeness_results = grouped.apply(check_completeness)
    return completeness_results
