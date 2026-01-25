""" 
20260124
Problem:
    Tom wants to know the minimum average speed of 60 second of his everyday running practice. 
    He runs 5km daily, and he can get the speed data on his watch every 5 seconds, for like 32m, 34m, 40m, and etc.

Deconstruction:
    Each number is distance in meters during that 5-second interval.
    A 60-second window contains 12 samples (60 / 5 = 12).
    Average speed over that window = (sum(distance_in_12_samples)) / 60.

So the problem now reduces to:
    Find the minimum sum of any consecutive 12 values, then divide by 60.
    
    So, no deque needed. A simple sliding window is best.
"""


from typing import List, Tuple 


def min_avg_speed_60(distance_per_5s: List[float]) -> float:
    """ 
    Each element is distance (meters) covered in a 5-second interval.
    Returns minimum average speed (m/s) over any 60-second window.
    """
    
    window_secs = 60
    step_secs = 5
    k = window_secs // step_secs # 12
    
    n = len(distance_per_5s)
    if n < k:
        raise ValueError("Need at leat 60 seconds of data (12 samples).")
    
    window_sum = sum(distance_per_5s[:k])
    min_sum = window_sum
    
    for i in range(k, n):
        window_sum += distance_per_5s[i] - distance_per_5s[i-k]
        if window_sum < min_sum:
            min_sum = window_sum
    
    return min_sum / window_secs # ms/


data = [32, 34, 40, 33, 31, 29, 30, 28, 35, 36, 34, 33, 20, 22]
print(min_avg_speed_60(data))