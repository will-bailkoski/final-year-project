import numpy as np

results = [
[2, 3, 1, 1, 1, 1, 1, 1, 1, 1],
[2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[2, 0, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 3, 1, 1, 1, 1, 1, 1, 1, 1],
[2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 2, 0, 0, 0, 0, 0, 0, 0, 0]
]

print(4**-0.5)

def normalize_vector(vector):
    # Normalize each dimension to be in the range [0, 1]
    return (np.array(vector) - 0) / 4

def euclidean_distance(vector1, vector2):
    normalized_vector1 = normalize_vector(vector1)
    normalized_vector2 = normalize_vector(vector2)
    return np.linalg.norm(normalized_vector1 - normalized_vector2)


datapoints = np.array([10, 10, 50, 50, 100, 100, 500, 500, 1000, 2500, 5000, 10000, 5000, 2000])
differences = [euclidean_distance(np.array([1,0,1,1,1,1,1,1,1,1]), x) for x in results]

import matplotlib.pyplot as plt
from scipy.stats import linregress

# Calculate line of best fit
slope, intercept, r_value, p_value, std_err = linregress(datapoints, differences)
line_of_best_fit = slope * datapoints + intercept

# Plot scatter graph
plt.scatter(datapoints, differences)

# Plot line of best fit
plt.plot(datapoints, line_of_best_fit, color='red', label='Line of Best Fit')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()