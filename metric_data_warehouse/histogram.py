def create_histogram(data_points, bins):
    histogram = {}

    # Iterate through each data point
    for point in data_points:
        bin_indices = tuple()
        for i in range(len(point)):
            # Find the index of the bin in the current dimension
            bin_index = min(range(len(bins[i]) - 1), key=lambda j: abs(bins[i][j] - point[i]))
            bin_indices += (bin_index,)

        # Update histogram
        if bin_indices in histogram:
            histogram[bin_indices] += 1
        else:
            histogram[bin_indices] = 1

    return histogram

# Example data points
data_points = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (3, 5, 7), (2, 4, 6), (1, 2, 3)]

# Example bin definitions for each dimension (X, Y, Z)
bins = [
    [0, 2, 4, 6, 8, 10],  # X bins
    [0, 3, 6, 9, 12],      # Y bins
    [0, 4, 8, 12, 16, 20]  # Z bins
]

# Create histogram
histogram = create_histogram(data_points, bins)

# Print histogram
print("Histogram:")
for bin_indices, count in histogram.items():
    print("Bin indices:", bin_indices, "Count:", count)

