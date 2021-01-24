import csv
import math
import random
import statistics

def gaussian_probability_distribution(x, u, sigma):
    exponent = math.exp(-(math.pow(x - u, 2) / (2 * math.pow(sigma, 2))))
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * exponent

with open("datasets/5.csv", "r") as csv_file:
    dataset = []
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        dataset.append([float(col) for col in row])

dataset_size = len(dataset)

print("The size of the dataset is %d"%dataset_size)

train_size = int(0.7 * dataset_size)
print("The size for training is %d"%train_size)

train_data = []
test_data = dataset.copy()

train_data_indexes = random.sample(range(dataset_size), train_size)

for index in train_data_indexes:
    train_data.append(dataset[index])
    test_data.remove(dataset[index])

target_classes = {}

for data in train_data:
    target_class = int(data[-1])
    if target_class not in target_classes:
        print("Target class: %d"%target_class)
        target_classes[target_class] = []
    target_classes[target_class].append(data)

mean_variances = {}

for target_class, data in target_classes.items():
    mean_variance = [(statistics.mean(attribute), statistics.stdev(attribute)) for attribute in zip(*data)]
    del mean_variance[-1]
    mean_variances[target_class] = mean_variance

predictions = []

for data in test_data:
    probabilities = {}
    for target_class, mean_variance in mean_variances.items():
        probabilities[target_class] = 1
        for index, values in enumerate(mean_variance):
            probabilities[target_class] *= gaussian_probability_distribution(data[index], values[0], values[1])
    best_label, best_prob = None, -1
    for target_class, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = target_class
    predictions.append(best_label)


correct = 0
for index, key in enumerate(test_data):
    if predictions[index] == test_data[index][-1]:
        correct += 1

accuracy = (correct/(float(len(test_data)))) * 100
print("Accuracy: %f"%accuracy)