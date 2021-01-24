import pandas as pd
import math
from collections import Counter

def entropy(attr_subset):
    count = Counter([x for x in attr_subset])
    attr_subset_length = len(attr_subset)
    prob = [x/attr_subset_length for x in count.values()]
    return sum([(-x * math.log(x, 2)) for x in prob])
    

def information_gain(dataset, concept, attr):
    subset = dataset.groupby(attr)
    dataset_length = len(dataset.index)
    average_information_table = subset.agg({concept : [entropy, lambda x: len(x)/dataset_length] })[concept]
    average_information_table.columns = ["Entropy", "pa+na/p+n"]
    average_information = sum(average_information_table["Entropy"] * average_information_table["pa+na/p+n"])
    gain = entropy(dataset[concept]) - average_information
    return gain

def id3(dataset, concept, decision_attributes):
    count = Counter(x for x in dataset[concept])
    if len(count) == 1:
        return next(iter(count))
    else:
        gain_list = [information_gain(dataset, concept, attr) for attr in decision_attributes]
        print("Gain list for " + ",".join(decision_attributes))
        print(gain_list)
        best_attribute = decision_attributes[gain_list.index(max(gain_list))]
        print("Best attribute:" + best_attribute)
        tree = {best_attribute: {}}
        remaining_attributes = [attr for attr in decision_attributes if attr != best_attribute]
        for attribute_value, attribute_subset in dataset.groupby(best_attribute):
            subtree = id3(attribute_subset, concept, remaining_attributes)
            tree[best_attribute][attribute_value] = subtree
        return tree

dataset = pd.read_csv("datasets/3.csv")

print("The dataset is")
print(dataset)

print("List of decision attributes are")
decision_attributes = list(dataset.columns)
decision_attributes.remove("PlayTennis")
print(decision_attributes)

tree = id3(dataset, "PlayTennis", decision_attributes)

print("The decision tree is")
print(tree)