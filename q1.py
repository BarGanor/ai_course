import pandas as pd
import numpy as np
import random
import math
import itertools
from pptree import *
from sklearn.model_selection import train_test_split
import scipy.stats


class Node:
    def __init__(self, entropy, parent, children, num_samples, num_samples_per_class, predicted_class, feature=None):
        self.entropy = entropy
        self.parent = parent
        self.children = children
        self.feature = feature
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.output = None
        self.threshold = None


class DecisionTree:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.is_fit = False
        self.tree = None

    def predict(self, inputs):
        predictions = []
        for i in range(len(inputs)):
            row = inputs.iloc[i]
            node = self.tree
            while len(node.children) > 0:

                if row[node.feature] in node.threshold:
                    node = node.children[0]
                elif len(node.children) > 1:
                    node = node.children[1]
                else:
                    break

            predictions.append(node.predicted_class)

        return predictions

    def plot_tree(self):
        if self.is_fit:
            print_tree(current_node=self.tree, childattr='children', nameattr='output')
        else:
            print('You must fit the algorithm before you can plot a tree')

    def fit(self, data):
        tree = self.grow_tree(data)
        self.prune_tree(tree)
        self.tree = tree
        self.is_fit = True

    def prune_tree(self, node):
        i = 2
        curr_children = node.children
        while len(curr_children) > 0:
            next_gen = []
            for child in curr_children:
                if len(child.children) > 0:
                    if self.is_significant(child):
                        pass
                    else:
                        child.children = []

                next_gen.extend(child.children)

            curr_children = next_gen
            i += 1

    def grow_tree(self, data, depth=0, parent=None):

        num_samples_per_class = data['is_busy'].value_counts()
        predicted_class = num_samples_per_class.idxmax()

        node = Node(entropy=self.get_entropy(data), parent=parent, children=[], num_samples_per_class=num_samples_per_class, num_samples=data.shape[0], predicted_class=predicted_class)

        if len(data['is_busy'].unique()) == 1:
            node.output = node.output = 'Feature: ' + str(node.feature) + '. Split: ' + str(node.threshold) + '. Entropy: ' + str(round(node.entropy, 3))
            return node

        if depth < self.max_depth:
            best_node, best_split, highest_gain = self.best_split(data)

            if best_node is not None and best_split is not None:
                left_split = data[data[best_node].isin(best_split)]
                right_split = data.drop(left_split.index)

                node.feature = best_node
                node.threshold = best_split
                node.output = 'Feature: ' + str(node.feature) + '. Split: ' + str(node.threshold) + '. Entropy: ' + str(round(node.entropy, 3))

                node.children.append(self.grow_tree(left_split.drop(columns=[best_node]), depth=depth + 1, parent=node))
                node.children.append(self.grow_tree(right_split.drop(columns=[best_node]), depth=depth + 1, parent=node))

        node.children = [child for child in node.children if child is not None and child.output is not None]
        return node

    def best_split(self, data):
        best_node, best_split, highest_gain = None, None, None

        for col in data.columns:
            if col == 'is_busy':
                continue

            #   If node is first, assign it as highest
            if highest_gain is None:
                best_split, highest_gain = self.best_attr_combination(col_name=col, current_df=data)
                best_node = col

            else:
                col_split, col_gain = self.best_attr_combination(col_name=col, current_df=data)

                if col_split is None and col_gain is None:
                    continue
                # If the gain of the current node is higher than the highest, assign it as highest
                if col_gain > highest_gain:
                    best_node, best_split, highest_gain = col, col_split, col_gain

        return best_node, best_split, highest_gain

    def best_attr_combination(self, col_name, current_df):
        original_entropy = self.get_entropy(current_df)
        stuff = current_df[col_name].unique()

        if len(stuff) == 1:
            return None, None

        best_split = None
        highest_gain = None
        # Iterate over all possible splits to find the best
        for L in range(0, len(stuff) + 1):
            for subset in itertools.combinations(stuff, L):
                # Entropy to subtract
                to_subtract = 0

                subset = list(subset)

                if 0 < len(subset) < len(stuff):

                    left_split = current_df[current_df[col_name].isin(subset)]
                    right_split = current_df.drop(left_split.index)

                    for split in [left_split, right_split]:
                        prob = (split.shape[0] / current_df.shape[0])
                        to_subtract += prob * self.get_entropy(split)

                    info_gain = original_entropy - to_subtract

                    if highest_gain is None or highest_gain < info_gain:
                        highest_gain = info_gain
                        best_split = subset

        return best_split, highest_gain

    @staticmethod
    def get_entropy(df):
        target_value_count = df['is_busy'].value_counts()

        # If the separation is p
        if len(target_value_count) == 1:
            return 0

        p_plus = target_value_count[1] / df['is_busy'].value_counts().sum()
        p_minus = target_value_count[0] / df['is_busy'].value_counts().sum()

        plus_product = -(p_plus * math.log(p_plus, 2))
        minus_product = -(p_minus * math.log(p_minus, 2))

        entropy = plus_product + minus_product
        return entropy

    import scipy.stats

    def is_significant(self, child):

        expected_1 = child.num_samples_per_class / child.num_samples * child.children[0].num_samples
        actual_1 = child.children[0].num_samples_per_class

        if len(child.children) == 2:
            expected_2 = child.num_samples_per_class / child.num_samples * child.children[1].num_samples
            actual_2 = child.children[1].num_samples_per_class
        else:
            expected_2 = child.num_samples_per_class / child.num_samples * (child.num_samples - child.children[0].num_samples)
            actual_2 = child.num_samples_per_class - child.children[0].num_samples_per_class

        chi_1 = ((actual_1 - expected_1).pow(2) / expected_1).pow(1 / 2)
        chi_2 = ((actual_2 - expected_2).pow(2) / expected_2).pow(1 / 2)

        chi_st = (chi_1 + chi_2).sum()
        chi_cr = scipy.stats.chi2.ppf(q=0.95, df=2)

        return chi_st > chi_cr

    def tree_error(self, k):
        pass

    def is_busy(self, row_input):
        pass


def process_data(data):
    # Split the date column to multiple columns
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data = data.drop(columns=['Date'])

    # Add a binary column
    data['is_busy'] = np.where(data['Rented Bike Count'] >= 650, 1, 0)

    # Change temperature to categorical
    temp_bins = [data['Temperature(°C)'].min() - 1, 0, 10, 20, 30, data['Temperature(°C)'].max()]
    data['Temperature(°C)'] = pd.cut(data['Temperature(°C)'], bins=temp_bins, labels=['very cold', 'cold', 'medium', 'hot', 'very hot'])

    # Change humidity to categorical
    humid_bins = [data['Humidity(%)'].min() - 1, 30, 45, 60, 80, data['Humidity(%)'].max()]
    data['Humidity(%)'] = pd.cut(data['Humidity(%)'], bins=humid_bins, labels=['very low humidity', 'low humidity', 'medium humidity', 'humid', 'very humid'])

    # Change wind speed to categorical
    wind_bins = [data['Wind speed (m/s)'].min() - 1, 1, 2, 3, data['Visibility (10m)'].max()]
    data['Wind speed (m/s)'] = pd.cut(data['Wind speed (m/s)'], bins=wind_bins, labels=['very weak wind', 'medium wind', 'strong wind', 'very strong wind'])

    # Change visibility to categorical
    visibility_bins = [data['Visibility (10m)'].min() - 1, 500, 1500, 1800, data['Visibility (10m)'].max()]
    data['Visibility (10m)'] = pd.cut(data['Visibility (10m)'], bins=visibility_bins, labels=['very bad visibility', 'bad visibility', 'medium visibility', 'good visibility'])

    # Change dew point to categorical
    dew_point_bins = [data['Dew point temperature(°C)'].min() - 1, -5, 5, 15, data['Dew point temperature(°C)'].max()]
    data['Dew point temperature(°C)'] = pd.cut(data['Dew point temperature(°C)'], bins=dew_point_bins, labels=['very low dew point', 'low dew point', 'medium dew point', 'high dew point'])

    # Change solar radiation to categorical
    solar_bins = [data['Solar Radiation (MJ/m2)'].min() - 1, .5, 1, 1.5, data['Solar Radiation (MJ/m2)'].max()]
    data['Solar Radiation (MJ/m2)'] = pd.cut(data['Solar Radiation (MJ/m2)'], bins=solar_bins, labels=['very low dew point', 'low dew point', 'medium dew point', 'high dew point'])

    # Change rainfall to categorical
    rain_bins = [data['Rainfall(mm)'].min() - 1, 0, 1, data['Rainfall(mm)'].max()]
    data['Rainfall(mm)'] = pd.cut(data['Rainfall(mm)'], bins=rain_bins, labels=['no rain', 'rain', 'heavy rain'])

    # Change snowfall to categorical
    snow_bins = [data['Snowfall (cm)'].min() - 1, 0, 1, data['Snowfall (cm)'].max()]
    data['Snowfall (cm)'] = pd.cut(data['Snowfall (cm)'], bins=snow_bins, labels=['no snow', 'snow', 'heavy snow'])

    # Change temperature to categorical
    temp_bins = [-1, 5, 8]
    data['Day'] = pd.cut(data['Day'], bins=temp_bins, labels=['workday', 'weekend'])

    # Change temperature to categorical
    hour_bins = [-1, 5, 12, 17, 21, 24]
    data['Hour'] = pd.cut(data['Hour'], bins=hour_bins, labels=['midnight', 'morning', 'mid-day', 'afternoon', 'night'])

    return data.drop(columns=['Rented Bike Count', 'Month'])


df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
df = process_data(df)
x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['is_busy']), df['is_busy'], test_size=0.1)
dt = DecisionTree()
dt.fit(pd.concat([x_train, y_train], axis=1))
dt.plot_tree()
