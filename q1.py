import pandas as pd
import numpy as np
import random
import math
import itertools
from pptree import *
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
    def __init__(self, max_depth=8):
        self.max_depth = max_depth
        self.is_fit = False
        self.tree = None

    # Fit the model with the data
    def fit(self, data):
        tree = self.grow_tree(data)
        self.prune_tree(tree)
        self.tree = tree

        self.is_fit = True

    # Receive a list input and output the prediction
    def predict(self, inputs):
        predictions = []
        for i in range(len(inputs)):
            row = inputs.iloc[i]
            node = self.tree
            while len(node.children) > 0:

                # Divide row if features are in the threshold
                if row[node.feature] in node.threshold:
                    node = node.children[0]
                elif len(node.children) > 1:
                    node = node.children[1]
                else:
                    break

            predictions.append(node.predicted_class)

        return predictions

    # Prune splits with no significance
    def prune_tree(self, node):
        i = 2
        curr_children = node.children
        while len(curr_children) > 0:
            next_gen = []

            #   Iterate over current generation
            for child in curr_children:
                if len(child.children) > 0:

                    #   If the split was significant, make no changes
                    if self.is_significant(child):
                        pass

                    #   If split was useless, delete it
                    else:
                        child.children = []

                next_gen.extend(child.children)

            curr_children = next_gen
            i += 1

    def grow_tree(self, data, depth=0, parent=None):

        #   Get target variable ratios
        num_samples_per_class = data['is_busy'].value_counts()

        # The predicted class is the majority class (argmax)
        predicted_class = num_samples_per_class.idxmax()

        # Create a node and fill it with the know data
        node = Node(entropy=self.get_entropy(data), parent=parent, children=[], num_samples_per_class=num_samples_per_class, num_samples=data.shape[0], predicted_class=predicted_class)

        # If only one target label is in the data (entropy=0), create output attribute and return the node.
        if len(data['is_busy'].unique()) == 1:
            node.output = node.output = 'Feature: ' + str(node.feature) + '. Split: ' + str(node.threshold) + '. Entropy: ' + str(round(node.entropy, 3))
            return node

        #   Enter if max depth not exceeded
        if depth < self.max_depth:
            best_node, best_split, highest_gain = self.best_split(data)

            if best_node is not None and best_split is not None:
                # Split the data into two samples by threshold
                left_split = data[data[best_node].isin(best_split)]
                right_split = data.drop(left_split.index)

                # Assign node attributes by best node combination found.
                node.feature = best_node
                node.threshold = best_split
                node.output = 'Feature: ' + str(node.feature) + '. Split: ' + str(node.threshold) + '. Entropy: ' + str(round(node.entropy, 3))

                # Recursively repeat until depth not exceeded
                node.children.append(self.grow_tree(left_split.drop(columns=[best_node]), depth=depth + 1, parent=node))
                node.children.append(self.grow_tree(right_split.drop(columns=[best_node]), depth=depth + 1, parent=node))

        # Delete all empty children
        node.children = [child for child in node.children if child is not None and child.output is not None]
        return node

    def plot_tree(self):
        # If the model is fit, plot the tree
        if self.is_fit:
            print_tree(current_node=self.tree, childattr='children', nameattr='output', horizontal=False)
        else:
            print('You must fit the algorithm before you can plot a tree')

    # Find the best node to split by
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

    # Get best label separation
    def best_attr_combination(self, col_name, current_df):

        # Get initial entropy to with draw from
        original_entropy = self.get_entropy(current_df)

        # Get all values
        stuff = current_df[col_name].unique()

        # If only one option, invalid column
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

                # Don't allow null combinations or combination of all
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

    # Get entropy of current data
    @staticmethod
    def get_entropy(data):
        target_value_count = data['is_busy'].value_counts()

        # If all from the same target value - entropy is 0
        if len(target_value_count) == 1:
            return 0

        p_plus = target_value_count[1] / data['is_busy'].value_counts().sum()
        p_minus = target_value_count[0] / data['is_busy'].value_counts().sum()

        plus_product = -(p_plus * math.log(p_plus, 2))
        minus_product = -(p_minus * math.log(p_minus, 2))

        entropy = plus_product + minus_product
        return entropy

    # Check if split is significant
    def is_significant(self, parent):
        """
        Expected_x - The ratio between target classes if split randomly (by probabilities)
        Actual_x - The actual ratio between classes
        """

        expected_1 = parent.num_samples_per_class / parent.num_samples * parent.children[0].num_samples
        actual_1 = parent.children[0].num_samples_per_class

        if len(parent.children) == 2:
            expected_2 = parent.num_samples_per_class / parent.num_samples * parent.children[1].num_samples
            actual_2 = parent.children[1].num_samples_per_class
        else:
            expected_2 = parent.num_samples_per_class / parent.num_samples * (parent.num_samples - parent.children[0].num_samples)
            actual_2 = parent.num_samples_per_class - parent.children[0].num_samples_per_class

        chi_1 = ((actual_1 - expected_1).pow(2) / expected_1)
        chi_2 = ((actual_2 - expected_2).pow(2) / expected_2)

        # Some of chi squares
        chi_st = (chi_1 + chi_2).sum()
        chi_cr = scipy.stats.chi2.ppf(q=0.95, df=len(parent.children) - 1)
        return chi_st > chi_cr


def process_data(data):
    # Split the date column to multiple columns
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data = data.drop(columns=['Date'])

    # Add a binary column
    if 'Rented Bike Count' in data.columns:
        data['is_busy'] = np.where(data['Rented Bike Count'] >= 650, 1, 0)
        data = data.drop(columns=['Rented Bike Count'])

    # Change temperature to categorical
    temp_bins = [-100, 0, 10, 20, 30, 100]
    data['Temperature(°C)'] = pd.cut(data['Temperature(°C)'], bins=temp_bins, labels=['very cold', 'cold', 'medium', 'hot', 'very hot'])

    # Change humidity to categorical
    humid_bins = [-100, 30, 45, 60, 80, 100]
    data['Humidity(%)'] = pd.cut(data['Humidity(%)'], bins=humid_bins, labels=['very low humidity', 'low humidity', 'medium humidity', 'humid', 'very humid'])

    # Change wind speed to categorical
    wind_bins = [-100, 1, 2, 3, 100]
    data['Wind speed (m/s)'] = pd.cut(data['Wind speed (m/s)'], bins=wind_bins, labels=['very weak wind', 'medium wind', 'strong wind', 'very strong wind'])

    # Change visibility to categorical
    visibility_bins = [-1, 500, 1500, 1800, 10000]
    data['Visibility (10m)'] = pd.cut(data['Visibility (10m)'], bins=visibility_bins, labels=['very bad visibility', 'bad visibility', 'medium visibility', 'good visibility'])

    # Change dew point to categorical
    dew_point_bins = [-100, -5, 5, 15, 100]
    data['Dew point temperature(°C)'] = pd.cut(data['Dew point temperature(°C)'], bins=dew_point_bins, labels=['very low dew point', 'low dew point', 'medium dew point', 'high dew point'])

    # Change solar radiation to categorical
    solar_bins = [-1, .5, 1, 1.5, 10]
    data['Solar Radiation (MJ/m2)'] = pd.cut(data['Solar Radiation (MJ/m2)'], bins=solar_bins, labels=['very low dew point', 'low dew point', 'medium dew point', 'high dew point'])

    # Change rainfall to categorical
    rain_bins = [- 1, 0, 1, 100]
    data['Rainfall(mm)'] = pd.cut(data['Rainfall(mm)'], bins=rain_bins, labels=['no rain', 'rain', 'heavy rain'])

    # Change snowfall to categorical
    snow_bins = [- 1, 0, 1, 10]
    data['Snowfall (cm)'] = pd.cut(data['Snowfall (cm)'], bins=snow_bins, labels=['no snow', 'snow', 'heavy snow'])

    # Change temperature to categorical
    temp_bins = [-1, 5, 8]
    data['Day'] = pd.cut(data['Day'], bins=temp_bins, labels=['workday', 'weekend'])

    # Change temperature to categorical
    hour_bins = [-1, 5, 12, 17, 21, 24]
    data['Hour'] = pd.cut(data['Hour'], bins=hour_bins, labels=['midnight', 'morning', 'mid-day', 'afternoon', 'night'])

    return data.drop(columns=['Month'])


def build_tree(ratio):
    # Read and process the data
    data = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
    data = process_data(data)

    # Create two random samples from the data
    random_index = random.sample(range(data.shape[0]), int(data.shape[0] * ratio))
    train_sample = data.iloc[random_index]
    validation_sample = data[~data.index.isin(random_index)]

    # Separate x and y values for the validation process
    x_validation, y_validation = validation_sample.drop(columns=['is_busy']), validation_sample['is_busy']

    # Train and test the model
    dt = DecisionTree()
    dt.fit(train_sample)
    y_pred = dt.predict(x_validation)

    # Plot the tree
    dt.plot_tree()

    # The accuracy is the percentage of correct labels assigned
    accuracy = (y_pred == y_validation).sum() / (y_pred == y_validation).count()
    error = 1-accuracy

    print('The Decision Tree accuracy is: %.3f' % accuracy)
    print('The Decision Tree error is: %.3f' % error)


def tree_error(k):
    # Read and process the data
    data = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
    data = process_data(data)

    scores = []
    for i in range(k):
        random_index = random.sample(range(data.shape[0]), int(data.shape[0] / k))
        validation_sample = data.iloc[random_index]
        train_sample = data[~data.index.isin(random_index)]

        # Separate x and y values for the validation process
        x_validation = validation_sample.drop(columns=['is_busy'])
        y_validation = validation_sample['is_busy']

        # Train and test the model
        dt = DecisionTree(max_depth=5)
        dt.fit(train_sample)
        y_pred = dt.predict(x_validation)

        # The accuracy is the percentage of correct labels assigned
        accuracy = (y_pred == y_validation).sum() / len(y_pred)
        scores.append(accuracy)

    print('The mean accuracy is: %.3f.' % np.mean(scores))
    print('The standard error is: %f.' % np.std(scores))


def is_busy(row_input):
    data = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape').iloc[20:]

    # to convert lists to dictionary
    row_dict = {}
    for key in data.drop(columns=['Rented Bike Count']).columns:
        for value in row_input:
            row_dict[key] = value
            row_input.remove(value)
            break

    row_df = pd.DataFrame(row_dict, index=[0])
    row_df = process_data(row_df)
    data = process_data(data)
    dt = DecisionTree(max_depth=5)
    dt.fit(data)

    return dt.predict(row_df)[0]


