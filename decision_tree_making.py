import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

import data_cleaning



home_data = pd.read_excel('raw-data.xlsx')

# create a duplicate for backup
data = home_data.__deepcopy__()

# Change the string variable into numerical variables for the model
data = data_cleaning.numerical_convertor(data)

# Split the data for validation
train_data = data[0:2800]
valid_data = data[2800:-1]


# State the output which is the size selection of each product
train_y = train_data['Size selection']
valid_y = valid_data['Size selection']


# Select the features for fitting the model
feature_columns = ['Brand', 'Product Type', 'Collar Length(cm)', 'Chest Lengh(cm)', 'Waist Length(cm)', 'Personal style', 'Where to live', 'Occupation']
train_X = train_data[feature_columns]
valid_X = valid_data[feature_columns]


# Specify the model for this problem, maximum depth is 8.
size_selection_model = DecisionTreeClassifier(random_state=1, max_depth=5)

# Fit the mode with training data
size_selection_model.fit(train_X, train_y)

# A small part of the data
print("First in-sample predictions:", size_selection_model.predict(train_X.head()))
print("Actual target values for those homes:", train_y.head().tolist())

# Print out the result of the Decision Tree with depth being 8 into a png file
dot_data = export_graphviz(size_selection_model, out_file= None, feature_names = feature_columns, class_names= ['XS', 'S', 'M', 'L', 'XL', 'XXL'],
                           filled = True, rounded = True, special_characters= True)
graph = graphviz.Source(dot_data, format = 'png')
graph
graph.render("decision_tree_graphivz")
'decision_tree_graphivz.png'

# Validation part:
valid_predic = size_selection_model.predict(valid_X)
print(f'The first five predictions of the model are {valid_predic[0: 5]}')
print(f'The first five validating outputs are{valid_y.head().tolist()}')

valid_y1 = valid_y.tolist()
mistake = 0
for i in range(len(valid_predic)):
    if valid_predic[i] != valid_y1[i]:
        mistake += 1
average_mistake_out = mistake/ len(valid_predic)

train_predict = size_selection_model.predict(train_X)
train_y1 = train_y.tolist()
mistake = 0
for i in range(len(train_predict)):
    if train_predict[i] != train_y[i]:
        mistake += 1
average_mistake_in = mistake/ len(train_predict)
print(average_mistake_in, average_mistake_out)





