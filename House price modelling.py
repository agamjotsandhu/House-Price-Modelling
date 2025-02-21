# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
# Path of the file to read
iowa_file_path = '/Users/agamjotsandhu/Desktop/Learning Data science/practical tutorials/housing predictions/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Specify the model
iowa_model = DecisionTreeRegressor(random_state = 1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_y, val_predictions)

print(val_mae)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
leaf_and_mae = [5, get_mae(5, train_X, val_X, train_y, val_y)]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    if my_mae < leaf_and_mae[1]:
        leaf_and_mae = [max_leaf_nodes, my_mae]

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = leaf_and_mae[0]

final_model = DecisionTreeRegressor(random_state = 1, max_leaf_nodes = best_tree_size)
final_model.fit(X, y)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print(f"Validation MAE for Random Forest Model: {rf_val_mae}")


test_file_path = "/Users/agamjotsandhu/Desktop/Learning Data science/practical tutorials/housing predictions/test.csv"
test_data = pd.read_csv(test_file_path)
test_X = test_data[feature_columns]

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data = RandomForestRegressor(random_state = 1)
rf_model_on_full_data.fit(X, y)

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)
print(test_preds)
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv')