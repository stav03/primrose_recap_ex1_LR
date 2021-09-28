
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_theta_by_norm_eqns(X, y):
    """ Calculate theta according to normal equations:
        theta = inv(X^T.X).(X^T.Y), e.g dim(X) = [2x700], dim(y) = [700x1] """
    x_tr_by_x = np.dot(X.T, X)  # [2x700].[700x2] = [2x2]
    inv_x_tr_by_x = np.linalg.inv(x_tr_by_x)  # [2x2]
    x_tr_by_y = np.dot(X.T, y)  # [2x700].[700x1] = [2x1]
    theta = np.dot(inv_x_tr_by_x, x_tr_by_y)  # [2x2].[2x1] = [2x1]
    return theta


def add_vector_of_ones(X):
    """ Add columns of ones at the beginning of X"""
    m, n = X.shape
    vector_of_ones = np.ones([m, 1])
    X_with_ones = np.hstack([vector_of_ones, X])
    return X_with_ones

# PART 1. Small data and bias
# Data:
# columns_names = ['older sibling', 'younger sibling', 'times talked']
#
# data_values = [[31, 22, 12], [22, 21, 3], [40, 37, 8], [26, 25, 12]]
# df = pd.DataFrame(data=data_values, columns=columns_names)
# print(df.head())
#
# # [1.1]
# # Convert to normal equations inputs:
# X = df.iloc[:, :2].to_numpy()
# y = df.iloc[:, 2].to_numpy()
# y = y[:, np.newaxis]
# print(X)
# print(y)
#
# # Calculate theta:
# theta = calc_theta_by_norm_eqns(X, y)
# print(theta)

# [1.2]
# Add column of ones:
# X_with_ones = add_vector_of_ones(X)
# # print(X_with_ones)
# # print(y)
#
# # Calculate theta:
# theta = calc_theta_by_norm_eqns(X_with_ones, y)
# print(theta)

# PART 2. Exploding normal equations issue --> (some data) --> not exploding
# 2.1 Read the data:
data = pd.read_csv('data_for_linear_regression.csv')
print("First 5 lines of data for linear regression:")
print(data.head())
print("Data length is: ", len(data))

# Outlier:
X_max = max(data[['x']].to_numpy())
print(f'Maximal value of X is: {X_max}')  # single outlier at 213


# Plot the data as sequences:
data[['x']].plot(c='green')
curr_axes = plt.gca()
data[['y']].plot(ax=curr_axes, c='red')
plt.title("all X,y data as sequences")
plt.grid()
plt.legend()
plt.show()


# 2.2 Convert to numpy array:
data_arr = data.to_numpy()

# 2.3 Plot the data as scatter:
# Divide into X and y:
X = data_arr[:, 0]
y = data_arr[:, 1]

# Check the shapes:
# print(X.shape)
# print(y.shape)

# Plot:
plt.close("all")
plt.figure()
# plt.xlim((0, 4000))
# plt.ylim((0, 4000))
plt.title("Data for regression (all the data)")
plt.scatter(X,y)
plt.grid()
plt.show()

# 2.4 Take some data:
number_of_points = 200
X_some = X[: number_of_points]
y_some = y[: number_of_points]
# Plot some data scatter:
plt.title(f"Data for regression ('some data', {number_of_points} points)")
plt.scatter(X_some,y_some)
plt.xlabel("X values")
plt.ylabel("y values")
plt.grid()
# plt.show()
# Convert to normal equations inputs:
X_some_for_normal = X_some[:, np.newaxis]
y_some_for_normal = y_some[:, np.newaxis]
X_some_with_ones = add_vector_of_ones(X_some_for_normal)

# Calculate theta:
theta = calc_theta_by_norm_eqns(X_some_with_ones, y_some_for_normal)

# Theta representations:
print("Theta as vector:\n", theta)

theta_as_di = {}
for ind in range(len(theta)):
    key = 'theta_' + str(ind)
    theta_as_di[key] = np.round(theta[ind, 0],4)

print("Theta as dictionary:\n", theta_as_di)

# 2.5 Add trend line to some data:
# plot_line_on_data(some_data_axes, X_some, theta_as_di)
x_1 = min(X_some)
y_1 = theta_as_di['theta_1']*x_1 + theta_as_di['theta_0'] # y = mx+b

x_2 = max(X_some)
y_2 = theta_as_di['theta_1']*x_2 + theta_as_di['theta_0']  # y = mx+b

some_data_axes = plt.gca()
some_data_axes.plot([x_1, x_2], [y_1, y_2], label='trend line', c='black')
plt.legend()
plt.show()

# 2.6 Add trend line for all the data (ouliner deleted0:
plt.title("Data for regression (all the data) with trend line")
#
x_1 = min(np.delete(X,213))
y_1 = theta_as_di['theta_1']*x_1 + theta_as_di['theta_0'] # y = mx+b

x_2 = max(np.delete(X,213))
y_2 = theta_as_di['theta_1']*x_2 + theta_as_di['theta_0']  # y = mx+b

plt.scatter(X,y)
plt.plot([x_1, x_2], [y_1, y_2], label='trend line', c='black')
plt.grid()
plt.show()

