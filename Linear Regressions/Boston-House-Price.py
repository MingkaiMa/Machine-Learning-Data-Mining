import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model


#load the dataset
boston = datasets.load_boston()

fnum = 2

data_x = boston.data[:, np.newaxis, fnum]
data_y = boston.target

#perfect linear relations
##x = np.array([1,2,3,4])
##data_x = x[:, np.newaxis]
##data_y = [2,4,6,8]


#Split the data into training and testing subsets
N = 20
test_x = data_x[-N:]
train_x = data_x[:-N]

test_y = data_y[-N:]
train_y = data_y[:-N]

#Create linear regression model
regr = linear_model.LinearRegression()

#Train dataset
regr.fit(train_x, train_y)
#Predict the targets
predicted_y = regr.predict(test_x)

#Print the coefficient
print('Cofficients: ', regr.coef_, regr.intercept_)

#Calculate the Mean Squared Error (MSE)
mse = np.mean((predicted_y - test_y) ** 2)
print(f'Mean squared error: {mse:.2f}')


plt.scatter(train_x, train_y)
plt.plot(test_x, predicted_y, color = 'blue')
plt.show()
