import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model


#load the dataset
diabetes = datasets.load_diabetes()

fnum = 9

data_x = diabetes.data[:, np.newaxis, fnum]
data_y = diabetes.target

#Create linear regression model
regr = linear_model.LinearRegression()

#Train dataset
regr.fit(data_x, data_y)
#Predict the targets
predicted_y = regr.predict(data_x)

#Print the coefficient
print('COfficients: ', regr.coef_, regr.intercept_)

#Calculate the Mean Squared Error (MSE)
mse = np.mean((predicted_y - data_y) ** 2)
print(f'Mean squared error: {mse:.2f}')


plt.scatter(data_x, data_y)
plt.plot(data_x, predicted_y, color = 'blue')
plt.show()
