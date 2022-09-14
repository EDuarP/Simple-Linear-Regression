import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([5,15,25,35,45,55,65,75]).reshape((-1, 1))
y = np.array([3,13,26,36,24,45,56,67])
model = LinearRegression()
model.fit(x,y)
R_sq = model.score(x,y)
print('coefficient of determination:', R_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(x)
print('predicted response:', y_pred)
x_new = np.random.randint(10, size=(10, 1))
y_new = model.predict(x_new)
print('New values:', x_new)
print('predicted response:', y_new)
plt.scatter(x_new, y_new, color='orange')
plt.plot(x_new,model.coef_*x_new+model.intercept_, '-r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()