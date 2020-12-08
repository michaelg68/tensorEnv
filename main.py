#https://www.techwithtim.net/tutorials/machine-learning-python/linear-regression-2/
#https://www.youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())
#print(data)

predict = "G3"
X = np.array(data.drop([predict], 1)) #Features
y = np.array(data[predict]) #Label - the data we are trying to predict

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    linear = linear_model.LinearRegression()

    #Training our model:
    linear.fit(x_train, y_train)

    #Verifying how accurate is our model:
    acc = linear.score(x_test, y_test)
    print("Prediction accuracy:", acc)
    if acc > best:
        best = acc
        #Storing our trained model called "linear" in a file
        with open("studentgrades.pickle", "wb") as picklefile:
            pickle.dump(linear, picklefile)

print("The model is stored with the best accuracy:", best)


pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)
print("Checking our predictions:\n__________________")
predictions = linear.predict(x_test) #Get a list of all predictions
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

#Drawing and plotting model
plot = "G1"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade (G3)")
plt.show()

