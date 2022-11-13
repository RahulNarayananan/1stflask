#importing necessary modules
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

#reading the data file and extracting values from csv file
read=pd.read_csv("C:\\Users\\rahul\\Desktop\\API\\FoodTruck.csv")
x=read['Population'].values
y=read['Profit'].values

#reshaping array and fitting into model to check accuracy of the model
x=x.reshape(-1,1)
model= LinearRegression()
model.fit(x,y)
accuracy=model.score(x,y) 

pickle.dump(model,open('model.pkl','wb'))