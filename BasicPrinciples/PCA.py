import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Manual Principle Component Analysis on iris dataset
# reduction of 4 attributes to 2.

data = pd.read_csv('iris.data.txt',sep=",", header= None)
data.columns = ["s_length","s_width","p_length","p_width","class"]
class_label = data.drop(["s_length","s_width","p_length","p_width"],1)
data = data.drop(["class"],1)
data = pd.DataFrame.as_matrix(data)

n = 149                                     # number of instances - 1
mean = np.mean(data,axis=0)
center_mean = data-mean                     #center the data
tranpose_mean = np.transpose(center_mean)   #tranpose the centered data
matrix = tranpose_mean.dot(center_mean)     #matrix multiplication
cov_matrix = np.divide(matrix,n)            # get covariance matrix
eigenVal, eigenVector = np.linalg.eig(cov_matrix) # get eigen stuff
two_component = eigenVector[:,:2]  # get first two vectors with biggest eigenVal
finaldata = center_mean.dot(two_component)  #data after PCA

chart_data = pd.DataFrame(data=finaldata,columns=['x','y'])
chart_data['class_label'] = class_label
labels = ['Iris-setosa','Iris-versicolor','Iris-virginica']
color = ['r','g','b']
chart_data.class_label.replace(labels,color,inplace=True)
for row in chart_data.itertuples():
    plt.scatter(row.x,row.y,c=row.class_label)
plt.show()

#colors for plot, red = iris-setosa, green = iris-versicolor, blue = iris-virginica
# To use np.cov() - use data.T , Transpose cuz python looks at features in rows...
