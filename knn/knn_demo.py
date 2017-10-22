import numpy as np
class KnearestNeighbours():
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        
    def fit(self , x , y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.classes = np.unique(y)
        
    def predict(self , fvector , k):
        if(not len(fvector) == len(self.x[0])):
            print("err: given vector of not same length as feature vector")
            return 0
        
        lenX = len(self.x)
        lenY = len(self.y)
        
        if(not lenX == lenY):
            print("err: unequal dataset given X:",lenX," Y:",lenY)
            return 0
        
        list_of_distance = []
        for vector in self.x:
           list_of_distance.append(np.linalg.norm(vector - fvector)) 
        
        list_of_distance = np.array(list_of_distance)
        k_nearest_classes = (self.y[list_of_distance.argsort()])[0:k] 
        return k_nearest_classes[(np.bincount(k_nearest_classes)).argmax()]



from sklearn.datasets import load_iris
iris = load_iris()
x_test = iris.data
y_test = iris.target


knn= KnearestNeighbours()
knn.fit(x_test , y_test) 
cls =knn.predict(iris.data[101] , 3)

print(iris.target_names[cls])
print(iris.target_names[y_test[101]])