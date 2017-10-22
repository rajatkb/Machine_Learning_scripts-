import numpy as np

class KnearestNeighbours():
        def __init__(self):
            self.x = 0
            self.y = 0
        def fit(self , x,y):
            self.x = x
            self.y = y
        
        
            
            
        def predict(self , feature_list):
            if(not len(feature_list) == len(x[0])):
                print("err: feature list length not same as data feature list length")
                return 0
            else:
                list_of_distances= np.array([])
                
                for data_feature in x:
                    print(data_feature)
            
        

        




from sklearn.datasets import load_iris
iris = load_iris();
x = iris.data
y = iris.target

knn = KnearestNeighbours();
knn.fit(x,y)
print(knn.predict([2 , 3 , 4 ,5]));
