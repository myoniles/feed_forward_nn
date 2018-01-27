from ff import ff
import numpy as np

class iris(ff):
    
    def test(self):
        testing = True
        while (testing):
            sepal_length = input("Enter the sepal length in cm ")
            sepal_length = float(sepal_length)

            sepal_width = input("Enter the sepal length in cm ")
            sepal_width = float(sepal_width)
            
            petal_length = input("Enter the sepal length in cm ")
            petal_length = float(petal_length)
           
            petal_width = input("Enter the sepal length in cm ")
            petal_width = float(petal_width)

            testX = [sepal_length, sepal_width, petal_length, petal_width]
            iris = self.predict(testX).tolist()

            print(iris)
            names = ['setosa','versicolor','virginica']

            print("I would reckon thats a", names[iris.index(max(iris))])

        return names[iris.index(max(iris))]


iris_classifier = iris(2,4,3, 10000, 0.01)
iris_classifier.train()
iris_classifier.test()
