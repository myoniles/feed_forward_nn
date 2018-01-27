import random
import os

random.seed(22)

numbers = []
Yvect = []

def to_vector(iris_line):
    iris = []
    iris = iris_line.split(",")
    iris = [float(i) for i in iris[0:4]]
    return iris

def get_expected(iris_line):
    iris = iris_line.split(",")
    name =  iris[-1][5:-1]
    return [int(name == 'setosa') , int(name == 'versicolor'), int(name =='virginica')]

def random_order(data_set, expected_set):
    new_X = []
    new_Y = []

    while (len(data_set)>=1 ):
        rn = int(random.random() * (len(data_set)-1))
        new_X.append(data_set.pop(rn))
        new_Y.append(expected_set.pop(rn))

    return new_X, new_Y
    
    

def make_data_set():
    with open("data/iris.data") as data_file:
        data = data_file.readlines()
    
    x = []
    y = []

    for element in data:
        iris_x = to_vector(element)
        iris_y = get_expected(element)
        x.append(iris_x)
        y.append(iris_y)
    
    x, y = random_order(x,y)
    return x,y 
