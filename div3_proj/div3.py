from ff import ff
import numpy as np

#The sigmoid and the derivate of the sigmoid function are used for regularization respectively

class div3(ff):

    def test(self):
        testing = True
        while(testing):
            testInput = input("Enter a number ")
            testInput = int(testInput)
            if(testInput >= 0 and testInput < 256 ):
                if(self.predict(testInput) >= 0.5):
                    heck = "yes"
                else:
                    heck = "no"
                print("The probability that" , testInput , "is divisible by three is" , self.predict(testInput))
                print("In other words:" , heck)
            else:
                testing = False


    def num_wrong(self):
        y = [(i, self.predict(i)) for i in range(256)]
        y = [ j[0] for j in y if((j[0]%3==0 and j[1]%3<0.5 )or(j[0]%3!=0 and j[1] > 0.5))]
        return y

heck = div3(3, 8, 1, 15000, 0.1)
heck.train()
print("Numbers wrong: ",heck.num_wrong(), "\n")
heck.test()
