from data import to_vector, make_data_set
import numpy as np

#The sigmoid and the derivate of the sigmoid function are used for regularization respectively
def sigmoid(value):
    return np.exp(value)/(np.exp(value)+1)

def dsigmoid(value):
    return value * (1 - value)


class ff:

    def __init__(self, num_layers, layer_dim, output_dim, epochs, learning_rate):
        # This fuction is from the data file and produces an array of 8-bit representations of numbers
        X, Y = make_data_set()
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.num_layers = num_layers

        ### NOTICE THAT BIAS HAS NOT BEEN IMPLEMENTED
        ### initializes random weights for the net to produce a single probability
        #bias = 1
        self.W_i2h = 2* np.random.random((layer_dim,layer_dim)) -1
        if (num_layers > 1 ):
            self.W_h2h = 2 * np.random.random((num_layers-1,layer_dim,layer_dim)) -1
        self.W_h2o = 2* np.random.random((layer_dim,output_dim)) -1
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self):
        for j in range ( self.epochs ):
            # This its what trains the nueral net and updates the weights
            inputV = self.X
            # forward pass
            unsq_h = np.dot(inputV, self.W_i2h) 
            sq_h = sigmoid(unsq_h)
            if (self.num_layers > 1):
                unsq_hidden = [unsq_h]
                sq_hidden = [sq_h]
                for i in range(self.num_layers-1):
                    if (i == 0):
                        unsq_hidden.append(sq_h.dot(self.W_h2h[0]))
                        sq_hidden.append(sigmoid(unsq_hidden[i+1]))
                    else:
                        unsq_hidden.append(sq_hidden[i].dot(self.W_h2h[i]))
                        sq_hidden.append(sigmoid(unsq_hidden[i+1]))
                    
                unsq_o = sq_hidden[-1].dot(self.W_h2o) 
                sqo = sigmoid(unsq_o)
            else:
                unsq_o = sq_h.dot(self.W_h2o) 
                sqo = sigmoid(unsq_o)
            

            #find loss
            loss = self.Y - sqo

            #for documentation purposes
            if ( j % 2000 == 0 ):
                print ("Error:", str(np.mean(np.abs(loss))), "\r", end ="")

            #find deltas
            dW_h2o = loss * dsigmoid(sqo)

            if(self.num_layers > 1):
                d_h2h = []
                d_h2h.append(dW_h2o.dot(self.W_h2o.T) * dsigmoid(sq_hidden[-1]))

                for i in range(1, (self.num_layers-1)):
                    #print(d_h2h[i-1])
                    #print(self.W_h2h[i].T)
                    #print(sq_hidden[i])
                    d_h2h.insert(0, d_h2h[i-1].dot(self.W_h2h[i].T) * dsigmoid(sq_hidden[i]))

                dW_i2h = (d_h2h[0].dot(self.W_h2h[0].T)) * dsigmoid(sq_h)
            else:
                dW_i2h = (dW_h2o.dot(self.W_h2o.T)) * dsigmoid(sq_h)

            #Update the weights
            self.W_i2h += inputV.T.dot(dW_i2h) * self.learning_rate

            if(self.num_layers>1):
                for i in range(len(d_h2h)):
                    if(i==0):
                        self.W_h2h[i] += sq_h.T.dot(d_h2h[i]) * self.learning_rate
                    else:
                        self.W_h2h[i] += sq_hidden[i].T.dot(d_h2h[i]) * self.learning_rate
                self.W_h2o += sq_hidden[-1].T.dot(dW_h2o) * self.learning_rate

            else:
                self.W_h2o += sq_h.T.dot(dW_h2o) * self.learning_rate
        print("\n")

    def predict(self, int_G):
        x = to_vector(int_G)

        # forward pass
        unsq_h = np.dot(x, self.W_i2h) 
        sq_h = sigmoid(unsq_h)
        if (self.num_layers > 1):
            unsq_hidden = [unsq_h]
            sq_hidden = [sq_h]
            for i in range(self.num_layers-1):
                if (i == 0):
                    unsq_hidden.append(sq_h.dot(self.W_h2h[0]))
                    sq_hidden.append(sigmoid(unsq_hidden[i+1]))
                else:
                    unsq_hidden.append(sq_hidden[i].dot(self.W_h2h[i]))
                    sq_hidden.append(sigmoid(unsq_hidden[i+1]))
                
            unsq_o = sq_hidden[-1].dot(self.W_h2o) 
            sqo = sigmoid(unsq_o)
        else:
            unsq_o = sq_h.dot(self.W_h2o) 
            sqo = sigmoid(unsq_o)
        #This just does a forward pass and returns the squashed value of the output layer
        #its a repeat of the first half of the loop
        return sqo[0]

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


#heck = ff(3, 8, 1, 15000, 0.1)
#heck.train()
#heck.test()
