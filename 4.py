import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

#Inputs and target values to learn the xor gate
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
t = [0, 1, 1 ,0]

#Hidden layer perceptron 1 
b1 = -0.3
w11 = 0.21
w12 = -0.4

#Hidden layer perceptron 2
b2=0.25
w21 = 0.15
w22 = 0.1

#Output layer perceptron 1
b3=-0.4
w13 = -0.2
w23=0.3


epochs = 0
train = True

print("The weights are")
print("w11 : %4.2f w12: %4.2f w21: %4.2f w22: %4.2f w13: %4.2f w23: %4.2f \n"%(w11,w12,w21,w22,w13,w23))

while(train):
    for i in range(len(x1)):
        #Hidden layer
        net_h1 = b1 + (w11 * x1[i]) + (w21 * x2[i])
        net_h2 = b2 + (w12 * x1[i]) + (w22 * x2[i])
        out_h1 = round(sigmoid(net_h1), 4)
        out_h2 = round(sigmoid(net_h2), 4)

        #Output layer
        net_o1 = b3 + (w13 * out_h1) + (w23 * out_h2)
        out_o1 = round(sigmoid(net_o1), 4)

        #Output layer error calculation
        del_o1 = round((out_o1 * (1 - out_o1) * (t[i] - out_o1)), 4)
        b3 = round((b3 + del_o1), 4)
        w13 = round((w13 + (del_o1 * out_h1)), 4)
        w23 = round(w23 + (del_o1 * out_h2), 4)

        #Hidden layer error calculation
        del_h1 = round((out_h1 * (1 - out_h1) * w13 * del_o1), 4)
        del_h2 = round((out_h2 * (1- out_h2) * w23 * del_o1), 4)

        #Update weights of hidden layer 1
        b1 = round((b1 + del_h1), 4)
        w11 = round((w11 + (del_h1 * x1[i])), 4)
        w12 = round(w12 + (del_h1 * x1[i]), 4)

        #Update weights of hidden layer 2
        b2= round((b2 + del_h2), 4)
        w21= round(w21 + (del_h2*x2[i]), 4)
        w22= round(w22 + (del_h2*x2[i]), 4)

        print("Epoch: " + str(epochs))
        print("w11 : %5.4f w12: %5.4f w21: %5.4f w22: %5.4f w13: %5.4f w23: %5.4f "%(w11,w12,w21,w22,w13,w23))
        print("Error: %5.3f" % abs(del_o1))
        epochs = epochs + 1
    if(epochs >= 10):
        train=False
