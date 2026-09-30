import math
import matplotlib.pyplot as plt
def summation(inputs,weights,bias):
    sum=0
    for i, w in zip(inputs,weights):
        sum+= i*w
    return sum+bias

def stepfunc(val):
    if val>0:
        return 1
    else:
        return 0

def error_comparator(target,output):
    error=target-output
    return error

inputs = [[0, 0],[0, 1],[1, 0],[1, 1]] #AND GATE LOGIC
target = [0, 0, 0, 1]
weights=[0.2,-0.75]
bias=10 #w0
alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
iter_per_alpha=[]
epochs_err=[]

for a in alpha:
 target = [0, 0, 0, 1]
 weights=[0.2,-0.75]
 for epochs in range(1000):

    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=stepfunc(s)
        error = error_comparator(z, output)

        weights[0]+=a*error* x[0]
        weights[1]+=a*error* x[1]
        bias+=a * error

    sse=0
    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=stepfunc(s)
        error = error_comparator(z, output)
    
        sse+=error**2  
    epochs_err.append(sse) 

    if sse<=0.002:
        print("Converged")
        print("number of epochs for learning rate:",a)
        iter_per_alpha.append(epochs+1)
        print(epochs+1)
        break

plt.plot(alpha, iter_per_alpha, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Iterations to Converge")
plt.show()