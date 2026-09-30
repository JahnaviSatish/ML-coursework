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
alpha=0.05
epochs_err=[]
for epochs in range(1000):

    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=stepfunc(s)
        error = error_comparator(z, output)

        weights[0]+=alpha*error* x[0]
        weights[1]+=alpha*error* x[1]
        bias+=alpha * error

    sse=0
    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=stepfunc(s)
        error = error_comparator(z, output)
    
        sse+=error**2  
    epochs_err.append(sse) 

    if sse<=0.002:
        print("Converged")
        break
    print("input:",x)
    print("target:",z)
    print("summation:",s)
    print("output:",output)
    print("error:",epochs_err)
    print()

plt.plot(range(1, len(epochs_err) + 1), epochs_err)
plt.xlabel("Epoch")
plt.ylabel("Sum Squared Error")
plt.show()