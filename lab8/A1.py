import math

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

def bipolarstepfunc(val):
    if val>0:
        return 1
    elif val==0:
        return 0
    else:
        return -1

def sigmoid(val):
    y= 1/(1+math.exp(-val))
    return y

def tanh(val):
    y= (1-math.exp(-val))/(1+math.exp(-val))
    return y

def relu(val):
    y=max(0,val)
    return y

def leakyrelu(val, beta=0.01):
    return max(beta*val, val)

def error_comparator(target,output):
    error=target-output
    return error

inputs = [[0, 0],[0, 1],[1, 0],[1, 1]] #AND GATE LOGIC
target = [0, 0, 0, 1]
weights=[0.2,-0.75]
bias=10 #w0
alpha=0.05

for x,z in zip(inputs,target):
    s=summation(x, weights, bias)
    output=stepfunc(s)
    error = error_comparator(z, output)

    print("input:",x)
    print("target:",z)
    print("summation:",s)
    print("output:",output)
    print("error:",error)
    print()