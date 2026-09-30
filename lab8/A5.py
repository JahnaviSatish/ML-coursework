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

def relu(val):
    y=max(0,val)
    return y

def error_comparator(target,output):
    error=target-output
    return error

inputs = [[0,0],[0,1],[1,0],[1,1]] #XOR GATE LOGIC
target = [0,1,1,0]
weights=[0.2,-0.75]
bias=10 #w0
alpha=0.05
bipolar_errors =[]
sigmoid_errors =[]
relu_errors =[]
step_errors =[]
e_bipolar=0
e_sigmoid=0
e_relu=0
e_step=0

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
    step_errors.append(sse)
    e_step+=1 

    if sse<=0.002:
        print("Converged")
        print("number of epochs for step function:",e_step)
        break

else:
    print("Step function did NOT converge in 1000 epochs")


target = [0,1,1,0]
weights=[0.2,-0.75]
for epochs in range(1000):

    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=bipolarstepfunc(s)
        error = error_comparator(z, output)

        weights[0]+=alpha*error* x[0]
        weights[1]+=alpha*error* x[1]
        bias+=alpha * error

    sse=0
    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=bipolarstepfunc(s)
        error = error_comparator(z, output)
    
        sse+=error**2  
    bipolar_errors.append(sse)
    e_bipolar+=1 

    if sse<=0.002:
        print("Converged")
        print("number of epochs for bipolarstep function:",e_bipolar)
        break

else:
    print("Bipolar step function did NOT converge in 1000 epochs")

weights=[0.2, -0.75]
bias=10
for epochs in range(1000):

    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=sigmoid(s)
        error = error_comparator(z, output)

        weights[0]+=alpha*error* x[0]
        weights[1]+=alpha*error* x[1]
        bias+=alpha * error

    sse=0
    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=sigmoid(s)
        error = error_comparator(z, output)
    
        sse+=error**2  
    sigmoid_errors.append(sse)
    e_sigmoid+=1 

    if sse<=0.002:
        print("Converged")
        print("number of epochs for sigmoid function:",e_sigmoid)
        break

else:
    print("Sigmoid did NOT converge in 1000 epochs")


weights=[0.2, -0.75]
bias=10
for epochs in range(1000):

    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=relu(s)
        error = error_comparator(z, output)

        weights[0]+=alpha*error* x[0]
        weights[1]+=alpha*error* x[1]
        bias+=alpha * error

    sse=0
    for x,z in zip(inputs,target):
        s=summation(x, weights, bias)
        output=relu(s)
        error = error_comparator(z, output)
    
        sse+=error**2  
    relu_errors.append(sse)
    e_relu+=1 

    if sse<=0.002:
        print("Converged")
        print("number of epochs for relu function:",e_relu)
        break

else:
    print("ReLU did NOT converge in 1000 epochs")
    