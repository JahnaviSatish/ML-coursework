import math
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_excel("ML-lab2 dataset.xlsx",sheet_name="Purchase data")
df["High Value Tx?"]=["Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "No"]
def summation(inputs,weights,bias):
    sum=0
    for i, w in zip(inputs,weights):
        sum+= i*w
    return sum+bias

def sigmoid(val):
    if val>=0:
        return 1/(1+math.exp(-val))
    else:
        exp_val=math.exp(val)
        return exp_val/(1+exp_val)

def error_comparator(target,output):
    error=target-output
    return error

inputs=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)','Payment (Rs)']].values
target=df["High Value Tx?"].map({"Yes": 1, "No": 0}).values #.values to make it an array
weights=[0.02,-0.03,0.25,0.05]
alpha=0.01 #learning rate
bias=0.0 #w0
epoch_errs=[]

for epochs in range(1000):
    for x,z in zip(inputs,target):
        s=summation(x,weights,bias)
        output=sigmoid(s)
        error=error_comparator(z,output)

        weights[0]+=alpha*error* x[0]
        weights[1]+=alpha*error* x[1]
        weights[2]+=alpha*error* x[2]
        weights[3]+=alpha*error* x[3]
        bias+=alpha * error

    # Training
    for x, z in zip(inputs, target):

        s = summation(x, weights, bias)
        output = sigmoid(s)
        error = error_comparator(z, output)

        weights[0] += alpha * error * x[0]
        weights[1] += alpha * error * x[1]
        weights[2] += alpha * error * x[2]
        weights[3] += alpha * error * x[3]
        bias += alpha * error

    # Calculate SSE after the epoch
    sse = 0
    for x,z in zip(inputs,target):
        s=summation(x,weights,bias)
        output=sigmoid(s)
        error=error_comparator(z,output)
        sse+=error**2
    epoch_errs.append(sse)

    print("Epoch:",epochs+1,"SSE:",sse)

    if sse<=0.002:
        print("Converged")
        break



plt.plot(range(1,len(epoch_errs)+1), epoch_errs)
plt.xlabel("Epoch")
plt.ylabel("Sum Squared Error")
plt.grid(True)
plt.show()