import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
df = pd.read_excel("ML-lab2 dataset.xlsx", sheet_name="IRCTC Stock Price")

def auto_calc():
    m = np.mean(df["Price"])
    v = np.var(df["Price"])
    return m, v

def mean_own(price):
    total1 = 0
    count = 0
    for x in price:
        total1 += x
        count += 1
    M = total1 / count
    return M

def var_own(price):
    M = mean_own(price)
    total2 = 0
    count = 0
    for y in price:
        total2 += (y - M) ** 2    # variance formula = sigma(value-mean)^2 / N
        count += 1
    V = total2 / count
    return V

def computation():
    price = df["Price"]

    my_mean_time = []
    numpy_mean_time = []

    for i in range(10):
        start = time.time()
        mean_own(price)
        end = time.time()

        my_mean_time.append(end - start)

        start = time.time()
        np.mean(price)
        end = time.time()

        numpy_mean_time.append(end - start)

    my_var_time = []
    numpy_var_time = []

    for i in range(10):
        start = time.time()
        var_own(price)
        end = time.time()

        my_var_time.append(end - start)

        start = time.time()
        np.var(price)
        end = time.time()

        numpy_var_time.append(end - start)

    return my_mean_time, numpy_mean_time, my_var_time, numpy_var_time

def accuracy():
    price = df["Price"]
    ma = abs(mean_own(price) - np.mean(price))
    va = abs(var_own(price) - np.var(price))
    return ma, va

def wednesdays():
    total3 = 0
    count = 0

    for day, x in zip(df["Day"], df["Price"]): #iterating over both the day and the price together using zip().
        if day == "Wed":
            total3 += x
            count += 1

    mean_wed = total3 / count
    return mean_wed

def april():
    total4=0
    count=0
    for month,x in zip(df['Month'],df['Price']):
        if month=="Apr":
            total4+=x
            count+=1
    mean_apr=total4/count
    return mean_apr

def probability():
    loss=df["Chg%"].apply(lambda x:x<0)
    prob_loss=loss.sum()/len(df)#counts the number of True values because True = 1 and False = 0
    return prob_loss

def profit_wednesdays():
    pro=0
    total_wed=0
    for day,p in zip(df['Day'],df['Chg%']):
        if day=="Wed":
            total_wed+=1
            if p>0:
                pro+=1
    wed_pro=pro/total_wed
    return wed_pro


mean_numpy, var_numpy = auto_calc()

mean_manual = mean_own(df["Price"])
var_manual = var_own(df["Price"])

r, s, q, p = computation()

MA, VA = accuracy()   # MA = mean accuracy, VA = variance accuracy
mean_wed=wednesdays()
mean_april=april()
prob_loss=probability()
wed_pro=profit_wednesdays()

print(df)
print(f"Mean (NumPy): {mean_numpy:.4f}")
print(f"Mean (own): {mean_manual:.4f}")
print(f"Variance (NumPy): {var_numpy:.4f}")
print(f"Variance (own): {var_manual:.4f}")

print("Average My Mean Time for 10 runs:", np.mean(r))
print("Average NumPy Mean Time for 10 runs:", np.mean(s))
print("Average My Variance Time for 10 runs:", np.mean(q))
print("Average NumPy Variance Time for 10 runs:", np.mean(p))

print("Mean Difference:", MA)
print("Variance Difference:", VA)

print(f"Population Mean: {mean_numpy:.4f}")
print(f"Wednesday Sample Mean: {mean_wed:.4f}")#observation=sample mean is lesser than population mean

print(f"Population Mean: {mean_numpy:.4f}")
print(f"April Sample Mean: {mean_april:.4f}")#observation=sample mean much greater than population mean

print("Probability of Loss:", prob_loss)

print("Probability of profit on wednesdays:", wed_pro)
CP = profit_wednesdays()
print("Conditional Probability of Profit given Wednesday:", CP)
#Conditional probability P(Profit | Wednesday) means the probability of making a profit only on Wednesdays. 
#Since the previous calculation also took only Wednesday data, the values are identical.
plt.scatter(df['Chg%'],df['Day'])
plt.title("Chg5 vs Day")
plt.xlabel("Chg%")
plt.ylabel("Day")

plt.show()