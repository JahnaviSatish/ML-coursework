import random
def mean(num):
 result1=0
 for i in range(len(num)):
  result1+=num[i]
 result1/=len(num)
 return result1

def median(num):
 num.sort()
 if len(num) % 2 == 1:
    return num[len(num) // 2]
 else:
    return (num[len(num) // 2 - 1] + num[len(num) // 2]) / 2

def mode(num):
 max=0
 result3=0
 for i in range(len(num)):
  count=0
  for j in range(len(num)):
   if num[i]==num[j]:
    count+=1
  if max<count:
   max=count
   result3=num[i]
 return result3
numbers=[]
for x in range(100):
 numbers.append(random.randint(100,150))
print(numbers)
print("mean=",mean(numbers))
print("median=",median(numbers))
print("mode=",mode(numbers))