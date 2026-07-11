def commonlist(l1,l2):
    l3=[]
    for x in l1:
        for y in l2:
            if x == y:
                l3.append(x)
    return l3

print("enter list 1:")
list1=list(map(int,input().split()))
print("enter list 2:")
list2=list(map(int,input().split()))
len1=len(list1)
len2=len(list2)
res=commonlist(list1,list2)
print("common=",res) #assumption-elements do not duplicate in the lists
print("count=",len(res))