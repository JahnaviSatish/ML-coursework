def multiply(a,b):
 result=[[0 for j in range(columnsB)]for i in range(rowsA)]
 for i in range(rowsA):
  for j in range(columnsB):
   for k in range(columnsA):
    result[i][j]+=a[i][k]*b[k][j]
 return result



print("enter number of rows and columns in Matrix A")
rowsA,columnsA=map(int, input().split())
print("enter the matrix A")
matrixA=[]
for i in range(rowsA):
 rows=[]
 for j in range(columnsA):
  rows.append(int(input(f"element ({i},{j}):")))
 matrixA.append(rows)


print("enter number of rows and columns in Matrix B")
rowsB,columnsB=map(int, input().split())
print("enter the matrix B")
matrixB=[]
for i in range(rowsB):
 rows=[]
 for j in range(columnsB):
  rows.append(int(input(f"element ({i},{j}):")))
 matrixB.append(rows)

if columnsA == rowsB :
 result=multiply(matrixA,matrixB)
 for i in range(rowsA):
  rows=[]
  for j in range(columnsB):
   rows.append(result[i][j])
  print(rows)
else:
 print("matrix multiplication not possible")

