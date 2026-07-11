def transpose(a):
 trans=[[0 for j in range(rowsA)]for i in range(columnsA)]
 for i in range(rowsA):
  for j in range(columnsA):
   trans[j][i]=a[i][j]
 return trans

print("enter number of rows and columns in Matrix A")
rowsA,columnsA=map(int, input().split())
print("enter the matrix A")
matrixA=[]
for i in range(rowsA):
 rows=[]
 for j in range(columnsA):
  rows.append(int(input(f"enter element ({i},{j}):")))
 matrixA.append(rows)
print("original matrix:",matrixA)
res=transpose(matrixA)
print("transpose:",res)
