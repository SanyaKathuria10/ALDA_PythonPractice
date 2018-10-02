import numpy as np

A = np.identity(5)
print("5*5 Identity matrix: \n")
print(A)

for i in range(0, 5):
    A[i, 1] = 3
print("\n Matrix A after modification: ")
print(A)

sum = 0
for i in range(0, 5):
    for j in range(0, 5):
        sum = sum + A[i, j]
print("\nFinal Sum =")
print(sum)

y = np.transpose(A)
print("\n 5*5 Transpose matrix: ")
print(y)

row = 0
diagonal = 0
for i in range(0, 5):
    for j in range(0, 5):
        if (i == 2):
            row = row + A[i, j]
        if (i == j):
            diagonal = diagonal + A[i, j]
print("\n3rd Row: ")
print(row)
print("\nDiagonal: ")
print(diagonal)

mean = 5
sd = 1.73205
B = np.random.normal(mean, sd, (5, 5))
print("\nMatrix B: ")
print(B)

C = np.zeros((2, 5))

for i in range(0, 5):
    for j in range(0, 5):
        if (i == 0):
            C[i, j] = B[i, j] * B[i + 1, j]
        if (i == 1):
            C[i, j] = B[i + 1, j] + B[i + 2, j] - B[i + 3, j]
print("\nMatrix C:")
print(C)

D = np.zeros((2, 5))
for i in range(0, 2):
    for j in range(0, 5):
        D[i, j] = C[i, j] * (j + 2)
print("\nMatrix D:")
print(D)

X = np.array([2, 4, 6, 8])
X.transpose()
Y = np.array([6, 5, 4, 3])
Y.transpose()
Z = np.array([1, 3, 5, 7])
Z.transpose()
print("\n Covariance Matrix:")
np.vstack([X, Y, Z])
Cov = np.cov([X, Y, Z])
print(Cov)

x = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
x = np.transpose(x)
print("\nLHS:")
print(np.mean(np.square(x)))
print("\n")
print("RHS:")
print(np.var(x) + np.mean(x) ** 2)