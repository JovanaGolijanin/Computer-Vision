import numpy as np

a = np.arange(15).reshape(3, 5)
print(a)
# [[ 0 1 2 3 4]
# [ 5 6 7 8 9]
# [10 11 12 13 14]]

print(a.shape) # (3, 5)
print(a.ndim) # 2
print(a.dtype.name) # int32
print(a.itemsize) # 4
print(a.size) # 15
print(type(a)) # <class 'numpy.ndarray'>

#Inicijalizacija Numpy niza

# Na osnovu liste
a = np.array([[1.5, 2, 3], [4, 5, 6]])
print(a)
# Nule (uz specificiranje tipa)
b = np.zeros((3, 4, 2), dtype=np.uint8)
print(b)
# Jedinice (uz specificiranje tipa)
c = np.ones((3, 4, 2), dtype=np.float32)
print(c)
# Uniformna raspodela u opsegu od -1 do 1
d = np.random.uniform(-1, 1, (3, 4))
print(d)
# Gausova raspodela oko vrednosti 0
e = np.random.normal(0, 1, (3, 4))
print(e)


#Osnovne operacije

a = np.array([1, 2, 3, 4, 5, 6]).reshape((2, 3))
b = np.array([10, 20, 30, 40, 50, 60]).reshape((2, 3))
print(a)
print(b)
c = a + b # sabiranje po elementima
print(c)
d = a * b # mnozenje po elementima
print(d)
e = a @ b.T # transponovanje i matricno mnozenje
print(e)
f = a.dot(b.T) # drugi zapis za matricno mnozenje
print(f)

#nastavak

a = np.array([1, 2, 3, 4, 5, 6]).reshape((2, 3))
b = np.array([10, 20, 30, 40, 50, 60]).reshape((2, 3))
print(a)
print(b)
print(a.sum()) # 21
print(a.sum(axis=0)) # [5 7 9]
print(a.sum(axis=-1)) # [6 15]
print(a.sum(axis=(0, 1))) # 21
print(a.min()) # 1
print(a.max()) # 6
print(np.minimum(a, b)) # [[1 2 3] [4 5 6]]
print(np.maximum(a, b)) # [[10 20 30] [40 50 60]]


#Promena dimenzije i spajanje

a = np.array([1, 2, 3, 4, 5, 6]).reshape((2, 3))
b = np.array([10, 20, 30, 40, 50, 60]).reshape((2, 3))
print(a)
print(b)
ax = np.expand_dims(a, axis=2)
print(ax) # [[[1] [2] [3]] [[4] [5] [6]]]
bx = b.reshape((2, 3, 1)) # isti efekat na drugi nacin
print(bx) # [[[10] [20] [30]] [[40] [50] [60]]]
cx = np.append(ax, bx, axis=2)
print(cx)
# [[[ 1 10] [ 2 20] [ 3 30]]
# [[ 4 40] [ 5 50] [ 6 60]]]


#Pristup elementima
print("\n\nPristup elementima\n\n")

a = np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6,
-6])
a = a.reshape((2, 3, 2))
print(a)
# [[[ 1 -1] [ 2 -2] [ 3 -3]]
# [[ 4 -4] [ 5 -5] [ 6 -6]]]
print(a[0, 0, 0]) # 1
print(a[0]) # [[ 1 -1] [ 2 -2] [ 3 -3]]
print(a[:, 1]) # [[ 2 -2] [ 5 -5]]
print(a[:, 1, 0]) # [2 5]
print(a[:, 1, 1:]) # [[-2] [-5]]
print(a[0, 0:2, -1]) # [-1 -2]
