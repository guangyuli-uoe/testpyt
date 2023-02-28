a = [1, 2, 3]
print(a[:-1])

b = [('the', 'time'), ('time', 'machine')]
print(b[0])
print(isinstance(b[0], list))
tokens = [token for line in b for token in line]
print(tokens)