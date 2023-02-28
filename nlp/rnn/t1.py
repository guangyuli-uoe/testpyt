import torch
from torch.nn import functional as F


print(F.one_hot(torch.tensor([0, 2]), 10))

'''
    一个mini-batch
        （batch-size， num—steps）
        （2，5）
    (time_step, batch_size, len(vocab))
'''
X = torch.arange(10).reshape((2, 5))
print(f'X: {X}')
print(f'x.t: {X.T}')
# X.T = (5, 2)
print(F.one_hot(X.T, 10).shape)
print(F.one_hot(X.T, 10))


'''
    for 在 多维tensor上
'''
print('==============================')

aaaaa = torch.ones(size=(2,3,4))
print(aaaaa)
for i in aaaaa:
    print(i)

print(f'len(aaaaa): {len(aaaaa)}')
print(f'aaaaa.shape: {aaaaa.shape}')