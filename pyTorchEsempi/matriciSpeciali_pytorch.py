import torch
from datetime import datetime


print("Vediamo come si dichiarano le matrici identità, quelle zero e di 1")
identity = torch.eye(n=3)
zeros = torch.zeros(size=(3,3))
ones = torch.ones(size=(3,3))
print(identity)
print(zeros)
print(ones)

print("diagonalizzazione, autovettori e autovalori di una matrice")
x = torch.tensor(data=[[1,2,3],[4,5,6], [7,8,9]])
diag_x = x.diag(diagonal=0)
print(diag_x)
#x.eig(eigenvectors=True)
# print CUDNN backend version
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The CUDNN backend version: {}'.format(now, torch.backends.cudnn.version()))
print(torch.cuda.is_available())


print('[LOG {}] The CUDNN backend version: {}'.format(now, torch.backends.cudnn.version()))
print(torch.cuda.is_available())
