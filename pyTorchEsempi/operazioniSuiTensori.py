import torch

x = torch.tensor(data=[[1,2,3],[4,5,6], [7,8,9]])
y = torch.tensor(data=[[10,11, 12], [13, 14, 15], [16, 17, 18]])

print(x)
print(y)
print("Sono definite le seguenti operazioni: somma (differenza), "
      "somma per uno scalare, prodotto per uno scalare,"
      "prodotto tra tensori, divisione e trasposizione")
print("Somma tra tensori x e y \n{}".format(x+y))
print("Somma tra il tensore x e lo scalare 3 \n{}".format(x+3))
print("Prodotto tra 2 tensori \n{}".format(x*x))
print("Prodotto tra un tensore x e lo scalare 3 \n{} ".format(x*3))
print("Divisione tra tensori \n{}".format(x/x))
print("Divisione tra tensori \n{}".format(y/x))
print("La trasposizione del tensore \n{} \n{}".format(x, x.transpose(0,1)))


print("----------operazioni_base--------------")
print("ottenere le dimensioni del tensore, le dimensioni del tensore x è {}".format(x.size()))
print("Ottenere il num di elementi presenti nel tensore, in x sono presenti {}".format(x.numel()))

