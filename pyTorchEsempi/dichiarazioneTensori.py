import torch

y = torch.zeros(size=(3,3))
print(y)

print("Il tipo di ogni elemento contenuto"
      "nel tensore è {}".format(y.dtype))

print("Vediamo la creazione dei diversi tensori")
y1 = torch.tensor(data=8)
print("Il tensore scalare è composto da {} e ha dimensione {}"
      "".format(y1, y1.dim()))
y2 = torch.tensor(data=[1,2,3])
print("Il tensore vettore è composto da {} e ha dimensione {}"
      "".format(y2, y2.dim()))

y3 = torch.tensor(data=[[1,2,3], [4,5,6]])
print("Il tensore matrice è composto da {} e ha dimensione {}"
      "".format(y3, y3.dim()))
y4 = torch.tensor(data=[[[1,2,3], [4,5,6]],
                        [[7,8,9],[10,11,12]]])
print("Il tensore multidimensionale è composto da {} e ha dimensione {}"
      "".format(y4, y4.dim()))

print("Possiamo combinare tensori in"
      " NumPy e PyTorch transformando l'uno o l'altro.")

