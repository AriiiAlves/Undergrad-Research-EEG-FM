import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 1. nn.Sequential empilha as camadas em ordem. Os dados passam por elas, uma após a outra.

modelo = nn.Sequential(
    nn.Linear(64, 64),   # camada de entrada: 64 dados -> 64 neurônios
    nn.ReLU(),           # ativação não-linear
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)     # saída: 10 valores (classifica)
    # Sem função de ativação, pois é responsabilidade da Loss.
)

print(modelo)

# 2. Preparando os dados
# Carrega dataset de dígitos
digits = datasets.load_digits()

# Separa 20% para bench (teste) e 80% para treino. X = dados, Y = label
X_train, X_test, Y_train, Y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Normalizando dados (O brilho do pixel varia de 0 a 16)
X_train = X_train / 16.0
X_test = X_test / 16.0

# Converter para tensores PyTorch (necessário para TensorDataset)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)   # CrossEntropyLoss exige long (int64)
Y_test = torch.tensor(Y_test, dtype=torch.long)

dataset = TensorDataset(X_train, Y_train)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Função de perda (Loss) e otimizador
loss_fn = nn.CrossEntropyLoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

#lr = Learning Rate (passo)

# 4. Loop de treino
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)

n_epocas = 100

for epoca in range(n_epocas):
    modelo.train()  # modo treino (ativa dropout, etc)
    perda_total = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        # 1. Zerar gradientes acumulados
        otimizador.zero_grad()

        # 2. Forward: passar os dados pela rede
        pred = modelo(xb)

        # 3. Calcular a perda
        perda = loss_fn(pred, yb)

        # 4. Backward: calcular gradientes
        perda.backward()

        # 5. Atualizar os pesos
        otimizador.step()

        perda_total += perda.item() * xb.size(0)

    perda_media = perda_total / len(dataset)
    if (epoca + 1) % 10 == 0:
        print(f"Época {epoca+1}/{n_epocas} - perda: {perda_media:.4f}")

# 5. Avaliação
modelo.eval()  # desliga dropout/batchnorm em modo treino
with torch.no_grad():  # não precisa calcular gradiente para avaliar
    X_teste = X_test.to(device)
    Y_teste = Y_test.to(device)
    saida = modelo(X_teste)
    classes_previstas = saida.argmax(dim=1)  # pega a classe com maior "score"
    acertos = (classes_previstas == Y_teste).sum().item()
    total = Y_teste.size(0)
    acuracia = acertos / total

    print(f"Acurácia no teste: {acuracia:.2%}")
    print("Previsões:", classes_previstas[:20])
    print("Labels:    ", Y_teste[:20])

# 6. Salvando e Carregando o modelo
## Salvar
#torch.save(modelo.state_dict(), "digit-recognition.pth")

## Carregar (precisa recriar a mesma arquitetura antes)
# modelo_novo = nn.Sequential(
#     nn.Linear(10, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1)
#)
#modelo_novo.load_state_dict(torch.load("modelo.pth"))
#modelo_novo.eval()