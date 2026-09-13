import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. nn.Sequential empilha as camadas em ordem. Os dados passam por elas, uma após a outra.

modelo = nn.Sequential(
    nn.Linear(10, 32),   # camada de entrada: 10 features -> 32 neurônios
    nn.ReLU(),           # ativação não-linear
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)     # saída: 1 valor (ex: regressão ou classificação binária)
)

print(modelo)

# 2. Preparando os dados
## Dados fake: 200 amostras, 10 features
X = torch.randn(200, 10)
y = (X.sum(dim=1, keepdim=True) > 0).float()  # alvo binário: 0 ou 1

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Função de perda (Loss) e otimizador
loss_fn = nn.BCEWithLogitsLoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

# 4. Loop de treino
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)

n_epocas = 50

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
    X_teste = torch.randn(20, 10).to(device)
    saida = modelo(X_teste)
    probs = torch.sigmoid(saida)      # se usou BCEWithLogitsLoss
    classes = (probs > 0.5).int()
    print(classes)

# 6. Salvando e Carregando o modelo
## Salvar
torch.save(modelo.state_dict(), "modelo.pth")

## Carregar (precisa recriar a mesma arquitetura antes)
modelo_novo = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
modelo_novo.load_state_dict(torch.load("modelo.pth"))
modelo_novo.eval()