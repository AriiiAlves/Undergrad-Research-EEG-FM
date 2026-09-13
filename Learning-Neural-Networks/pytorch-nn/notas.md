# Tutorial: Criando e Treinando Redes Neurais com PyTorch usando `nn.Sequential`

Este guia te ensina o fluxo completo do PyTorch: construir uma rede com `nn.Sequential`, preparar dados, treinar, avaliar e salvar o modelo. A ideia é que você consiga adaptar esse esqueleto para **qualquer** rede simples (classificação, regressão, imagens achatadas, etc).

---

## 1. Instalação

```bash
pip install torch torchvision --break-system-packages
```

(Se tiver GPU NVIDIA, veja em pytorch.org o comando específico com CUDA.)

---

## 2. A ideia do `nn.Sequential`

`nn.Sequential` é a forma mais simples de montar uma rede: você empilha camadas em ordem, e os dados passam por elas uma após a outra, sem precisar escrever uma classe.

```python
import torch
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(10, 32),   # camada de entrada: 10 features -> 32 neurônios
    nn.ReLU(),           # ativação não-linear
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)     # saída: 1 valor (ex: regressão ou classificação binária)
)

print(modelo)
```

Regras importantes:
- O número de saída de uma camada `Linear` tem que bater com o número de entrada da próxima.
- Você pode misturar `nn.Linear`, `nn.Conv2d`, `nn.ReLU`, `nn.Sigmoid`, `nn.Dropout`, `nn.BatchNorm1d`, etc.
- A última camada geralmente **não** tem ativação (a ativação final fica embutida na função de perda, como veremos).

---

## 3. Camadas mais comuns (para você montar a sua)

| Camada | Uso típico |
|---|---|
| `nn.Linear(in, out)` | Dados tabulares, MLPs |
| `nn.Conv2d(in_ch, out_ch, kernel_size)` | Imagens |
| `nn.Flatten()` | Achatar imagem antes de um `Linear` |
| `nn.ReLU()`, `nn.LeakyReLU()`, `nn.GELU()` | Ativações |
| `nn.Sigmoid()` | Saída para classificação binária (0 a 1) |
| `nn.Softmax(dim=1)` | Saída para classificação multi-classe (raramente usada dentro do modelo, pois `CrossEntropyLoss` já aplica) |
| `nn.Dropout(p=0.3)` | Regularização, evita overfitting |
| `nn.BatchNorm1d(n)` | Estabiliza o treino |

Exemplo de rede convolucional simples para imagens 28x28 em escala de cinza (tipo MNIST):

```python
modelo_cnn = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),           # 28x28 -> 14x14
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),           # 14x14 -> 7x7
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 64),
    nn.ReLU(),
    nn.Linear(64, 10)          # 10 classes
)
```

---

## 4. Preparando os dados

PyTorch trabalha com **tensores**. Você quase sempre vai usar `Dataset` + `DataLoader` para organizar os dados em lotes (batches).

### Exemplo com dados sintéticos (regressão)

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Dados fake: 200 amostras, 10 features
X = torch.randn(200, 10)
y = (X.sum(dim=1, keepdim=True) > 0).float()  # alvo binário: 0 ou 1

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

Se você tiver um CSV, o fluxo é: carregar com `pandas` → converter colunas para `torch.tensor(...)` → montar o `TensorDataset`.

Batch size é o número de amostras que a rede processa de uma vez antes de atualizar os pesos. Isso significa que, a cada iteração do for xb, yb in loader:, o PyTorch pega 16 amostras do dataset de treino, passa todas juntas pela rede, calcula a perda média desse grupo, e só então atualiza os pesos.

Você tem 3 opções:

1. Batch size = tamanho total do dataset (ex: 1437 amostras de treino inteiras de uma vez)
→ chamado batch gradient descent. Mais estável, mas lento e consome muita memória — em datasets grandes pode nem caber na GPU/RAM.
2. Batch size = 1 (uma amostra por vez)
→ chamado SGD puro (stochastic). Rápido por passo, mas o gradiente é "ruidoso" (varia muito de amostra pra amostra), tornando o treino instável.
3. Batch size intermediário (16, 32, 64, 128...)
→ chamado mini-batch gradient descent. É o que quase todo mundo usa na prática — equilibra estabilidade e velocidade.

Com `batch_size=16`, cada época faz `1437 / 16 ≈ 90` atualizações de peso. Se você aumentasse para `batch_size=64`, teria só `~22` atualizações por época — mais rápido, porém com gradientes mais "médios" (menos variação).

Valores comuns são potências de 2 (16, 32, 64, 128) porque isso costuma otimizar melhor o uso de memória em GPUs

---

## 5. Função de perda (loss) e otimizador

A escolha da função de perda depende do problema:

| Problema | Última camada | Loss |
|---|---|---|
| Regressão | `Linear` (sem ativação) | `nn.MSELoss()` |
| Classificação binária | `Linear` (1 saída, sem ativação) | `nn.BCEWithLogitsLoss()` |
| Classificação multi-classe | `Linear` (N saídas, sem ativação) | `nn.CrossEntropyLoss()` |

```python
loss_fn = nn.BCEWithLogitsLoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
```

`Adam` é uma escolha segura e padrão para começar. `lr` (learning rate) controla o tamanho do passo do treino — comece com `1e-3` e ajuste se necessário.

---

## 6. O loop de treino

Esse é o coração do PyTorch. Todo treino segue os mesmos 5 passos por batch:

```python
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
```

Esses 5 passos (`zero_grad → forward → loss → backward → step`) são **sempre os mesmos**, independente da arquitetura. É o padrão que você vai repetir em qualquer projeto.

---

## 7. Avaliação

```python
modelo.eval()  # desliga dropout/batchnorm em modo treino
with torch.no_grad():  # não precisa calcular gradiente para avaliar
    X_teste = torch.randn(20, 10).to(device)
    saida = modelo(X_teste)
    probs = torch.sigmoid(saida)      # se usou BCEWithLogitsLoss
    classes = (probs > 0.5).int()
    print(classes)
```

Para multi-classe, a predição da classe é `saida.argmax(dim=1)`.

---

## 8. Salvando e carregando o modelo

```python
# Salvar
torch.save(modelo.state_dict(), "modelo.pth")

# Carregar (precisa recriar a mesma arquitetura antes)
modelo_novo = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
modelo_novo.load_state_dict(torch.load("modelo.pth"))
modelo_novo.eval()
```

---

## 9. Exemplo completo (do zero até treinado)

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. Dados
X = torch.randn(500, 10)
y = (X.sum(dim=1, keepdim=True) > 0).float()
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# 2. Modelo
modelo = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 3. Loss e otimizador
loss_fn = nn.BCEWithLogitsLoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

# 4. Treino
for epoca in range(30):
    for xb, yb in loader:
        otimizador.zero_grad()
        pred = modelo(xb)
        perda = loss_fn(pred, yb)
        perda.backward()
        otimizador.step()
    if (epoca + 1) % 5 == 0:
        print(f"Época {epoca+1}: perda = {perda.item():.4f}")

# 5. Salvar
torch.save(modelo.state_dict(), "modelo.pth")
print("Treino concluído e modelo salvo!")
```

Copie e cole esse bloco inteiro num arquivo `.py` ou notebook — ele roda sozinho.

---

## 10. Checklist para adaptar a QUALQUER rede

1. **Defina o problema**: regressão, classificação binária ou multi-classe? → isso define a loss e a última camada.
2. **Defina o formato dos dados de entrada**: quantas features? É imagem (use `Conv2d`) ou tabular (use `Linear`)?
3. **Monte o `nn.Sequential`** empilhando camadas, garantindo que a saída de uma bate com a entrada da próxima.
4. **Escolha a loss** certa pra tabela da seção 5.
5. **Escolha o otimizador** (Adam é um bom padrão) e a taxa de aprendizado.
6. **Rode o loop de treino padrão** (os 5 passos da seção 6).
7. **Avalie** com `model.eval()` + `torch.no_grad()`.
8. **Salve** com `state_dict()`.

Esse esqueleto serve tanto pra um MLP simples quanto pra uma CNN — só muda o que vai dentro do `nn.Sequential` e o formato dos dados.

---

## Dúvidas comuns

- **"Erro de shape (tamanho incompatível)"**: quase sempre é porque a saída de uma camada não bate com a entrada da próxima. Confira os números dentro dos `Linear`/`Conv2d`.
- **A perda não diminui**: tente reduzir a `lr` (ex: de `1e-3` para `1e-4`), ou verifique se a loss escolhida é a certa para o problema.
- **Quero usar GPU**: basta mover modelo e dados com `.to(device)`, como mostrado no loop de treino.

## Funções de perda (`torch.nn`)

**Regressão**
- `nn.MSELoss()` — erro quadrático médio
- `nn.L1Loss()` — erro absoluto médio (MAE)
- `nn.SmoothL1Loss()` (também chamada Huber Loss) — mistura MSE e L1, mais robusta a outliers
- `nn.HuberLoss()` — versão configurável da Huber loss

**Classificação binária**
- `nn.BCELoss()` — espera que a saída já tenha passado por `Sigmoid`
- `nn.BCEWithLogitsLoss()` — aplica sigmoid internamente, mais estável numericamente (preferida)

**Classificação multi-classe**
- `nn.CrossEntropyLoss()` — combina `LogSoftmax` + `NLLLoss`, a mais usada
- `nn.NLLLoss()` — usada se você já aplicou `LogSoftmax` manualmente
- `nn.KLDivLoss()` — divergência KL entre distribuições

**Ranking / similaridade / distância**
- `nn.MarginRankingLoss()`
- `nn.HingeEmbeddingLoss()`
- `nn.CosineEmbeddingLoss()`
- `nn.TripletMarginLoss()`
- `nn.TripletMarginWithDistanceLoss()`
- `nn.PairwiseDistance()` (não é loss em si, mas usada em conjunto)

**Multi-label / multi-classe especial**
- `nn.MultiLabelSoftMarginLoss()`
- `nn.MultiLabelMarginLoss()`
- `nn.MultiMarginLoss()`
- `nn.SoftMarginLoss()`

**Segmentação de imagens / pixel a pixel**
- `nn.CrossEntropyLoss()` (também usada aqui, com `ignore_index`)
- `nn.BCEWithLogitsLoss()` (segmentação binária)

**Sequências / áudio / OCR**
- `nn.CTCLoss()` — usada em reconhecimento de fala e OCR (sequências não alinhadas)

**Distribuições de probabilidade**
- `nn.PoissonNLLLoss()`
- `nn.GaussianNLLLoss()`

**Outras**
- `nn.NLLLoss2d()` (versão antiga, hoje `NLLLoss` já lida com >2D)

---

## Otimizadores (`torch.optim`)

**Mais usados no dia a dia**
- `optim.SGD(params, lr, momentum=0.9)` — descida do gradiente estocástica, simples e robusta
- `optim.Adam(params, lr=1e-3)` — o padrão mais popular, boa escolha inicial
- `optim.AdamW(params, lr=1e-3, weight_decay=0.01)` — Adam com weight decay correto, muito usado em Transformers

**Variações de Adam**
- `optim.NAdam()` — Adam + Nesterov momentum
- `optim.RAdam()` — Adam com "rectification", mais estável no início do treino
- `optim.Adamax()` — variante baseada na norma infinita
- `optim.SparseAdam()` — para embeddings esparsos

**Adaptativos mais antigos**
- `optim.Adagrad()` — bom para dados esparsos, taxa de aprendizado decai muito
- `optim.Adadelta()` — extensão do Adagrad que corrige o decaimento excessivo
- `optim.RMSprop()` — muito usado antes do Adam, ainda popular em RL (aprendizado por reforço)

**Outros**
- `optim.Rprop()` — resilient backpropagation, usa apenas o sinal do gradiente
- `optim.ASGD()` — SGD com média (averaged SGD)
- `optim.LBFGS()` — otimizador de segunda ordem (quasi-Newton), usado quando o dataset é pequeno e cabe tudo em um batch

---

### Dica prática
Para 90% dos casos, comece com:
```python
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
```
e a loss de acordo com a tabela que te mandei antes (`MSELoss`, `BCEWithLogitsLoss` ou `CrossEntropyLoss`). Só vale a pena explorar as opções mais exóticas quando você já tiver um baseline funcionando e quiser otimizar performance.