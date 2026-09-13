import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

# Carrega dataset de dígitos
digits = datasets.load_digits()

# Separa 20% para bench (teste) e 80% para treino. X = dados, Y = label
X_train, X_test, Y_train, Y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Normalizando dados (O brilho do pixel varia de 0 a 16)
X_train = X_train / 16.0
X_test = X_test / 16.0

# Mostra informação
print(digits.keys()) # Que informação o dataset contém?

## digits.data -> Matriz de dados. Nesse caso, os pixels.
## digits.target -> Labels
## digits.DESCR -> Descrição do dataset pelo autor

print(digits.data[-1]) # Última imagem (array) do dataset 
print(digits.data.shape)   # (1797, 64) -> 1797 imagens, cada uma com 64 pixels (8x8)
print(digits.target.shape) # (1797,)   -> 1797 labels

## Mostra o último dígito
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
print(digits.target[-1]) # Label
plt.show()

# Rede neural
'''
1 Layer:

[w11 w12 w13 ... w1n]   [p1]    [b1]   [r1]    [func(r1)]
[w21 w22 w23 ... w2n] X [p2]  - [b2] = [r2] => [func(r2)] (output of 1 Layer)
[wn1 wn2 wn3 ... wnn]   [pn]    [b3]   [r3]    [func(r3)]

Uma linha = um neurônio e seus pesos

Vamos ter 3 camadas.

- Layer 1: Pequenos padrões
- Layer 2: Grandes padrões
- Layer 3: Classificador
'''

# Função ReLU: usada nas hidden layers somente.
def reluFunc(v):
    if(v > 0):
        return v
    else:
        return 0

def reluDerivativeFunc(v):
    if(v > 0):
        return 1
    else:
        return 0

# Softmax é mais simples se usada com a função-custo cross-entropy.
def softmax(v):
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v)

def sigmoidFunc(v):
    return 1/(1 + np.exp(-v))

def sigmoidDerivativeFunc(v):
    return np.exp(-v) / (pow(1 + np.exp(-v), 2))

# Transforma a função para operar com matrizes
relu = np.vectorize(reluFunc)
reluDerivative = np.vectorize(reluDerivativeFunc)
sigmoid = np.vectorize(sigmoidFunc)
sigmoidDerivative = np.vectorize(sigmoidDerivativeFunc)

# Hidden Layers

print("Creating matrices...")

## Usando inicialização He (multiplicar por np.sqrt(2.0 / 64))

# Input é um vetor 64x1
W1 = np.random.randn(64, 64) * np.sqrt(2.0 / 64) # Matriz de pesos (64x64). 64 neurônios, 64 inputs/weights
L1B = np.zeros((64, 1)) # Vetor Bias, deve ser inicializado com zeros
W2 = np.random.randn(32, 64) * np.sqrt(2.0 / 64) # Matriz de pesos (32x64). 32 neurônios, 64 inputs/weights
L2B = np.zeros((32, 1)) # Vetor Bias. Deve ser incializado com zeros

# Output Layer (Classificação)
W_OUT = np.random.randn(10, 32) * np.sqrt(2.0 / 64)
OUTB = np.zeros((10, 1))

N = X_train.shape[0] # N vetores de 64 elements (treino)
k = 0.1 # Passo (com o qual os pesos são ajustados)
rounds = 100 # Repetições

# TREINAMENTO
for n in range(0, rounds):
    print(f"Running round {n}/{rounds}")
    for i in range(0, N):
        # Obtendo input
        IN = np.array(X_train[i]).reshape(-1, 1) # Transformar em vetor nx1

        # Calcular saídas dos hidden layers
        L1OUT = relu(np.dot(W1, IN) + L1B)
        L1OUT_D = reluDerivative(np.dot(W1, IN) + L1B)

        L2OUT = relu(np.dot(W2, L1OUT) + L2B)
        L2OUT_D = reluDerivative(np.dot(W2, L1OUT) + L2B)

        # Sigmoid ao invés de ReLU na última camada (saída deve ser probabilidade)
        S = sigmoid(np.dot(W_OUT, L2OUT) + OUTB)
        S_D = sigmoidDerivative(np.dot(W_OUT, L2OUT) + OUTB)

        Y = np.zeros((10, 1)) # 10x1 vector
        Y[Y_train[i], 0] = 1 # Only works because Y_train[i] = int

        # Backpropagation

        ## 1° Passo: Calculando Gradientes

        ### Final Layer
        OUT_DELTA = 2 * (S-Y) * S_D # Multiplicação elemento por elemento (*)
        OUT_GRAD = np.dot(OUT_DELTA, L2OUT.T)

        W2_DELTA = np.dot(W_OUT.T, OUT_DELTA) * L2OUT_D
        W2_GRAD = np.dot(W2_DELTA, L1OUT.T)

        W1_DELTA = np.dot(W2.T, W2_DELTA) * L1OUT_D
        W1_GRAD = np.dot(W1_DELTA, IN.T)

        ## 2° Passo: Mudando Matrizes de Pesos com os Gradientes
        W_OUT -= k * OUT_GRAD
        W2 -= k * W2_GRAD
        W1 -= k * W1_GRAD

        ## 3° Step: Mudando Bias com os Deltas
        OUTB -= k * OUT_DELTA
        L2B -= k * W2_DELTA
        L1B -= k * W1_DELTA

# Nota: O classificador NN deve mostrar a classificação EXATA (apenas uma). Uma classificação é considerada o maior neurônio que dispara.

# BENCHMARK
print("Train complete. Running benchmark.")
N = X_test.shape[0]

correct = 0
wrong = 0

for i in range(0, N):
        # Obter Input
        IN = np.array(X_test[i]).reshape(-1, 1) # Transformar em vetor nx1

        # Obter saída dos Hidden Layers
        L1OUT = relu(np.dot(W1, IN) + L1B)
        L2OUT = relu(np.dot(W2, L1OUT) + L2B)
        S = sigmoid(np.dot(W_OUT, L2OUT) + OUTB)

        final_class = np.argmax(S) # Retorna index do maior valor dado pela última camada
        
        if final_class == Y_test[i]: correct += 1
        else: wrong += 1

print(f"Correct: {correct}\nWrong: {wrong}")
print(f"Accuracy: {correct/(correct + wrong) * 100} %")

# Salvando matrizes de pesos
np.save('W1.npy', W1)
np.save('W2.npy', W2)
np.save('W_OUT.npy', W_OUT)

# Carregando matrizes de pesos
# W1 = np.load('W1.npy')