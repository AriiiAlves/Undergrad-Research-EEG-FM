8x8 images of digits labeled repository: https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits

Instead, using code: https://scikit-learn.org/1.5/auto_examples/datasets/plot_digits_last_image.html

# Explicação Matemática: Minha primeira rede neural!

(fazer teoria aqui, e mostrar as funções)

# Notas

- Para cada função custo, as funções de treinamento precisam ser recalculadas através de derivadas parciais.

# Correções

## 1. Inicialização de pesos microscópica

No início, multipliquei a inicialização dos neurônios (aleatória) por 0.01. Como a rede tem 3 acamadas, os valores minúsculos são multiplicados em cadeia no forward pass e multiplicados novamente no backpropagation. O resultado é que o gradiante que chega na primeira camada é microscópico.

Os neurônios simplesmente não se movem.

Solução: Usar inicialização He (para a ReLU): multiplicar valores aleatórios por np.sqrt(2.0 / 64)

## 2. Arquitetura subdimensionada

Usei uma camada com apenas 10 neurônios. Tentar comprimir uma imagem de 64 pixels em apenas 10 características na primeira camada destrói a informação antes que a rede possa aprender.

Solução: 64 na primeira camada, 32 na segunda e 10 na final (classificação)

## 3. Taxa de aprendizado baixa

O erro quadrático MSE somado à função sigmoide na saída gera gradientes naturalmente baixos se comparado ao Softmax + Cross-Entropy. Com um passo k = 0.01, os pesos não se movem.

Solução: Aumentar passo para k = 0.1

## 4. Subtração no bias

No bias, é melhor uma soma ao invés de subtração. A subtração força o output a ter um valor negativo para ser desprezado (o que não é bom).

Exemplo: se estiver entre 0-1, não ativa, e >1 ativa.

## 5. Sigmoid no final layer ao invés de ReLU

A Sigmoide transforma os valores em uma probabilidade entre 0 e 1, enquanto a ReLU não tem limite máximo.

Em um problema de classificação, você quer que a rede expresse um nível de certeza (ex: "Tenho 90% de certeza que este número é um 5", que equivale à saída 0.90).