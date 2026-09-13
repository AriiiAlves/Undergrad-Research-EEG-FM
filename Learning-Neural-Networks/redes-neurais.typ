#set page(paper: "a4", margin: 2.5cm) // Configurações globais
#set text(font: "Libertinus Serif", size: 12pt) 
#set heading(numbering: "1.1") // Isso numera as seções (1, 1.1, etc.)
#show link: it => box(
  stroke: green + 1pt, // Define a borda verde de 1pt
  radius: 0pt,         // Arredonda levemente os cantos (opcional)
  inset: 0pt,          // Espaçamento entre o texto e a borda
  it                   // O conteúdo do link 
)
#show ref: it => box(
  stroke: orange + 1pt, // Define a borda verde de 1pt
  radius: 0pt,         // Arredonda levemente os cantos (opcional)
  inset: 0pt,          // Espaçamento entre o texto e a borda
  it                   // O conteúdo do link 
)
#show figure.where(kind: image): set figure(supplement: [Figura])

#outline(title: "Sumário")

#pagebreak()

= Introdução

== A rede neural e suas camadas

A rede neural consiste em um aglomerado de neurônios divididos por camadas. Em cada camada, cada neurônio recebe como input todas as saídas dos neurônios anteriores, aplicando pesos.

No exemplo, estamos tentando reconhecer o número 9. De modo teórico, pode-se assumir que:

*1° Camada* → Inputs (Pixels da imagem). É uma imagem 27x27 (729 pixels, 729 inputs)

*2° Camada* → Reconhecimento de pequenos padrões. Reconhece pequenas retas e curvas, onde cada tipo de reta/curva é uma feature, representada por um único neurônio.

*3° Camada* → Junta todos os padrões anteriores encontrados (retas e curvas) para formar features mais complexas. Agora é capaz de reconhecer círculos e retas grandes.

*4° Camada* → Camada de classificação. Utiliza os padrões encontrados pela 3° camada para classificar.

Cada camada reconhece padrões com base nas camadas anteriores, de modo que a camada seguinte reconhecerá padrões mais complexos, pois a camada atual já reconheceu alguns padrões pequenos.

#figure(
  image("./images/1.png", width: 60%),
)

== Neurônio

Cada neurônio coleta como input TODAS as saídas da camada anterior, e aplica pesos.

#figure(
  image("./images/2.png", width: 100%),
)

A ideia é que, mudando os pesos, os neurônios consigam enfatizar mais certas features específicas dos inputs.

A saída do neurônio não será um número grande. Usamos a função sigmoide (ou curva logística) para forçar o valor entre 0 e 1.

$ sigma = (1)/(1+e^(-x)) $

#figure(
  image("./images/3.png", width: 100%),
)

Assim, essa função será aplicada à saída do neurônio.

$ S=sigma(w_1a_1+...+w_n a_n) $

E não para por aí. Bom, o neurônio ativará quando o S = 0, pois o resultado será 0.5. Mas, se você quer que o neurônio ative somente a partir de S = 10, adicionamos um bias:

$ S=sigma(w_1a_1+...+w_n a_n-10) $

Assim:

- *Pesos*: Extraem característica com todos os inputs da camada anterior
- *Sigmoide*: Normaliza resultado (entre 0 e 1). Ativação do neurônio ocorre a partir de 0.5
- *Bias*: Controla quando o neurônio ativa.

= Modelo matemático de output para a rede neural
== Para um único neurônio

Para o primeiro neurônio da 2° camada, por exemplo, teremos a seguinte função:

$ S = sigma(
  w_1^(21) dot underbrace(sigma(w_1^(11) a_1 + ... + w_n^(11) a_n - b_(11)), "Saída 1° neurônio, 1° camada") + 
  ... + 
  w_n^(2n) dot underbrace(sigma(w_1^(1n) a_1 + ... + w_n^(1n) a_n - b_(1n)), "Saída último neurônio, 1° camada")
) $

De modo matricial ( $1 times M, M times N, N times 1)$:

$ S_(1 times 1)^(21) = sigma(
  underbrace(mat(w_1^(21), ..., w_m^(21)), "Pesos neurônio")
  
  underbrace(sigma(
    mat(
      w_1^(11), ..., w_n^(11);
      dots.v, dots.down, dots.v;
      w_1^(1m), ..., w_n^(1m)
    )
    underbrace(mat(a_1; ...; a_n), "Input")
    -
    mat(b^(11); ...; b^(1m))
  ), "1° camada")
  
  -
  underbrace(mat(b^(21)), "bias neurônio")
) $

Note que a saída será um único número ( $1 times M, M times N, N times arrow.r.double times 1$). Esse é o output do neurônio.

== Para toda a camada

Se ampliarmos para todos os neurônios da 2° camada:

$ S_(m times 1)^2 = underbrace(sigma(
  underbrace(mat(
    w_1^(21), ..., w_n^(21);
    dots.v, dots.down, dots.v;
    w_1^(2m), ..., w_n^(2m)
  ), "Pesos")

  underbrace(sigma(
    mat(
      w_1^(11), ..., w_n^(11);
      dots.v, dots.down, dots.v;
      w_1^(1m), ..., w_n^(1m)
    )
    underbrace(mat(a_1; ...; a_n), "Input")
    -
    mat(b^(11); ...; b^(1m))
  ), "1° camada")

  -
  underbrace(mat(b^(21); ...; b^(2m)), "bias")
), "2° camada") $

Podemos simplificar para:

$ S^2_("m x 1")=sigma(W_2 times sigma(W_1A-B_1)-B_2) $

Onde:

- $sigma:R arrow.r R = (1)(1+e^(-x))$  (Função sigmoide)
- $W_k$ → Matriz de pesos da camada K. Cada linha é um neurônio.
- A → Input
- $B_k$ → Matriz de bias da camada K. Cada linha é um neurônio.

== Para toda a rede

Note que cada camada consiste em uma única função:

$ S_k= sigma(W_k S_{k-1}-B_k)" , "S_0=A $

Onde:

- $S_k$ → Vetor de output dos M neurônios da camada anterior.
- $sigma:R arrow.r R =(1)(1+e^(-x))$  (Função sigmoide) → Pode ser trocada por outra função qualquer
- $W_k$ → Matriz de pesos da camada K. Cada linha é um neurônio. Possui N pesos/colunas
- A → Vetor de input (N entradas)
- $B_k$ → Vetor de bias da camada K. Cada linha é um neurônio.

== ReLU: Opção melhor do que Sigmoide

As sigmoides não são usadas mais, pois tornam mais difícil o treinamento. Atualmente, é usada uma função mais certeira, a ReLU.

$ "ReLU"(x)=max(0,x) $

Sim, é simples assim. Desse modo, se $x>0$, o neurônio será ativado.

#figure(
  image("./images/4.png", width: 100%),
)

As ReLU são utilizadas somente nas camadas intermediárias, pois propagam bem as features.

== Camada final: Sem ReLU

Na camada final, precisamos de uma classificação binária (é/não é). Não faz sentido 0.91 quando se espera 1 ou 0. Assim, usamos uma função de ativação binária.

A ReLU é usada somente para hidden layers, para que os outputs possam transferir as features adequadamente.

Abaixo usaremos o Erro Quadrático Médio (MSE) para o treinamento. Portanto, a melhor função para a última camada com essa função custo é a **Sigmoide**.

#align(center)[
  #block(
    fill: luma(230),
    inset: 8pt,
    radius: 4pt,
  )[
    *Sigmoide*

    Função que “espreme” qualquer valor real para o intervalo (0,1)

    $ S(z)=(1)/(1+e^(-z)) $
  ]
]

=== Problema matemático (saturação) com sigmoide

Ao usar MSE com sigmoide, a derivada na fórmula de ajuste torna-se um problema:

$ S'=S dot (1-S) $

Isso significa que se o neurônio estiver “muito convicto” (saída perto de 0 ou 1), o valor de $S'$ será quase zero. Isso significa que o ajuste dos pesos será minúsculo, mesmo que o erro seja grande. Isso é chamado de vanishing gradient.

#table(
  columns: 3, // Define 3 colunas de tamanhos iguais
  [*Objetivo*], [*Função de Saída*], [*Função de Custo*],
  [Classificação], [Softmax], [Cross-Entropy],
  [Regressão (prever)], [Linear], [MSE (Medium Square Error)],
  [Classificação Binária], [Sigmoide], [Binary Cross-Entropy]
)

= Como a rede neural aprende

== Custo do erro (módulo do vetor erro)
O custo do erro de uma rede neural é a soma do quadrado das diferenças entre o vetor resultado e o vetor rótulo.

#figure(
  image("./images/5.png", width: 90%),
)

Na verdade, isso é o módulo do vetor erro:

- B -> Vetor Rótulo \
- A -> Vetor Resultado \
- $||arrow(A B)||$ -> Módulo do vetor erro (diferença)

Esse custo pode ser tratado como uma função que recebe uma função de milhares de parâmetros (pesos sendo aplicados a outputs da camada anterior em cascata) e cospe um único número.

$ C(w_1, w_2, ..., w_k) = e $

Na verdade, existem muitas funções custo diferentes. Essa é a *função custo MSE (Medium Square Error)*.

== Achando peso ideal em $RR^1$
Mas vamos imaginar que temos um único peso na rede inteira. Assim:

$ C(w) = (w_1 a_1 - y)^2 $

E, se fazermos um gráfico $w times y$, veremos que o erro pode ser minimizado em um ponto:

#figure(
  image("./images/6.png", width: 80%),
)

Isso nos ajuda a achar o valor ideal para esse peso!

Mas nossa função real de custo leva muito mais parâmetros do que isso, o que nos leva a $RR^n$.

== Achando peso ideal em $RR^n$ com Função Custo MSE
=== Introdução
Agora, temos que levar em conta todos os pesos:

$ arrow(C)(w_1, w_2, ..., w_k) = (S - Y)^2 $

Como podemos achar os pesos ideais que minimizam este erro? Bom, podemos tratar os pesos como dimensões, e aplicar Cálculo II (Cálculo com muitas variáveis) para encontrar o mínimo global.

=== Gradiente e gradiente descendente

#align(center)[
  #block(
    fill: luma(230),
    inset: 8pt,
    radius: 4pt,
    align(left)[
      *Gradiente* $nabla$ \
      Lembra do gradiente? Ele leva de $RR^1 -> RR^n$.
      
      $ nabla [f(p_1, p_2, ..., p_n) : RR^n -> RR^1] = (partial f / partial p_1, partial f / partial p_2, ..., partial f / partial p_n) $
      
      Pela definição, o vetor gradiente $nabla f$:
      - Aponta para a direção de maior crescimento de $f$
        - Do mesmo modo, $-nabla f$ aponta para a direção de maior decrecimento (gradiente descendente).
      - Se $nabla f = arrow(0)$, não importa onde você vá, o crescimento/decrescimento sempre será o mesmo. Nesse ponto, caímos em um mínimo/máximo local (mesmo conceito de derivada = 0 em $R^2$)
        - Se for mínimo: Não importa a direção, $f$ decresce igualmente
        - Se for máximo: Não importa a direção, $f$ decresce igualmente \
        - Em uma tigela, o mínimo global é o fundo.
      - É perpendicular às curvas de nível ($f(p_1, ..., p_n) = K$, superfícies de dimensão $n-1$ onde a função possui sempre o mesmo valor).
      - Sua magnitude representa o "tamanho" desse crescimento.

      #figure(
        image("./images/7.png", width: 50%),
      )

      Aqui, encontramos um mínimo local, com $f(p_1, ..., p_n) = z$

      #figure(
        image("./images/8.png", width: 50%),
      )

      Aqui, o gradiente nos diz que se formos naquela direção, a função vai crescer mais do que em qualquer outra direção.
    ]
  )
]

Para ajustar os pesos, o gradiente vai nos ajudar a saber qual direção minimiza o erro.

#figure(
  image("./images/9.png", width: 60%),
)

Aqui, o gradiente (derivada comum, pois estamos em $R^2$) apontará para a direita, que é onde a função cresce mais. Portanto, devemos ir na direção $-nabla f$. \
O conceito de gradiente descendente consiste em fazer vários ajustes nesse peso (não um ajuste único) para a direção de maior decrescimento, de modo a manter ele lá. \
E não importa quão aleatório seja o peso inicial, ele sempre vai ser mantido em um mínimo local.

Antes:

#figure(
  image("./images/10.png", width: 60%),
)

Depois:

#figure(
  image("./images/11.png", width: 60%),
)

=== Aplicando gradiente descendente à Função Custo MSE
Vamos modelar a função custo de modo não matricial. Para um único neurônio da 2° camada...

$ S_(21) = sigma(w_1^(21) dot S_(11) + ... + w_n^(21) dot S_(11) - b_(21)) => W_2 times S_1 - b_2 $

E a função custo para esse único neurônio:

$ arrow(C)(w_1^(21), ..., w_n^(21)) = (S - Y)^2 " (Função Custo Total)" $

$ C = (S_(21) - y)^2 $
$ C = S_(21)^2 - 2y S_(21) + y^2 $

Obs: Vou definir: \
$sigma(W_k) = S_k$ \
Assim, temos uma função final, para todos os neurônios da 2° camada:

$ C(S_(21), y) = S_(21)^2 - 2y S_(21) + y^2 $

Vamos obter o primeiro termo do gradiente (a função de normalização tem que ser derivada também!) \
Calculando as derivadas individuais:

$ d(S_(21)^2) / (d w_1^(21)) = 2 d(S_(21)) / (d w_1^(21)) = 2 sigma'(W_(21)) d(W_(21)) / (d w_1^(21)) = 2 sigma'(W_(21)) S_(11) $

$ d(2y S_(21)) / (d w_1^(21)) = 2y sigma'(W_(21)) d(W_(21)) / (d w_1^(21)) = 2y sigma'(W_(21)) S_(11) $

$ d(y^2) / (d w_1^(21)) = 0 $

Por fim:

$ (d C) / (d w_1^(21)) = 2 S_(21) dot sigma'(W_(21)) dot S_(11) - 2y dot sigma'(W_(21)) dot S_(11) $

Assim, se queremos minimizar o erro nessa direção, basta somar os pesos ao gradiente descendente (negativo do gradiente), para andar na direção que minimiza o erro.

$ w_1^(21) = w_1^(21) - K (d C) / (d w_1^(21)) (S_(21), y, A) $
$ w_1^(21) = w_1^(21) - K (2 S_(21) dot sigma'(W_(21)) dot S_(11) - 2y dot sigma'(W_(21)) dot S_(11)) $
$ w_1^(21) = w_1^(21) - 2K (S_(21) - y) dot sigma'(W_(21)) dot S_(11) $

Tcharam! Esse é o ajuste de peso de um único neurônio, por gradiente descendente. \
O termo K é o tamanho do "passo" que queremos dar, e é arbitrário. Se for muito grande, vamos passar muitas vezes do ponto ideal. Se for muito pequeno, serão necessárias muitas iterações.

=== Conclusão para última camada
O gradiente da última camada é:

$ (partial C) / (partial W) = underbrace(2 (S_n - Y) dot.circle S'_n, delta) dot S_("ant")^T $

Dimensões: $delta_(n times 1), W_(m times n), S_(n times 1), S'_(n times 1), Y_(n times 1)$. A transposta ao final faz com que haja produto externo ao invés de produto interno, e a dimensionalidade para W é mantida.

$ S_n = sigma(W_n dot S_(n-1) - b) $
$ S'_n = sigma(W_n dot S_(n-1) - b) $

Podemos generalizar o ajuste de peso na última camada (W = matriz de pesos agora):

$ W_n = W_n - K [(2(S_n - Y) dot.circle S'_n) dot S_("ant")^T] $

Na forma matricial: \
A primeira multiplicação é elemento a elemento ($dot.circle$), e não um produto escalar. Isso pois a derivada parcial afeta localmente cada peso somente, todos são independentes. \
Note que o ajuste de uma camada depende da camada anterior. Assim, o ajuste é feito do fim ao início. Isso é chamado de backpropagation.

=== Camadas anteriores

E para as camadas anteriores, ao invés de fazermos $(partial C) / (partial W_n)$, fazemos $(partial C) / (partial W_(n-1))$. Cuidado, eu tava tentando obter $(partial C) / (partial W_(n-1))$ a partir de $(partial C) / (partial W_n)$, o que é errado, acaba derivando duas vezes. \
Bom, fazendo isso para o layer $n-1$, temos (Obs: $S_("ant")$ seria do atual (penúltimo layer) e $S_("ant"^2)$ seria do anterior):

$ (partial C) / (partial W_(n-1)) = underbrace(W^T dot delta dot.circle S'_("ant"), delta_("ant")) dot S_("ant"^2)^T $

Generalizando para qualquer hidden layer:

$ (partial C) / (partial W_i) = delta_i dot S_("ant")^T $
$ delta_i = W_(i+1)^T dot delta_(i+1) dot.circle S'_i $

Para a primeira camada, $S_("ant") = "Input"$. Ah, e W é transposto, pois é o próximo W. Assim, temos que transpor para as linhas/colunas baterem (estamos voltando!) \
OBS: $delta_i = "Antes do ajuste"$. \
E, assim como fizemos na última camada, o ajuste é dado por:

$ W = W - K (partial C) / (partial W) $

#align(center)[
  #block(
    fill: luma(230),
    inset: 8pt,
    radius: 4pt,
    align(left)[
      *Produto de Hadamard* ($dot.circle$) \
      Multiplicação elemento a elemento.
    ]
  )
]

#align(center)[
  #block(
    fill: luma(230),
    inset: 8pt,
    radius: 4pt,
    align(left)[
      *Inner Product / Outer Product em Vetores* \
      Outer Product:
      $ mat(a_1; a_2)_(2 times 1) dot mat(b_1, b_2)_(1 times 2) = mat(a_1 b_1, a_1 b_2; a_2 b_1, a_2 b_2) $
      
      Inner product:
      $ mat(a_1, a_2)_(1 times 2) dot mat(b_1; b_2)_(2 times 1) = mat(a_1 b_1 + a_2 b_2)_(1 times 1) $
      
      (Do outer product eu não sabia).
    ]
  )
]


=== Resumo: Ajuste de peso em $RR^n$ (Função Custo: MSE)
==== Ajuste de peso para última camada 
$ (partial C) / (partial W) = underbrace(2 (S_n - Y) dots.c S'_n, delta) dot S_"ant"^T $

$ W_n = W_n - k (partial C) / (partial W) $

==== Ajuste de peso para hidden layers
$ (partial C) / (partial W_i) = delta_i dot S^T_"ant" $

$ delta_i = W_(i+1)^T dot delta_(i+1) dots.c S'_i $

$ W_i = W_i - k (partial C) / (partial W)_i $

== Achando Bias ideal em $RR^n$ para Função Custo Qualquer

O Bias também deve ser ajustado (aprendido) com o tempo. É ele quem define quando o neurônio ativa ou não.

$ (partial C)/(partial b) = (partial C)/(partial S) dot (partial S)/(partial z) dot (partial z)/(partial b) $

Obs: $S = sigma(z)$.

$ delta_"camada" = (partial C)/(partial S) dot (partial S)/(partial z) $

== Achando Peso ideal em $RR^n$ para Função Custo qualquer

Para achar o ajuste de pesos, basta repetir o mesmo processo, mas com a Função Custo Escolhida.

1. Para a última camada: Dada a função custo $C(S, Y)$ e $S = W times S_"ant" + b_2$, ache $(partial C)/(partial W)$. O $delta$ é o termo de $(partial C)/(partial W)$ que multiplica a saída anterior.

2. Para as camadas anteriores: Ache $(partial C)/(partial W_(n-1))$, depois $(partial C)/(partial W_(n-2))$, e veja se pode se induzir um padrão. O delta é o termo de $(partial C)/(partial W_i)$ que multiplica a saída anterior.

No código:

1. *Forward Pass*: Gere a saída S da rede neural, passando por todas as camadas.
2. *Backward Pass*: Calcule os deltas e gradientes de uma só vez
3. *Update*: Mude as matrizes de pesos com os gradientes
4. *Update*: Mude os bias com os deltas, utilizando a taxa de aprendizado ($k$)