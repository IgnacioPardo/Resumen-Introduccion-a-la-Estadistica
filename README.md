# Resumen Introducción a la Estadistica


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Resumen Introducción a la Estadistica](#resumen-introducción-a-la-estadistica)
  - [Esperanza y Varianza](#esperanza-y-varianza)
    - [Esperanza](#esperanza)
    - [Varianza](#varianza)
    - [Esperanza y Varianza Condicional](#esperanza-y-varianza-condicional)
    - [Desvio Estandard](#desvio-estandard)
    - [Ley de Esperanza Total](#ley-de-esperanza-total)
      - [Ejemplo](#ejemplo)
    - [Covarianza](#covarianza)
    - [Correlación](#correlación)
  - [Distribuciones](#distribuciones)
    - [Formulas Discretas](#formulas-discretas)
    - [Formulas Continuas](#formulas-continuas)
    - [Distribución Normal](#distribución-normal)
    - [Distribución Binomial](#distribución-binomial)
    - [Distribución de Bernoulli](#distribución-de-bernoulli)
    - [Distribución Uniforme continua](#distribución-uniforme-continua)
    - [Distribución Uniforme Discreta](#distribución-uniforme-discreta)
    - [Distribución de Poisson](#distribución-de-poisson)
    - [Distribución Geométrica](#distribución-geométrica)
    - [Distribución Hipergéometrica](#distribución-hipergéometrica)
    - [Distribuciones de variable continua](#distribuciones-de-variable-continua)
  - [El Teorema del Límite Central](#el-teorema-del-límite-central)
- [Vectores Aleatorios](#vectores-aleatorios)
    - [Probabilidad Conjunta](#probabilidad-conjunta)
    - [Probabilidad Marginal](#probabilidad-marginal)
    - [Bayes](#bayes)
- [Comandos R](#comandos-r)

<!-- /code_chunk_output -->

## Esperanza y Varianza

### Esperanza
$${\operatorname {E} [X]=\sum _{i=1}^{n}x_{i}\operatorname {P} [X=x_{i}]}$$

Si $X$ y $Y$ son variables aleatorias con esperanza finita y ${a,b,c\in \mathbb {R} }$ son constantes entonces

- ${\operatorname {E} [c]=c}$
- ${\operatorname {E} [cX]=c\operatorname {E} [X]}$
- Si ${X\geq 0}$ entonces ${\operatorname {E} [X]\geq 0}$
- Si ${X\leq Y}$ entonces ${\operatorname {E} [X]\leq \operatorname {E} [Y]}$
- Si $X$ está delimitada por dos números reales, $a$ y $b$, esto es ${a<X<b}$ entonces también lo está su media, es decir, ${a<\operatorname {E} [X]<b}$
- Si ${Y=a+bX}$, entonces ${\operatorname {E} [Y]=\operatorname {E} [a+bX]=a+b\operatorname {E} [X]}$

$${\operatorname {E} [X+Y]=\operatorname {E} [X]+\operatorname {E} [Y]}$$

$${\operatorname {E} [cX]=c\operatorname {E} [X]}$$

- Si $X$ y $Y$ son variables aleatorias independientes entonces

$${\operatorname {E} [XY]=\operatorname {E} [X]\operatorname {E} [Y]}$$

### Varianza
$$\operatorname {Var}[X]=\operatorname {E} [X^{2}]-\operatorname {E} [X]^{2} ⟹ \operatorname {E} [X^{2}] = \operatorname {Var}[X] + \operatorname {E} [X]^{2}$$

Sean $X$ y $Y$ dos variables aleatorias con varianza finita y ${a\in \mathbb {R} }$

- ${\operatorname {Var} (X)\geq 0}$
- ${\operatorname {Var} (a)=0}$
- ${\operatorname {Var} (aX)=a^{2}\operatorname {Var} (X)}$
- ${\operatorname {Var} (X+Y)=\operatorname {Var} (X)+\operatorname {Var} (Y)+2\operatorname {Cov} (X,Y)}$, donde ${\operatorname {Cov} (X,Y)}$ denota la covarianza de $X$ e $Y$
- ${\operatorname {Var} (X+Y)=\operatorname {Var} (X)+\operatorname {Var} (Y)}$ si $X$ y $Y$ son variables aleatorias independientes.
- ${\operatorname {Var} (Y)=\operatorname {E} (\operatorname {Var} (Y|X))+\operatorname {Var} (\operatorname {E} (Y|X))}$ cálculo de la Varianza por Pitágoras, dónde ${Y|X}$ es la variable aleatoria condicional $Y$ dado $X$.

### Esperanza y Varianza Condicional

$${\operatorname {E} (X|Y=y)=\sum _{x\in {\mathcal {X}}}x\ \operatorname {P} (X=x|Y=y)=\sum _{x\in {\mathcal {X}}}x{\frac {\operatorname {P} (X=x,Y=y)}{\operatorname {P} (Y=y)}}}$$

$$\operatorname{Var}[Y|X=x]=\operatorname{E}[Y^2|X=x]−{\operatorname{E}[Y|X=x]}^2=\sum _{y}y^2 \operatorname{p}(y|x)−\{\operatorname{E}[Y|X=x]\}^2$$

### Desvio Estandard

$${\operatorname{SD}(X) =  \sigma ={\sqrt {{\text{Var}}(X)}}} \implies{\sigma ^{2}={\text{Var}}(X)}$$

### Ley de Esperanza Total

$$\operatorname {E}(C) = \operatorname {E}(\operatorname {E}(C ∣ N)) = \operatorname{E} (\operatorname{E}(C) ⋅ N)$$

$$\operatorname {E}(N ⋅ Y ) = \operatorname {E}(\operatorname {E}(N ⋅ Y ∣ N)) = \operatorname {E}(g(N))$$

$$\operatorname {g}(n) = \operatorname {E}(N ⋅ Y ∣ N = n)$$


#### Ejemplo

Si ${N}$ es una v.a. de ${μ_{n}}$ y ${\sigma^2_{n}}$, ${X_i}$ es una v.a. de ${μ_{x_i}}$ y ${\sigma^2_{x_i}}$, ${N}$ y ${X_i}$ son iid., y ${K} = {\sum _{i=0}^N {X_i}}$

$${\begin{aligned}\operatorname{E}(K) & = \operatorname{E}({\sum _{i=0}^N {X_i}}) \\ & = \operatorname{E}(\operatorname{E}(K|N)) \\& = \operatorname{E}(\operatorname{g}(N))\\& = {\operatorname{E}(N ⋅ \operatorname{E}(X_i))} \\ &= {\operatorname{E}(N ⋅ \mu_{x_i})} \\ &= {\mu_{x_i} ⋅ \operatorname{E}(N)} \\ &= {\mu_{x_i} ⋅ \mu_N}\end{aligned}\begin{aligned}\operatorname{g}(n) &= \operatorname{E}(K|N=n) \\& = \operatorname{E}({\sum _{i=0}^n {X_i}}|N=n) \\& = \sum _{i=0}^n{\operatorname{E}(X_i|N=n)} \\& = \sum _{i=0}^n{\operatorname{E}(X_i)} \\& = n ⋅ {\operatorname{E}(X_i)}\end{aligned}}$$

### Covarianza

$${\operatorname {Cov} (X,Y)=\operatorname {E} \left[XY\right]-\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]}$$

### Correlación

$$ρ_{xy} = {\frac{\operatorname{cov}_{xy}}{\sigma_x\sigma_y}} = {\frac{\operatorname{cov}_{xy}}{\operatorname{SD}(x)\operatorname{SD}(y)}}$$
## Distribuciones

### Formulas Discretas
$$\operatorname{F}_{X}(x) = \mathrm {Prob} (X\leq x)$$

$$\operatorname{P}(X \leq b)=\operatorname{P}(X\leq a)+\operatorname{P}(a < X \leq b)$$

$${\operatorname{P}(a < X \leq b)=\operatorname{P}(X\leq b)-\operatorname{P}(X\leq a)}$$

$${\operatorname{F}(x)=\operatorname{P}(X\leq x)=\sum _{k=-\infty }^{x}f(k)}$$

$${\operatorname{P}(a < X \leq b)=\operatorname{F}(b)-\operatorname{F}(a)}$$

### Formulas Continuas

$${\operatorname{F_x}(x) = \int_{-\infty}^{x}{f_x(u)du}}$$

$${\operatorname{P}(a < X < b) = \int_a^b{f(x)dx} = \operatorname{F_x}(b) -\operatorname{F_x}(a)} $$

$${\operatorname{P}(a < X < b) = \operatorname{P}(a < X \leq b) = \operatorname{P}(a \leq X < b)= \operatorname{P}(a \leq X \leq b)}$$

### Distribución Normal
Si ${X\sim N(\mu ,\sigma ^{2})}$ y ${a,b\in \mathbb {R} }$, entonces ${aX+b\sim N(a\mu +b,a^{2}\sigma ^{2})}$

Si ${X\,\sim N(\mu ,\sigma ^{2})\,}$, entonces ${Z={\frac {X-\mu }{\sigma }}\!}$ es una variable aleatoria normal estándar: $Z$ ~ $N(0,1)$.

$${X\,\sim N(\mu, \sigma ^{2}) ⟹ Z={\frac {X-\mu }{\sigma }} \sim N(0,1)}$$

$${Z\sim N(0,1) ⟹ X = \sigma Z + μ ∼ N(\mu, \sigma ^{2})}$$

### Distribución Binomial

Si una variable aleatoria discreta $X$ tiene una distribución binomial con parámetros $n\in\mathbb{N}$ y $p$ con ${0<p<1}$ entonces escribiremos ${X\sim \operatorname {Bin} (n,p)}$

La **distribución binomial**, describe el número de aciertos en una serie de n experimentos independientes con posibles resultados binarios, es decir, de «sí» o «no», todos ellos con probabilidad de acierto p y probabilidad de fallo q = 1 − p.


$${\operatorname {P} [X=x]={n \choose x}p^{x}(1-p)^{n-x}}$$

$${\!{n \choose x}={\frac {n!}{x!(n-x)!}}}$$

$${F_{X}(x)=\operatorname {P} [X\leq x]=\sum _{k=0}^{x}{n \choose k}p^{k}(1-p)^{n-k}}$$

$${\operatorname {E} [X]=np} \ , \ \ {\operatorname {Var} [X]=np(1-p)}$$

### Distribución de Bernoulli

Si ${X}$ es una variable aleatoria discreta que mide el "número de éxitos" y se realiza un único experimento con dos posibles resultados denominados éxito y fracaso, se dice que la variable aleatoria ${X}$ se distribuye como una Bernoulli de parámetro ${p}$ con ${0<p<1} $ y escribimos ${X\sim \operatorname {Bernoulli} (p)}$

$${\operatorname {P} [X=x]=p^{x}(1-p)^{1-x}\qquad x=0,1}$$

$${F(x)={\begin{cases}0&x<0\\1-p&0\leq x<1\\1&x\geq 1\end{cases}}}$$

$${\operatorname {E} \left[X\right]=\operatorname {E} \left[X^n\right]=p}$$

$${{\begin{aligned}\operatorname {Var} \left[X\right]&=\operatorname {E} [X^{2}]-\operatorname {E} [X]^{2}\\&=p-p^{2}\\&=p\left(1-p\right)\end{aligned}}}$$

Si ${X_{1},X_{2},\dots ,X_{n}}$ son $n$  variables aleatorias independientes e identicamente distribuidas con ${X_{i}\sim \operatorname {Bernoulli} (p)}$ entonces la variable aleatoria ${X_{1}+X_{2}+\dots +X_{n}}$ sigue una distribución binomial con parámetros $n$ y $p$, es decir

$${\sum _{i=1}^{n}X_{i}\sim \operatorname {Bin} (n,p)}$$

### Distribución Uniforme continua
Si $X$ es una variable aleatoria continua con distribución uniforme continua entonces escribiremos ${X\sim \operatorname {U} (a,b)}$ o ${X\sim \operatorname {Unif} (a,b)}$

$${f_{X}(x)={\frac {1}{b-a}}}$$

$${{\begin{aligned}F_{X}(x)={\frac {x-a}{b-a}}\end{aligned}}}$$

$${\operatorname {E} [X]={\frac {a+b}{2}}}$$

$${\operatorname {Var} (X)={\frac {(b-a)^{2}}{12}}}$$


### Distribución Uniforme Discreta

Si $X$ es una variable aleatoria discreta cuyo soporte es el conjunto ${\{x_{1},x_{2},\dots ,x_{n}\}}$ y tiene una distribución uniforme discreta entonces escribiremos ${X\sim \operatorname {Uniforme} (x_{1},x_{2},\dots ,x_{n})}$

La **distribución uniforme discreta**, recoge un conjunto finito de valores que son resultan ser todos igualmente probables. Esta distribución describe, por ejemplo, el comportamiento aleatorio de una moneda, un dado, o una ruleta de casino equilibrados (sin sesgo).

$${\operatorname {P} [X=x]={\frac {1}{n}}}$$

$${\operatorname {E} [X]={\frac {1}{n}}\sum _{i=1}^{n}x_{i}}$$


$${\operatorname {Var} (X)={\frac {1}{n}}\sum _{i=1}^{n}(x_{i}-\operatorname {E} [X])^{2}}$$

### Distribución de Poisson

Sea ${\lambda >0}$ y $X$ una variable aleatoria discreta, si la variable aleatoria $X$ tiene una distribución de Poisson con parámetro $\lambda$  entonces escribiremos ${X\sim \operatorname {Poisson} (\lambda )}$ o ${X\sim \operatorname {Poi} (\lambda )}$

$${\operatorname {P} [X=k]={\frac {e^{-\lambda }\lambda ^{k}}{k!}}}$$

$${\operatorname {E} [X]=\operatorname {Var} (X)=\lambda }$$

Como consecuencia del teorema central del límite, para valores grandes de $\lambda$ , una variable aleatoria de Poisson $X$ puede aproximarse por otra normal dado que el cociente 
$${Y={\frac {X-\lambda }{\sqrt {\lambda }}}}$$ converge a una distribución normal de media 0 y varianza 1.


### Distribución Geométrica

Si una variable aleatoria discreta ${X}$ sigue una distribución geométrica con parámetro ${0<p<1}$  entonces escribiremos ${X\sim \operatorname {Geometrica} (p)}$ o simplemente ${X\sim \operatorname {Geo} (p)}$

La **distribución geométrica**, describe el número de intentos necesarios hasta conseguir el primer acierto.

$${\operatorname {P} [X=x]=p(1-p)^{x}}$$

$${{\begin{aligned}\operatorname {P} [X\leq x]=1-(1-p)^{x+1}\end{aligned}}}$$

$${\operatorname {E} [X]={\frac {1}{p}}}$$

$${\operatorname {Var} (X)={\frac {1-p}{p^{2}}}}$$

### Distribución Hipergéometrica

Una variable aleatoria discreta $X$ tiene una distribución hipergeométrica con parámetros ${N=0,1,\dots }$, ${K=0,1,\dots ,N}$ y ${n=0,1,\dots ,N}$ y escribimos ${X\sim \operatorname {HG} (N,K,n)}$

$${\operatorname {P} [X=x]={\frac {{K \choose x}{N-K \choose n-x}}{N \choose n}},}$$

$${\operatorname {E} [X]={\frac {nK}{N}}}$$

$${\operatorname {Var} [X]={\frac {nK}{N}}{\bigg (}{\frac {N-K}{N}}{\bigg )}{\bigg (}{\frac {N-n}{N-1}}{\bigg )}}$$

Si una variable aleatoria ${X\sim \operatorname {HG} (N,K,1)}$ entonces ${X\sim \operatorname {Bernoulli} \left({\frac {K}{N}}\right)}$

La **distribución hipergeométrica**,  mide la probabilidad de obtener x (0 ≤ x ≤ d) elementos de una determinada clase formada por d elementos pertenecientes a una población de N elementos, tomando una muestra de n elementos de la población sin reemplazo.


### Distribuciones de variable continua

Se denomina variable continua a aquella que puede tomar cualquiera de los infinitos valores existentes dentro de un intervalo. En el caso de variable continua la distribución de probabilidad es la integral de la función de densidad, por lo que tenemos entonces que:

$$F(x)=P(X\leq x)=\int _{-\infty }^{x}f(t)\,dt$$


## El Teorema del Límite Central

Si ${X1, \dots,X_n}$ son i.i.d. y ${s2 = \operatorname{Var}(Xi) < \infty}$ entonces para cualquier ${z}$, donde ${Z\sim\operatorname{N}(0, 1)}$

$${\operatorname{P} ({\frac{\overline{X}-\mu}{\sqrt{{\sigma^2}_x}}} < z) \to_{n\to\infty}} \operatorname{P}(Z<z) ⟹ {\frac{\overline{X}-\mu}{\sqrt{{\sigma^2}_x}}} ≈ \operatorname{N}(0, 1)$$

Una **distribución binomial** de parámetros $n$ y $p$ es aproximadamente normal para grandes valores de $n$, y $p$ no demasiado cercano a 0 o a 1
La normal aproximada tiene parámetros $μ = np$, $σ^2 = np(1 − p)$

Una **distribución de** Poisson con parámetro $λ$ es aproximadamente normal para grandes valores de $λ$

La **distribución normal** aproximada tiene parámetros $μ = σ^2 = λ$.

$$X_1+X_2+\dots+X_n \sim N(\mu, \sigma^2)$$

$$X_1+X_2+\dots+X_n \sim N(n ⋅ \operatorname{E}(X), n ⋅ \operatorname{Var}(X))$$

Sea ${X1}$, ${X2}$, $\dots$ una secuencia de v.a. independientes e igualmente distribuidas tales que ${µ = \operatorname{E} (X_i)}$ existe. Sea
$$\overline{X}_n = \frac{(X_1+X_2+\dots+X_n)}{n} \sim N(\operatorname{E}(X), \frac{\operatorname{Var}(X)}{n}) $$

# Vectores Aleatorios

### Probabilidad Conjunta

$${\operatorname{P}_{XY}}(x, y) = \operatorname{P}(X=x \cap Y=y)$$

### Probabilidad Marginal

$${\operatorname{P}_{X}(x) = \sum_{y ∈ R_Y}\operatorname{P}_{XY}(X=x, \ \  Y=y)}$$

$${\operatorname{P}_{Y}(y) = \sum_{x ∈ R_X}\operatorname{P}_{XY}(X=x, \ \  Y=y)}$$

$${\operatorname{P}(Y|X=x) = \frac{\operatorname{P}_{XY}(x, y)}{\operatorname{P}_{X}(x)}}$$

### Bayes

$${\operatorname{P}_{Y|X=x}(y) = \frac{\operatorname{P}_{X|Y=y}(x) \operatorname{P}_{Y}(y)}{\operatorname{P}_{X}(x)}}$$

<!-- pagebreak -->

# Comandos R

```r
c(2, 4, 6) => 2, 4, 6
2:6 => 2, 3, 4, 5, 6
x[1] => el 1er elemento
x[4] => el 4to elemento
x[-4] => Todos menos el 4to
x[2:4] => del 2do-4to
x[-(2:4)] => Todos menos del 2do-4to
x[x(1, 5)] => 1er y 5to elemento
x[x == 10] => Todos los iguales a 10
x[x < 0] => Todos los menores a 0
x[x %in% c(1, 2, 5)] => ⋂  con el set 1, 2, 5

dbinom(x, size, prob)
pbinom(q, size, prob)
x, q: vector of quantiles.
size: number of trials (zero or more).
prob: probability of success on each trial.
dpois(x, lambda)
ppois(q, lambda)
x, q: vector of quantiles.
lambda: vector of (non-negative) means.
```