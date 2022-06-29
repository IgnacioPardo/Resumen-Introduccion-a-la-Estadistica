# Resumen Introducción a la Estadistica


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Resumen Introducción a la Estadistica](#resumen-introducción-a-la-estadistica)
  - [Esperanza y Varianza](#esperanza-y-varianza)
    - [Esperanza](#esperanza)
    - [Varianza](#varianza)
    - [Esperanza y Varianza Condicional](#esperanza-y-varianza-condicional)
  - [Distribuciones](#distribuciones)
    - [Distribución normal](#distribución-normal)
    - [Distribución binomial](#distribución-binomial)
    - [Distribución de Bernoulli](#distribución-de-bernoulli)
    - [Distribución Uniforme continua](#distribución-uniforme-continua)
    - [Distribución Uniforme Discreta](#distribución-uniforme-discreta)
    - [Distribución de Poisson](#distribución-de-poisson)
    - [Distribución geométrica](#distribución-geométrica)
    - [Distribución hipergéometrica](#distribución-hipergéometrica)
    - [Distribuciones de variable continua](#distribuciones-de-variable-continua)
  - [El Teorema del Límite Central](#el-teorema-del-límite-central)

<!-- /code_chunk_output -->

## Esperanza y Varianza

### Esperanza
$${\displaystyle \operatorname {E} [X]=\sum _{i=1}^{n}x_{i}\operatorname {P} [X=x_{i}]}$$

Si $X$ y $Y$ son variables aleatorias con esperanza finita y ${\displaystyle a,b,c\in \mathbb {R} }$ son constantes entonces

- ${\displaystyle \operatorname {E} [c]=c}$
- ${\displaystyle \operatorname {E} [cX]=c\operatorname {E} [X]}$
- Si ${\displaystyle X\geq 0}$ entonces ${\displaystyle \operatorname {E} [X]\geq 0}$
- Si ${\displaystyle X\leq Y}$ entonces ${\displaystyle \operatorname {E} [X]\leq \operatorname {E} [Y]}$
- Si $X$ está delimitada por dos números reales, $a$ y $b$, esto es ${\displaystyle a<X<b}$ entonces también lo está su media, es decir, ${\displaystyle a<\operatorname {E} [X]<b}$
- Si ${\displaystyle Y=a+bX}$, entonces ${\displaystyle \operatorname {E} [Y]=\operatorname {E} [a+bX]=a+b\operatorname {E} [X]}$

$${\displaystyle {\begin{aligned}\operatorname {E} [X+Y]&=\operatorname {E} [X]+\operatorname {E} [Y]\\\operatorname {E} [cX]&=c\operatorname {E} [X]\end{aligned}}}$$

- Si $X$ y $Y$ son variables aleatorias independientes entonces

$${\displaystyle \operatorname {E} [XY]=\operatorname {E} [X]\operatorname {E} [Y]}$$

### Varianza
$$\operatorname {Var}[X]=\operatorname {E} [X^{2}]-\operatorname {E} [X]^{2}$$

Sean $X$ y $Y$ dos variables aleatorias con varianza finita y ${\displaystyle a\in \mathbb {R} }$

- ${\displaystyle \operatorname {Var} (X)\geq 0}$
- ${\displaystyle \operatorname {Var} (a)=0}$
- ${\displaystyle \operatorname {Var} (aX)=a^{2}\operatorname {Var} (X)}$
- ${\displaystyle \operatorname {Var} (X+Y)=\operatorname {Var} (X)+\operatorname {Var} (Y)+2\operatorname {Cov} (X,Y)}$, donde ${\displaystyle \operatorname {Cov} (X,Y)}$ denota la covarianza de $X$ e $Y$
- ${\displaystyle \operatorname {Var} (X+Y)=\operatorname {Var} (X)+\operatorname {Var} (Y)}$ si $X$ y $Y$ son variables aleatorias independientes.
- ${\displaystyle \operatorname {Var} (Y)=\operatorname {E} (\operatorname {Var} (Y|X))+\operatorname {Var} (\operatorname {E} (Y|X))}$ cálculo de la Varianza por Pitágoras, dónde ${\displaystyle Y|X}$ es la variable aleatoria condicional $Y$ dado $X$.

### Esperanza y Varianza Condicional

$${\displaystyle \operatorname {E} (X|Y=y)=\sum _{x\in {\mathcal {X}}}x\ \operatorname {P} (X=x|Y=y)=\sum _{x\in {\mathcal {X}}}x{\frac {\operatorname {P} (X=x,Y=y)}{\operatorname {P} (Y=y)}}}$$

$$\operatorname{Var}[Y|X=x]=\operatorname{E}[Y^2|X=x]−{\operatorname{E}[Y|X=x]}2=\sum _{y}y^2 \operatorname{p}(y|x)−\{\operatorname{E}[Y|X=x]\}^2$$

### Desvio Estandard

$${\displaystyle \operatorname{SD}(X) =  \sigma ={\sqrt {{\text{Var}}(X)}}\,\!} \implies{\displaystyle \sigma ^{2}={\text{Var}}(X)\,\!}$$

### Ley de Esperanza Total

$$\operatorname {E}(C) = \operatorname {E}(\operatorname {E}(C ∣ N))$$

$$\operatorname {E}(N ⋅ Y ) = \operatorname {E}(\operatorname {E}(N ⋅ Y ∣ N)) = \operatorname {E}(g(N))$$

$$\operatorname {g}(n) = \operatorname {E}(N ⋅ Y ∣ N = n)$$

### Covarianza

$${\displaystyle \operatorname {Cov} (X,Y)=\operatorname {E} \left[XY\right]-\operatorname {E} \left[X\right]\operatorname {E} \left[Y\right]}$$

### Correlación

$$ρ_{xy} = {\frac{\operatorname{cov}_{xy}}{\sigma_x\sigma_y}} = {\frac{\operatorname{cov}_{xy}}{\operatorname{SD}(x)\operatorname{SD}(y)}}$$
## Distribuciones


$$F_{X}(x)= \mathrm {Prob} (X\leq x)$$

$$P(X\leq b)=P(X\leq a)+P(a<X\leq b)$$

$$P(a<X\leq b)=P(X\leq b)-P(X\leq a)$$

$${\displaystyle F(x)=P(X\leq x)=\sum _{k=-\infty }^{x}f(k)}$$

$$P(a<X\leq b)=F(b)-F(a)$$

### Distribución normal
Si ${\displaystyle X\sim N(\mu ,\sigma ^{2})}$ y ${\displaystyle a,b\in \mathbb {R} }$, entonces ${\displaystyle aX+b\sim N(a\mu +b,a^{2}\sigma ^{2})}$

Si ${\displaystyle X\,\sim N(\mu ,\sigma ^{2})\,}$, entonces ${\displaystyle Z={\frac {X-\mu }{\sigma }}\!}$ es una variable aleatoria normal estándar: $Z$ ~ $N(0,1)$.


### Distribución binomial

Si una variable aleatoria discreta $X$ tiene una distribución binomial con parámetros $n\in\mathbb{N}$ y $p$ con ${\displaystyle 0<p<1}$ entonces escribiremos ${\displaystyle X\sim \operatorname {Bin} (n,p)}$

La **distribución binomial**, describe el número de aciertos en una serie de n experimentos independientes con posibles resultados binarios, es decir, de «sí» o «no», todos ellos con probabilidad de acierto p y probabilidad de fallo q = 1 − p.


$${\displaystyle \operatorname {P} [X=x]={n \choose x}p^{x}(1-p)^{n-x}}$$

$${\displaystyle \!{n \choose x}={\frac {n!}{x!(n-x)!}}\,\!}$$

$${\displaystyle F_{X}(x)=\operatorname {P} [X\leq x]=\sum _{k=0}^{x}{n \choose k}p^{k}(1-p)^{n-k}}$$

$${\displaystyle \operatorname {E} [X]=np} \ , \ \ {\displaystyle \operatorname {Var} [X]=np(1-p)}$$

### Distribución de Bernoulli

Si $X$ es una variable aleatoria discreta que mide el "número de éxitos" y se realiza un único experimento con dos posibles resultados denominados éxito y fracaso, se dice que la variable aleatoria ${\displaystyle X\,}$ se distribuye como una Bernoulli de parámetro ${\displaystyle p\,}$ con ${\displaystyle 0<p<1} $ y escribimos ${\displaystyle X\sim \operatorname {Bernoulli} (p)}$

$${\displaystyle \operatorname {P} [X=x]=p^{x}(1-p)^{1-x}\qquad x=0,1}$$

$${\displaystyle F(x)={\begin{cases}0&x<0\\1-p&0\leq x<1\\1&x\geq 1\end{cases}}}$$

$${\displaystyle \operatorname {E} \left[X\right]=\operatorname {E} \left[X^n\right]=p}$$

$${\displaystyle {\begin{aligned}\operatorname {Var} \left[X\right]&=\operatorname {E} [X^{2}]-\operatorname {E} [X]^{2}\\&=p-p^{2}\\&=p\left(1-p\right)\end{aligned}}}$$

Si ${\displaystyle X_{1},X_{2},\dots ,X_{n}}$ son $n$  variables aleatorias independientes e identicamente distribuidas con ${\displaystyle X_{i}\sim \operatorname {Bernoulli} (p)}$ entonces la variable aleatoria ${\displaystyle X_{1}+X_{2}+\dots +X_{n}}$ sigue una distribución binomial con parámetros $n$ y $p$, es decir

$${\displaystyle \sum _{i=1}^{n}X_{i}\sim \operatorname {Bin} (n,p)}$$

### Distribución Uniforme continua
Si $X$ es una variable aleatoria continua con distribución uniforme continua entonces escribiremos ${\displaystyle X\sim \operatorname {U} (a,b)}$ o ${\displaystyle X\sim \operatorname {Unif} (a,b)}$

$${\displaystyle f_{X}(x)={\frac {1}{b-a}}}$$

$${\displaystyle {\begin{aligned}F_{X}(x)={\frac {x-a}{b-a}}\end{aligned}}}$$

$${\displaystyle \operatorname {E} [X]={\frac {a+b}{2}}}$$

$${\displaystyle \operatorname {Var} (X)={\frac {(b-a)^{2}}{12}}}$$


### Distribución Uniforme Discreta

Si $X$ es una variable aleatoria discreta cuyo soporte es el conjunto ${\displaystyle \{x_{1},x_{2},\dots ,x_{n}\}}$ y tiene una distribución uniforme discreta entonces escribiremos ${\displaystyle X\sim \operatorname {Uniforme} (x_{1},x_{2},\dots ,x_{n})}$

La **distribución uniforme discreta**, recoge un conjunto finito de valores que son resultan ser todos igualmente probables. Esta distribución describe, por ejemplo, el comportamiento aleatorio de una moneda, un dado, o una ruleta de casino equilibrados (sin sesgo).

$${\displaystyle \operatorname {P} [X=x]={\frac {1}{n}}}$$

$${\displaystyle \operatorname {E} [X]={\frac {1}{n}}\sum _{i=1}^{n}x_{i}\,\!}$$


$${\displaystyle \operatorname {Var} (X)={\frac {1}{n}}\sum _{i=1}^{n}(x_{i}-\operatorname {E} [X])^{2}}$$

### Distribución de Poisson

Sea ${\displaystyle \lambda >0}$ y $X$ una variable aleatoria discreta, si la variable aleatoria $X$ tiene una distribución de Poisson con parámetro $\lambda$  entonces escribiremos ${\displaystyle X\sim \operatorname {Poisson} (\lambda )}$ o ${\displaystyle X\sim \operatorname {Poi} (\lambda )}$

$${\displaystyle \operatorname {P} [X=k]={\frac {e^{-\lambda }\lambda ^{k}}{k!}}}$$

$${\displaystyle \operatorname {E} [X]=\operatorname {Var} (X)=\lambda }$$

Como consecuencia del teorema central del límite, para valores grandes de $\lambda$ , una variable aleatoria de Poisson $X$ puede aproximarse por otra normal dado que el cociente 
$${\displaystyle Y={\frac {X-\lambda }{\sqrt {\lambda }}}}$$ converge a una distribución normal de media 0 y varianza 1.


### Distribución geométrica

Si una variable aleatoria discreta {\displaystyle X}X sigue una distribución geométrica con parámetro ${\displaystyle 0<p<1}$  entonces escribiremos ${\displaystyle X\sim \operatorname {Geometrica} (p)}$ o simplemente ${\displaystyle X\sim \operatorname {Geo} (p)}$

La **distribución geométrica**, describe el número de intentos necesarios hasta conseguir el primer acierto.

$${\displaystyle \operatorname {P} [X=x]=p(1-p)^{x}}$$

$${\displaystyle {\begin{aligned}\operatorname {P} [X\leq x]=1-(1-p)^{x+1}\end{aligned}}}$$

$${\displaystyle \operatorname {E} [X]={\frac {1}{p}}}$$

$${\displaystyle \operatorname {Var} (X)={\frac {1-p}{p^{2}}}}$$

### Distribución hipergéometrica

Una variable aleatoria discreta $X$ tiene una distribución hipergeométrica con parámetros ${\displaystyle N=0,1,\dots }$, ${\displaystyle K=0,1,\dots ,N}$ y ${\displaystyle n=0,1,\dots ,N}$ y escribimos ${\displaystyle X\sim \operatorname {HG} (N,K,n)}$

$${\displaystyle \operatorname {P} [X=x]={\frac {{K \choose x}{N-K \choose n-x}}{N \choose n}},}$$

$${\displaystyle \operatorname {E} [X]={\frac {nK}{N}}}$$

$${\displaystyle \operatorname {Var} [X]={\frac {nK}{N}}{\bigg (}{\frac {N-K}{N}}{\bigg )}{\bigg (}{\frac {N-n}{N-1}}{\bigg )}}$$

Si una variable aleatoria ${\displaystyle X\sim \operatorname {HG} (N,K,1)}$ entonces ${\displaystyle X\sim \operatorname {Bernoulli} \left({\frac {K}{N}}\right)}$

La **distribución hipergeométrica**,  mide la probabilidad de obtener x (0 ≤ x ≤ d) elementos de una determinada clase formada por d elementos pertenecientes a una población de N elementos, tomando una muestra de n elementos de la población sin reemplazo.


### Distribuciones de variable continua

Se denomina variable continua a aquella que puede tomar cualquiera de los infinitos valores existentes dentro de un intervalo. En el caso de variable continua la distribución de probabilidad es la integral de la función de densidad, por lo que tenemos entonces que:

$$F(x)=P(X\leq x)=\int _{-\infty }^{x}f(t)\,dt$$


## El Teorema del Límite Central

Una **distribución binomial** de parámetros $n$ y $p$ es aproximadamente normal para grandes valores de $n$, y $p$ no demasiado cercano a 0 o a 1.
La normal aproximada tiene parámetros $μ = np$, $σ2 = np(1 − p)$.

Una **distribución de** Poisson con parámetro $λ$ es aproximadamente normal para grandes valores de $λ$.

La **distribución normal** aproximada tiene parámetros $μ = σ2 = λ$.

$X_1+X_2+\dots+X_n \sim N(\mu, \sigma^2)\,$

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