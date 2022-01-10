# Implied-Stock-Probability-Mass-Function-from-Market-European-Option-Prices
The purpose of this project is to retrieve a probability mass function from market call and put european option prices.  I use tensorflow and tensorflow_probability to learn the implied PMF.

## SPX Market Prices

![alt text](https://github.com/PhilipFelizarta/Implied-Stock-Probability-Mass-Function-from-Market-European-Option-Prices/blob/main/figures/MarketCallPrices.png?raw=true)

![alt text](https://github.com/PhilipFelizarta/Implied-Stock-Probability-Mass-Function-from-Market-European-Option-Prices/blob/main/figures/MarketPutPrices.png?raw=true)

## Option Price and Stock Price Probability Density
<img src="https://latex.codecogs.com/png.image?\bg_white&space;u_{call}(K)&space;=&space;\int_{K}^{\infty}(x-K)&space;s(x)&space;dx&space;" title="u_{call} = \int_{K}^{\infty}(x-k) s(x) dx " />
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;u_{put}(K)&space;=&space;\int_{0}^{K}(K-x)&space;s(x)&space;dx&space;" title="\bg_white u_{put} = \int_{0}^{K}(K-x) s(x) dx " />
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;L_{model}&space;=&space;\frac{1}{2M}&space;\sum_{j}^{M}&space;\frac{1}{2}(ln(u_{model,j})&space;-&space;ln(u_{true,j}))^2&space;&plus;&space;\frac{1}{2}(u_{model,&space;j}&space;-&space;u_{true,&space;j})^2" title="\bg_white L_{model} = \frac{1}{2M} \sum_{j}^{M} \frac{1}{2}(ln(u_{model,j}) - ln(u_{true,j}))^2 + \frac{1}{2}(u_{model, j} - u_{true, j})^2" />

## Parameterizing S(.) and Discretizing the Optimization Problem
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;s&space;=&space;softmax(z)" title="\bg_white s = softmax(z)" />
Where z will be a learned vector, s_i will represent <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;\inline&space;P(S_T&space;=&space;x_i)" title="\bg_white \inline P(S_T = x_i)" /> x_i will is an element of the mesh spanning the moneyness of strike prices. Thus, the option prices can be estimated like such (N=1000 in the code)...
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;u_{call}(K)&space;=&space;\sum_{i}^{N}&space;s_i(x_i&space;-&space;K)^&plus;&space;" title="\bg_white u_{call}(K) = \sum_{i}^{N} s_i(x_i - K)^+ " />
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;u_{put}(K)&space;=&space;\sum_{i}^{N}&space;s_i(K&space;-&space;x_i)^&plus;&space;" title="\bg_white u_{put}(K) = \sum_{i}^{N} s_i(K - x_i)^+ " />

## Continuity Regularization
To achieve "smooth" functions I utilized a neat regularization trick where I sum the square distances between neighboring parameters.

<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;\min_{z}&space;\gamma&space;\sum_{i}^{N-1}\frac{(z_i&space;-&space;z_{i&plus;1})^2}{|x_i&space;-&space;x_{i&plus;1}|}" title="\bg_white \min_{z} \gamma \sum_{i}^{N-1}\frac{(z_i - z_{i+1})^2}{|x_i - x_{i+1}|}" />
I chose to weigh the square difference between points by the inverse of the absolute difference between corrosponding mesh points because the mesh is not linear (its geometric so that I may concentrate on PMF on values near at-the-money).

## Optimization Problem
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_white&space;\min_{z}&space;L_{call}(z)&space;&plus;&space;L_{put}(z)&space;&plus;&space;\gamma&space;\sum_{i}^{N-1}\frac{(z_i&space;-&space;z_{i&plus;1})^2}{|x_i&space;-&space;x_{i&plus;1}|}" title="\bg_white \min_{z} L_{call}(z) + L_{put}(z) + \gamma \sum_{i}^{N-1}\frac{(z_i - z_{i+1})^2}{|x_i - x_{i+1}|}" />

gamma is set to 1e-4 for SPX European Options. I use tensorflow_probability to optimize this equation with L-BFGS.

## Results
### PMF from SPX European Call and Put Options
![alt text](https://github.com/PhilipFelizarta/Implied-Stock-Probability-Mass-Function-from-Market-European-Option-Prices/blob/main/figures/PMF.png?raw=true)

### Predicted Call Option Prices (Blue) vs Ground Truth (Orange)
![alt text](https://github.com/PhilipFelizarta/Implied-Stock-Probability-Mass-Function-from-Market-European-Option-Prices/blob/main/figures/ModelCall.png?raw=true)

### Model Call Option Residuals (Absolute Difference)
![alt text](https://github.com/PhilipFelizarta/Implied-Stock-Probability-Mass-Function-from-Market-European-Option-Prices/blob/main/figures/CallResiduals.png?raw=true)

### Predicted Put Option Prices (Blue) vs Ground Truth (Orange)
![alt text](https://github.com/PhilipFelizarta/Implied-Stock-Probability-Mass-Function-from-Market-European-Option-Prices/blob/main/figures/ModelPut.png?raw=true)

### Model Put Option Residuals (Absolute Difference)
![alt text](https://github.com/PhilipFelizarta/Implied-Stock-Probability-Mass-Function-from-Market-European-Option-Prices/blob/main/figures/PutResiduals.png?raw=true)
