# Core concept
- The existence of Black Litterman Model is to solve the problem from [[Modern Portfolio Theory]]
	- MPT is a simple model, but too sensitive on expected return, by changing a little ER, the whole efficient frontier changes
	- also, under MPT, assets might be shorted, making the allocation negative. Although we can impose no-short constraint into the model, the allocation would still end up with 0, and this movement is unintuitive, making we hard to forecast the return. 
	- Also, if we know the 'expected return', we actually have no need to 'predict the return' anymore
# Steps to do BLM
## 1. Calculate Implied Equilibrium Returns using reverse optimization. 

- The Black litterman model starts at a capital weighted portfolio, which is base on the market capital. (asset price $\times$ liquidity in the market). We'll call it **market portfolio** afterwards.
	- It's base on the assumption that, the best allocation is the market itself.
- Then we can get the **[[Implied equilibrium returns]]** using the market portfolio and risk aversion parameter by **reverse optimization**. 
	- [[Mean Variance Optimization]] will be used in later steps and MVO requires expected returns. However, directly estimating expected returns are challenging, so we use implied equilibrium return to provide a starting point so we can avoid bias and extreme portfolio allocation. 
	- Expected excess return of an asset is the return an investor expects it to earn above the risk-free rate. 
	- $E[R_i] - R_f$ where $E[R_i]$ is the expected return of asset $i$ and $R_f$ is the risk-free rate, usually the US treasury yield. The difference represents the compensation for risk (the return for higher risk instead of investing in the risk free asset)
	- $\lambda$ is also incorporated, as a compensation of risk
$$ \Pi = \lambda \Sigma \omega_{mkt}$$ where 
- $\lambda$ = risk aversion rate (use the S&P market return!)
- $\Sigma$ = covariance matrix of asset returns (measuring risk and correlations)
- $\omega_{mkt}$ = market-cap weighted portfolio (how the market allocates capital)
- so the equation means the risk bearing


The risk-aversion rate $\lambda$ shows how much return an investor is willing to give up to reduce risk. In other words, an extra return an investor requires for taking on additional risk. Therefore, lambda is scales up the excess return, which means the higher $\lambda$, the investor expects more excess return for each unit of risk.
In our example, the implied equilibrium return is around 20%, which means investors need an extra 20% above the risk-free rate to be willing to accept the risk involved.

## 2. Incorporate Views — Specify investor's views on specific assets (where magic happens)
To incorporate investor's view, we have to build 4 view parameters, P, Q, omega ($\Omega$) and tau ($\tau$).
### P, The pick matrix

The pick matrix is "picking" the stocks we have idea on, where each row / vector corresponds to one view. And each column represents to an asset. We can have multiple views on a portfolio so the size of P is $K \times N$ where K is the number of views and N is the number of assets in the portfolio. In our example, we have 3 views among 7 assets, so the size of P is a $3 \times 7$ matrix.

How we write P shows how an asset is involved in that view and there are 2 types of view, the absolute view and the relative view. 

For an absolute view, we have an idea in one single stock, like "NVDA will have a return of 200% above the risk-free rate", then, the row would become $[0, 0, 0, 0, 1, 0, 0]$, where only column "NVDA" is "on".

```Python
# ['NVDA', 'TSM', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA']

P = np.array([
[1, 0, 0, 0, 0, 0, 0], # NVDA is involved
[0, 0, 0, 0, 0, 0, 1], # TSLA is involved
[0, 0, -1, 1, 0, 0, 0], # GOOG is outperforming MSFT
], dtype=float)
```

For a relative view, we have idea in a pair of stocks, like "GOOG will outperform MSFT by 30%", then, the view would become $[0, 0, -1, 1, 0, 0, 0]$ 

### Q, the view vector

After identifying which stock is involved in each view, we can implement the view into them. Q is basically the vector of our independent views. So Q will be a $\text{K} \times 1$ vector and each element in Q matches the row in $\text{P}$ 

```Python
Q = np.array([2, 0.2, 0.3], dtype=float)
```

Essentially, Q states how large we think the return or the relative outperformance is, differentiating from the market implied return.
### $\Omega {\space} \text{(Omega)}$ The uncertainty associated with investor's views
Omega is the uncertainty of the investor's views. Each vector in the $\Omega$ matrix is corresponding to each element in the $Q$ vector, so the size of $\Omega$ will be $\text{K} \times \text{K}$. Since we assume views are independent, the omega is a diagonal matrix. If a particular view is highly uncertain (ie. we don't have much confidence in that view), the entry will be large.
```Python
omega = np.diag([0.002, 0.002, 0.003])

# array([ [0.002, 0. , 0. ], 
#         [0. , 0.002, 0. ], 
#         [0. , 0. , 0.003] ])
```

Basically, $\Omega$ determines how strong the model tilts away from the prior market model in favor of our new views. If $\Omega$ is small, the view carries a lot of weight pushing the final return estimates towards our view.
### $\tau {\space} \text{(Tau)}$ The uncertainty to the market
$\tau$ is the scaler representing the uncertainty in the prior, or in other words, how much do we believe in our (investor's) view rather than the market. The higher $\tau$ give more weight to investor's view, vice versa.

However, in the Efficient Market Hypothesis and CAPM, where Black Litterman model is embodied, the market portfolio is considered to be priced fairly and so the prior is reliable. Implementing our views into the model, we would still trust the model a little bit less than how history had done. Therefore, it is typically set between 0.01 and 0.05. *(Dr. Wai Lee is quoted in "a step by step guide to the black-litterman model" - 9)*

```Python
tau = 0.025
```
## 3. Compute the New Expected Returns using the Posterior return formula

$$\mu_{BL} = [(\tau \Sigma)^{-1} + P^T \Omega^{-1}P]^{-1} \times [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1}Q]$$
Let's explain the equation bit by bit.
There are two parts in the equation, the inverse part ( $[(\tau \Sigma)^{-1} + P^T \Omega^{-1}P]^{-1}$ ) and the right hand side part ( $[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1}Q]$ ).

### The left part  ( $[(\tau \Sigma)^{-1} + P^T \Omega^{-1}P]^{-1}$ ) 
After knowing what $\tau$ is, we can understand that $\tau \Sigma$ represents the asset covariance matrix scaled by scaler $\tau$, and we inverse it ( $(\tau \Sigma)^{-1}$ ) to represent how strongly we weight the prior $\Pi$ base on its variance. It's usually called the precision of the prior.

 $\Omega ^{-1}$ represents the uncertainty of investor's views, and P is the pick matrix indicated which assets are involved in each view. 

Combining the uncertainty of investor's views and the investor's view itself, we get the statement of "How confident am I in these views, and which assets do they involve", being represented in an $N \times N$ matrix.

So in the left part, we have the posterior covariance by inverting the combination of uncertainty $$(\text{prior precision} + \text{view precision})^{-1}$$
### The right part ( $[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1}Q]$ )
The right part also be divided into 2 parts: (Prior precision $\times$ prior mean) and the (view precision $\times$ view mean)

We have the same $(\tau \Sigma)^{-1}$ representing the prior precision, but this time, we multiplied it with the prior expected return $\Pi$, which can be think of it as the prior expected return scaled by how confident we are in them.

The second part is similar to the view precision, but this time, P is replaced by Q. As introduced, Q is the view returns, which is "how much we think the picked assets are going to change". By multiplying $P^T \times \Omega^{-1}$, we get the views weight by their precision, where again, $\Omega^{-1}$ is how confident we are in those views and $P^T$ is the indicator of which assets are involved in each view.
The right part is an $\text{N} \times 1$ vector, representing the combined information from the prior and views in the asset space.

### Combining 2 parts together
Combining 2 parts together, we get the (confidence or uncertainty in our prior) with the (confidence or uncertainty in the views), which yields the posterior expected return of the assets.

### The bayesian approach
Black Litterman model is considered as a Bayesian approach because we start the portfolio with a "prior" distribution of returns, the market portfolio, and then incorporate investors' views in order to obtain the "posterior" distribution $\mu_{BL}$. Essentially, we are updating our ideas and believes about expected returns base on how the market does in a Bayesian manner.

## 4. Derive the New Optimal Portfolio Weights after incorporating investor's views.
$$w_{BL} = \frac{1}{\lambda}\Sigma^{-1}\mu_{BL}$$
This formula is to find the new optimal portfolio weight after getting the posterior return. It's actually the same but changed in order. We replace the prior equilibrium expected return $\Pi$ with the posterior expected return $\mu_{BL}$, and rearranged so the weight is isolated and becomes the subject of the equation. The answer will become a vector of posterior weights.

# Conclusion

The Black Litterman model improves the Modern Portfolio Theory, which is highly sensitive to changes in expected returns and integrates the investor's subjective view into it, which then balances the view and the market by different confidence level ($\Omega$ and $\tau$ ). 

The model adjusts equilibrium returns based on investor views, weighing them by their confidence and market information. 

By treating both as distribution and updating them in the approach of Bayesian manner, we obtain the posterior expected returns $\mu_{BL}$. 

# Projects
🔒 [[Tech Savvy Project Brief]]