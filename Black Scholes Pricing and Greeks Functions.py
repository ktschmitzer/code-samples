#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# function to find d1 in the Black Scholes Equation
def findd1(S, K, sigma, r, delta, T):
    return (1/(sigma*np.sqrt(T)))*(np.log(S/K) + (r - delta + 0.5*sigma**2)*T)
    
# function to find d2 in the Black Scholes Equation
def findd2(S, K, sigma, r, delta, T):
    return (1/(sigma*np.sqrt(T)))*(np.log(S/K) + (r - delta + 0.5*sigma**2)*T) - (sigma*np.sqrt(T))

# function to find the price of a call using Black Scholes pricing
def callprice(S, K, sigma, r, delta, T, d1, d2):
    return (S*np.exp(-delta * T)*norm.cdf(d1)) - (K*np.exp(-r * T)*norm.cdf(d2))
    
# function to find the price of a put using Black Scholes pricing
def putprice(S, K, sigma, r, delta, T, d1, d2):
    return (K*np.exp(-r * T)*norm.cdf(-d2)) - (S*np.exp(-delta * T)*norm.cdf(-d1))
    
# function to find the Delta greek
def delta(S, T, sigma, delta, d1):
    return np.exp(-delta*T)*norm.cdf(d1)

