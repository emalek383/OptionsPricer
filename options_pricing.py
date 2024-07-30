import numpy as np
from functools import partial
from scipy.stats import norm

DEFAULT_NUM_STEPS = 1_000

def calculate_vanilla_BS(S, K, T, vol, r, call = True):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + vol**2/2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    if call:
        price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    return price

def calculate_digital_BS(S, K, T, vol, r, call = True):
    d1 = (np.log(S/K) + (r + vol**2/2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    if call:
        price = np.exp(-r*T)*norm.cdf(d2, 0, 1)
    else:
        price = np.exp(-r*T)*norm.cdf(-d2, 0, 1)
    return price

def price_barrier_option_bs(S, K, T, r, vol, barrier, barrier_type, call=True):
    """
    Vectorized pricing of European barrier options using the Black-Scholes model.
    
    Parameters:
    S : array_like
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    vol : array_like
        Volatility
    barrier : float
        Barrier price
    barrier_type : str
        One of 'up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in'
    call : bool
        True for a call option, False for a put option
    
    Returns:
    array_like : Option prices
    """
    
    valid_types = ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']
    if barrier_type not in valid_types:
        raise ValueError(f"option_type must be one of {valid_types}")
    
    S = np.asarray(S)
    vol = np.asarray(vol)
    scalar_input = S.ndim == 0 and vol.ndim == 0
    
    vanilla_price = calculate_vanilla_BS(S, K, T, vol, r, call)
    
    mu = (r - 0.5*vol**2) / vol**2
    lambda_param = np.sqrt(mu**2 + 2*r/vol**2)
    sigma_sqrt_T = vol * np.sqrt(T)
    y = np.log(barrier**2 / (S*K)) / sigma_sqrt_T + lambda_param * sigma_sqrt_T
    x1 = np.log(S/barrier) / sigma_sqrt_T + lambda_param * sigma_sqrt_T
    y1 = np.log(barrier/S) / sigma_sqrt_T + lambda_param * sigma_sqrt_T
    
    if call:
        if barrier_type.startswith('down'):
            barrier_above = barrier >= S
            barrier_below_K = barrier <= K
            
            in_price = np.where(barrier_below_K,
                                S * (barrier / S)**(2*lambda_param) * norm.cdf(y) 
                                - K * np.exp(-r * T) * (barrier / S)**(2 * lambda_param - 2) * norm.cdf(y - sigma_sqrt_T),
                                vanilla_price - (S * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sigma_sqrt_T) 
                                                 - S * (barrier / S)**(2*lambda_param) * norm.cdf(y1) 
                                                 + K * np.exp(-r * T) * (barrier / S)**(2*lambda_param - 2) * norm.cdf(y1 - sigma_sqrt_T)))
            
            if barrier_type.endswith('in'):
                result = np.where(barrier_above, vanilla_price, in_price)
            else:
                result = np.where(barrier_above, 0, vanilla_price - in_price)
        else:  # up options
            barrier_below = barrier <= S
            barrier_below_K = barrier <= K
            
            out_price = np.where(barrier_below_K,
                                 0,
                                 vanilla_price - (S * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sigma_sqrt_T)
                                                  - S * (barrier / S)**(2*lambda_param) * (norm.cdf(-y) - norm.cdf(-y1))
                                                  + K * np.exp(-r * T) * (barrier / S)**(2*lambda_param - 2) 
                                                  * (norm.cdf(-y + sigma_sqrt_T) - norm.cdf(-y1 + sigma_sqrt_T))))
            
            if barrier_type.endswith('in'):
                result = np.where(barrier_below, vanilla_price, vanilla_price - out_price)
            else:
                result = np.where(barrier_below, 0, out_price)
    else:  # put options
        if barrier_type.startswith('up'):
            barrier_below = barrier <= S
            barrier_above_K = barrier >= K
            
            in_price = np.where(barrier_above_K,
                                -S * (barrier / S)**(2 * lambda_param) * norm.cdf(-y) 
                                + K * np.exp(-r*T) * (barrier / S)**(2 * lambda_param - 2) * norm.cdf(-y + sigma_sqrt_T),
                                vanilla_price - (-S * norm.cdf(-x1) + K * np.exp(-r * T) * norm.cdf(-x1 + sigma_sqrt_T) 
                                                 + S * (barrier / S)**(2 * lambda_param) * norm.cdf(-y1) 
                                                 - K * np.exp(-r * T) * (barrier / S)**(2 * lambda_param - 2) * norm.cdf(-y1 + sigma_sqrt_T)))
            
            if barrier_type.endswith('in'):
                result = np.where(barrier_below, vanilla_price, in_price)
            else:
                result = np.where(barrier_below, 0, vanilla_price - in_price)
        else:  # down options
            barrier_above = barrier >= S
            barrier_above_K = barrier > K
            
            out_price = np.where(barrier_above_K,
                                 0,
                                 vanilla_price - (-S * norm.cdf(-x1) + K * np.exp(-r * T) * norm.cdf(-x1 + sigma_sqrt_T) 
                                                  + S * (barrier / S)**(2 * lambda_param) * (norm.cdf(y) - norm.cdf(y1))
                                                  - K * np.exp(-r * T) * (barrier / S)**(2 * lambda_param - 2) 
                                                  * (norm.cdf(y - sigma_sqrt_T) - norm.cdf(y1 - sigma_sqrt_T))))
            
            if barrier_type.endswith('in'):
                result = np.where(barrier_above, vanilla_price, vanilla_price - out_price)
            else:
                result = np.where(barrier_above, 0, out_price)
            
    return float(result) if scalar_input else result

def calculate_barrier_option_bs(S, K, T, r, vol, barrier, barrier_type, call=True):
    S = np.atleast_1d(S)
    result = price_barrier_option_bs(S, K, T, r, vol, barrier, barrier_type, call = call)
    return np.squeeze(result) if S.size == 1 else result

class log_normal_model():
    def __init__(self, stock_value, vol):
        self.stock_value = stock_value
        self.vol = vol
        
    def computePrice(self, i, j, dt):
        return self.stock_value * np.exp(self.vol * np.sqrt(dt) * (2 * j - i))

def base_binomial_price(S, K, T, vol, r, payoff, num_steps, option_specific_logic, american=False, asset_model=log_normal_model):
    dt = T / num_steps
    disc_factor = np.exp(-r*dt)
    
    asset_model_instance = asset_model(S, vol)
    S_maturity = asset_model_instance.computePrice(num_steps + 1, np.arange(0, num_steps + 1, 1), dt)
    
    value = payoff(S_maturity, K)
    
    for layer in np.arange(num_steps - 1, -1, -1):
        price_at_node = asset_model_instance.computePrice(layer, np.arange(0, layer + 1, 1), dt)
        price_up = S_maturity[1 : layer + 2]
        price_down = S_maturity[0 : layer + 1]
        prob_up = ((price_at_node / disc_factor) - price_down) / (price_up - price_down)
        prob_down = 1 - prob_up
        value = disc_factor * (prob_up * value[1 : layer + 2] + prob_down * value[0 : layer + 1])
        
        value = option_specific_logic(value, price_at_node)
        
        if american:
            exercise_value = payoff(price_at_node, K)
            value = np.maximum(value, exercise_value)
        
        S_maturity = price_at_node
    
    return value[0]

def barrier_logic(value, price_at_node, barrier, barrier_type):
    if barrier_type == "up-and-out":
        return np.where(price_at_node >= barrier, 0, value)
    elif barrier_type == "down-and-out":
        return np.where(price_at_node <= barrier, 0, value)
    else:
        raise ValueError("Invalid barrier type for binomial model")

def vanilla_payoff(S, K, call):
    return np.maximum(S - K, 0) if call else np.maximum(K - S, 0)

def digital_payoff(S, K, call):
    return (S > K).astype(int) if call else (S < K).astype(int)

def calculate_price(S, K, T, vol, r, call=True, american=False, option_type='vanilla', method='bs', **kwargs):
    """
    Calculate the price of an option based on the given parameters.

    Parameters:
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    vol : float
        Volatility (annualized)
    r : float
        Risk-free interest rate (annualized)
    call : bool
        True for a call option, False for a put option
    american : bool
        True for American-style options, False for European-style
    option_type : str
        'vanilla', 'digital', or 'barrier'
    method : str
        'bs' for Black-Scholes, 'tree' for binomial tree
    **kwargs : 
        Additional parameters, including 'barrier' and 'barrier_type' for barrier options

    Returns:
    float : Option price
    """

    if option_type not in ['vanilla', 'digital', 'barrier']:
        raise ValueError("Invalid option type. Must be 'vanilla', 'digital', or 'barrier'.")
    if method not in ['bs', 'tree']:
        raise ValueError("Invalid method. Must be 'bs' or 'tree'.")
    if american and method == 'bs':
        raise ValueError("Black-Scholes method cannot be used for American options.")

    if option_type == 'barrier':
        if 'barrier' not in kwargs or 'barrier_type' not in kwargs:
            raise ValueError("Barrier and barrier type must be specified for barrier options.")
        barrier = kwargs['barrier']
        barrier_type = kwargs['barrier_type'].lower()
        valid_barrier_types = ['up-and-in', 'down-and-in', 'up-and-out', 'down-and-out']
        if barrier_type not in valid_barrier_types:
            raise ValueError(f"Invalid barrier type. Must be one of {valid_barrier_types}.")

    if method == 'bs':
        if option_type == 'vanilla':
            return calculate_vanilla_BS(S, K, T, vol, r, call)
        elif option_type == 'digital':
            return calculate_digital_BS(S, K, T, vol, r, call)
        else:  # barrier
            return price_barrier_option_bs(S, K, T, r, vol, barrier, barrier_type, call)
    else:  # tree method
        payoff = vanilla_payoff if option_type != 'digital' else digital_payoff
        logic = lambda value, price_at_node: value  # for vanilla and digital
        if option_type == 'barrier':
            if barrier_type.endswith('-out'):
                logic = lambda value, price_at_node: barrier_logic(value, price_at_node, barrier, barrier_type)
            elif barrier_type.endswith('-in'):
                if american:
                    raise ValueError("American 'in' options are not supported due to in-out parity limitations")
                vanilla_price = calculate_vanilla_BS(S, K, T, vol, r, call)
                out_price = calculate_price(S, K, T, vol, r, call, False, 'barrier', 'tree', 
                                            barrier=barrier, barrier_type=barrier_type.replace('-in', '-out'))
                return vanilla_price - out_price
            
        return base_binomial_price(S, K, T, vol, r, partial(payoff, call=call), DEFAULT_NUM_STEPS, logic, american)