"""Module creating classes for option pricing using Black-Scholes and Binomial Tree models.

Classes
-------
OptionFactory: Factory class for creating different types of options.
log_normal_model: Class modelling log-normal asset price evolution.
Option: Abstract base class for options.
VanillaOption: Class for vanilla options.
DigitalOption: Class for digital options.
BarrierOption: Class for barrier options.
PricingModel: Abstract base class for pricing models.
BlackScholesModel: Black-Scholes pricing model.
BinomialTreeModel: Binomial Tree pricing model.
"""


import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod

DEFAULT_NUM_STEPS = 1_000

class OptionFactory:
    """
    Factory class for creating different types of options.
    
    Methods
    -------
    create_option(option_type, *args, **kwargs):
        Create and return an instance of the specified option type.
        
    """
    
    @staticmethod
    def create_option(option_type, *args, **kwargs):
        """
        Create and return an instance of the specified option type.

        Parameters
        ----------
        option_type : str
            Type of option to create ('vanilla', 'digital', or 'barrier').
        *args, **kwargs
            Arguments to pass to the option constructor.
            
        Raises
        ------
        ValueError
            If an invalid option type is specified.

        Returns
        -------
        Option
            An instance of the specified option type.
        
        """
        
        if option_type == 'vanilla':
            return VanillaOption(*args, **kwargs)
        elif option_type == 'digital':
            return DigitalOption(*args, **kwargs)
        elif option_type == 'barrier':
            return BarrierOption(*args, **kwargs)
        else:
            raise ValueError("Invalid option type")

class log_normal_model():
    """
    Class for modelling log-normal asset price evolution.
    
    Attributes
    ----------
    stock_value : float
        Initial stock value.
    vol : float
        Volatility of the stock.

    Methods
    -------
    computePrice(i, j, dt):
        Compute the stock price at a given node in the binomial tree.
        
    """
    
    def __init__(self, stock_value, vol):
        """
        Initialise the log_normal_model.

        Parameters
        ----------
        stock_value : float
            Initial stock value.
        vol : float
            Volatility of the stock.
            
        """
        
        self.stock_value = stock_value
        self.vol = vol
        
    def computePrice(self, i, j, dt):
        """
        Compute the stock price at a given node in the binomial tree.

        Parameters
        ----------
        i : int
            Time step in the tree.
        j : int
            Node index at the given time step.
        dt : float
            Time step size.

        Returns
        -------
        float
            Stock price at the specified node.
        
        """
        
        return self.stock_value * np.exp(self.vol * np.sqrt(dt) * (2 * j - i))

class Option(ABC):
    """
    Abstract base class for options.
    
    Attributes
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    vol : float
        Volatility of the underlying asset.
    r : float
        Risk-free interest rate.
    call : bool
        True for a Call option, False for a Put option.
    american : bool
        True for an American option, False for a European option.

    Methods
    -------
    price(model):
        Price the option using the specified model.
    vanilla_payoff(spot_price, strike, call = True):
        Static method to compute the payoff of a vanilla option.
    payoff(spot_price):
        Abstract method to compute the payoff of the option.
    option_specific_logic(option_values, spot_prices):
        Apply option-specific logic in the Binomial Tree pricing process.

    """

    def __init__(self, S, K, T, vol, r, call = True, american = False):
        """
        Initialise the Option.

        Parameters
        ----------
        S : float
            Spot price of the underlying asset.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        vol : float
            Volatility of the underlying asset.
        r : float
            Risk-free interest rate.
        call : bool, optional
            True for a Call option, False for a Put option. Default is True.
        american : bool, optional
            True for an American option, False for a European option. Default is False.
        
        """
        
        self.S = S
        self.K = K
        self.T = T
        self.vol = vol
        self.r = r
        self.call = call
        self.american = american
        
    def price(self, model):
        """
        Price the option using the specified model.

        Parameters
        ----------
        model : PricingModel
            The pricing model to use.

        Returns
        -------
        float
            The price of the option.
        
        """
        
        return model.price(self)
    
    @staticmethod
    def vanilla_payoff(spot_price, strike, call = True):
        """
        Compute the payoff of a vanilla option, allowing vectorised use.

        Parameters
        ----------
        spot_price : float or np.array
            Spot price(s) of the underlying asset.
        strike : float
            Strike price of the option.
        call : bool, optional
            True for a Call option, False for a Put option. Default is True.

        Returns
        -------
        float or np.array
            The payoff(s) of the vanilla option.
        
        """
        
        return np.maximum(spot_price - strike, 0) if call else np.maximum(strike - spot_price, 0)
    
    @abstractmethod 
    def payoff(self, spot_price):
        """
        Abstract method to compute the payoff of the option.

        Parameters
        ----------
        spot_price : float or np.array
            Spot price(s) of the underlying asset.

        Returns
        -------
        float or np.array
            The payoff(s) of the option.
        
        """
        
        pass
    
    def option_specific_logic(self, option_values, spot_prices):
        """
        Apply option-specific logic in the Binomial Tree pricing process.

        Parameters
        ----------
        option_values : np.array
            Option values at a given step in the pricing process.
        spot_prices : np.array
            Spot prices at a given step in the pricing process.

        Returns
        -------
        np.array
            Updated option values after applying option-specific logic.
        
        """
        
        return option_values


class VanillaOption(Option):
    """
    Class representing a vanilla option.
    
    Methods
    -------
    payoff(spot_price):
        Compute the payoff of the vanilla option.

    """

    def payoff(self, spot_price):
        """
        Compute the payoff of the vanilla option.

        Parameters
        ----------
        spot_price : float or np.array
            The spot price(s) of the underlying asset.

        Returns
        -------
        float or np.array
            The payoff(s) of the vanilla option.
        
        """
        
        return self.vanilla_payoff(spot_price, self.K, self.call)
    
class DigitalOption(Option):
    """
    Class representing a digital option.
    
    Methods
    -------
    payoff(spot_price):
        Compute the payoff of the digital option.

    """

    def payoff(self, spot_price):
        """
        Compute the payoff of the digital option.

        Parameters
        ----------
        spot_price : float or np.array
            The spot price(s) of the underlying asset.

        Returns
        -------
        int or np.array
            The payoff(s) of the digital option (0 or 1).
        
        """
        
        return (spot_price > self.K).astype(int) if self.call else (spot_price < self.K).astype(int)

class BarrierOption(Option):
    """
    Class representing a barrier option.
    
    Attributes
    ----------
    barrier : float
        The barrier level.
    barrier_type : str
        The type of barrier option ('up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in').

    Methods
    -------
    payoff(spot_price):
        Compute the payoff of the barrier option.
    knocked_out(spot_price):
        Check if the option has been knocked out.
    knocked_in(spot_price):
        Check if the option has been knocked in.
    option_specific_logic(option_values, spot_prices):
        Apply barrier option-specific logic in the Binomial Tree pricing process.
    
    """
    
    def __init__(self, S, K, T, vol, r, barrier, barrier_type, call = True, american = False):
        """
        Initialise the BarrierOption.

        Parameters
        ----------
        S : float
            Spot price of the underlying asset.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        vol : float
            Volatility of the underlying asset.
        r : float
            Risk-free interest rate.
        barrier : float
            The barrier level.
        barrier_type : str
            The type of barrier option ('up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in').
        call : bool, optional
            True for a call option, False for a put option. Default is True.
        american : bool, optional
            True for an American option, False for a European option. Default is False.
        
        """
        
        super().__init__(S, K, T, vol, r, call, american)
        self.barrier = barrier
        self.barrier_type = barrier_type
        
    def payoff(self, spot_price):
        """
        Compute the payoff of the barrier option.

        Parameters
        ----------
        spot_price : float or np.array
            The spot price(s) of the underlying asset.

        Returns
        -------
        float or np.array
            The payoff(s) of the barrier option.
        
        """
        
        vanilla_payoff = self.vanilla_payoff(spot_price, self.K, self.call)
        
        if self.barrier_type.endswith('-out'):
            return np.where(self.knocked_out(spot_price), 0, vanilla_payoff)
        else: # '-in'
            return np.where(self.knocked_in(spot_price), vanilla_payoff, 0)
        
    def knocked_out(self, spot_price):
        """
        Check if the option has been knocked out.

        Parameters
        ----------
        spot_price : float or np.array
            The spot price(s) of the underlying asset.

        Returns
        -------
        bool or np.array of bool
            True if the option has been knocked out, False otherwise.
        
        """
        
        if self.barrier_type.startswith('up'):
            return spot_price >= self.barrier
        else: # 'down'
            return spot_price <= self.barrier
        
    def knocked_in(self, spot_price):
        """
        Check if the option has been knocked in.

        Parameters
        ----------
        spot_price : float or np.array
            The spot price(s) of the underlying asset.

        Returns
        -------
        bool or np.array of bool
            True if the option has been knocked in, False otherwise.
        
        """
        
        return ~self.knocked_out(spot_price)
    
    def option_specific_logic(self, option_values, spot_prices):
        """
        Apply barrier option-specific logic in the Binomial Tree pricing process.

        Parameters
        ----------
        option_values : np.array
            Option values at a given step in the pricing process.
        spot_prices : np.array
            Spot prices at a given step in the pricing process.

        Returns
        -------
        np.array
            Updated option values after applying barrier option-specific logic.

        Raises
        ------
        ValueError
            If attempting to price an in-barrier option using the binomial tree model.
        
        """
        
        if self.barrier_type.endswith('-out'):
            return np.where(self.knocked_out(spot_prices), 0, option_values)
        else: # '-in'
            raise ValueError("In-barrier options not implemented for binomial tree model.")
 
class PricingModel(ABC):
    """
    Abstract base class for pricing models.
    
    Methods
    -------
    price(option):
        Abstract method to price an option.
    
    """
    
    @abstractmethod 
    def price(self, option):
        """
        Abstract method to price an option.

        Parameters
        ----------
        option : Option
            The option to price.

        Returns
        -------
        float
            The price of the option.
        
        """
        
        pass
    
 
class BlackScholesModel(PricingModel):
    """
    Class implementing the Black-Scholes pricing model for options and allowing vectorised usage.
    
    Methods
    -------
    price(option):
        Price an option using the Black-Scholes model.
    _price_vanilla(option):
        Price a vanilla option using the Black-Scholes formula.
    _price_digital(option):
        Price a digital option using the Black-Scholes model.
    _price_barrier(option):
        Price a barrier option using the Black-Scholes model.
    
    """
    
    @staticmethod
    def price(option):
        """
        Price an option using the Black-Scholes model.

        Parameters
        ----------
        option : Option
            The option to price.

        Raises
        ------
        ValueError
            If the option is American or of an unsupported type.

        Returns
        -------
        float
            The price of the option.
        
        """
        
        if option.american:
            raise ValueError("Black-Scholes model cannot price American options. Use a different method instead.")
        
        option.vectorised_S = np.atleast_1d(option.S)
        option.vectorised_vol = np.atleast_1d(option.vol)
                
        if isinstance(option, VanillaOption):
            return BlackScholesModel._price_vanilla(option)
        elif isinstance(option, DigitalOption):
            return BlackScholesModel._price_digital(option)
        elif isinstance(option, BarrierOption):
            return BlackScholesModel._price_barrier(option)
        else:
            raise ValueError("Unsupported option type for Black-Scholes model")
      
    @staticmethod
    def _price_vanilla(option):
        """
        Price a vanilla option using the Black-Scholes formula, allowing for vectorised usage.

        Parameters
        ----------
        option : VanillaOption
            The vanilla option to price.

        Returns
        -------
        float
            The price of the vanilla option.
        
        """
        
        S = option.vectorised_S
        K = option.K
        vol = option.vectorised_vol
        T = option.T
        r = option.r
        call = option.call
        
        d1 = (np.log(S/K) + (r + vol**2/2)*T)/(vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        if call:
            price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        return price.squeeze()
    
    @staticmethod
    def _price_digital(option):
        """
       Price a digital option using the Black-Scholes model, allowing for vectorised usage.

       Parameters
       ----------
       option : DigitalOption
           The digital option to price.

       Returns
       -------
       float
           The price of the digital option.
       
        """
        
        S = option.vectorised_S
        K = option.K
        vol = option.vol
        T = option.T
        r = option.r
        call = option.call

        d1 = (np.log(S/K) + (r + vol**2/2)*T)/(vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        if call:
            price = np.exp(-r*T)*norm.cdf(d2, 0, 1)
        else:
            price = np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        return price.squeeze()

    @staticmethod
    def _price_barrier(option):
        """
        Price a barrier option using the Black-Scholes model, allowing for vectorised usage.

        Parameters
        ----------
        option : BarrierOption
            The barrier option to price.

        Raises
        ------
        ValueError
            If an invalid barrier option type is specified.

        Returns
        -------
        float
            The price of the barrier option.

        """
        
        K = option.K
        S = option.vectorised_S
        vol = option.vectorised_vol
        T = option.T
        r = option.r
        call = option.call
        barrier_type = option.barrier_type
        barrier = option.barrier
                
        valid_types = ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']
        if barrier_type not in valid_types:
            raise ValueError(f"Barrier option type must be one of {valid_types}")
        
        vanilla_price = BlackScholesModel._price_vanilla(option)
        
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
                
        return result.squeeze()
    
class BinomialTreeModel(PricingModel):
    """
    Class implementing the Binomial Tree pricing model for options.
    
    Attributes
    ----------
    num_steps : int
        Number of steps in the binomial tree.
    asset_model : class
        Model for asset price evolution in the tree.

    Methods
    -------
    price(option):
        Price an option using the Binomial Tree model.
    _price_standard(option):
        Price a standard option using the Binomial Tree model.
    _price_in_barrier(option):
        Price an in-barrier option using the Binomial Tree model and in-out parity.
    
    """
    
    def __init__(self, num_steps = DEFAULT_NUM_STEPS, asset_model = log_normal_model):
        """
        Initialise the BinomialTreeModel.

        Parameters
        ----------
        num_steps : int, optional
            Number of steps in the binomial tree. Default is DEFAULT_NUM_STEPS.
        asset_model : class, optional
            Model for asset price evolution in the tree. Default is log_normal_model.
        
        """
        
        self.num_steps = num_steps
        self.asset_model = asset_model
        
    def price(self, option):
        """
        Price an option using the Binomial Tree model.

        Parameters
        ----------
        option : Option
            The option to price.

        Returns
        -------
        float
            The price of the option.

        Raises
        ------
        ValueError
            If attempting to price an American in-barrier option.
        
        """
        
        if isinstance(option, BarrierOption) and option.barrier_type.endswith('-in'):
            if option.american:
                raise ValueError("Cannot compute American '-in' barrier options with Binomial tree model.")
            else:
                return self._price_in_barrier(option)
        else:
            return self._price_standard(option)
        
    def _price_standard(self, option):
        """
        Price a standard option using the Binomial Tree model, i.e. where the Binomial Tree model can be applied straightforwardly.

        Parameters
        ----------
        option : Option
            The standard option to price.

        Returns
        -------
        float
            The price of the standard option.
        
        """
        
        S = option.S
        vol = option.vol
        r = option.r 
        T = option.T
        american = option.american
        
        payoff = option.payoff
        option_specific_logic = option.option_specific_logic
        
        num_steps = self.num_steps
        asset_model = self.asset_model
        
        dt = T / num_steps
        disc_factor = np.exp(-r*dt)
        
        asset_model_instance = asset_model(S, vol)
        S_expiry = asset_model_instance.computePrice(num_steps + 1, np.arange(0, num_steps + 1, 1), dt)
        
        value = payoff(S_expiry)
        
        for layer in np.arange(num_steps -1, -1, -1):
            price_at_node = asset_model_instance.computePrice(layer, np.arange(0, layer + 1, 1), dt)
            price_up = S_expiry[1 : layer + 2]
            price_down = S_expiry[0 : layer + 1]
            prob_up = ((price_at_node / disc_factor) - price_down) / (price_up - price_down)
            prob_down = 1 - prob_up
            value = disc_factor * (prob_up * value[1 : layer + 2] + prob_down * value[0 : layer + 1])
            
            value = option_specific_logic(value, price_at_node)
            
            if american:
                exercise_value = payoff(price_at_node)
                value = np.maximum(value, exercise_value)
                
            S_expiry = price_at_node
            
        return value[0]
    
    def _price_in_barrier(self, option):
        """
        Price an in-barrier option using the Binomial Tree model and in-out parity.

        Parameters
        ----------
        option : BarrierOption
            The in-barrier option to price.

        Returns
        -------
        float
            The price of the in-barrier option.
        
        """
        
        vanilla_option = VanillaOption(option.S, option.K, option.T, option.vol, option.r, option.call)
        out_barrier_type = option.barrier_type.replace('-in', '-out')
        out_barrier_option = BarrierOption(option.S, option.K, option.T, option.vol, option.r,
                                           option.barrier, out_barrier_type, option.call)
        
        vanilla_price = BlackScholesModel.price(vanilla_option)
        out_barrier_price = self._price_standard(out_barrier_option)
        
        in_barrier_price = vanilla_price - out_barrier_price
        
        return in_barrier_price
