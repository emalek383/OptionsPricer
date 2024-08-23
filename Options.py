"""Module creating classes for option pricing using Black-Scholes and Binomial Tree models.

Classes
-------
OptionFactory: Factory class for creating different types of options.
UnderlyingAsset: Class representing the underlying asset of an option.
Option: Abstract base class for options.
VanillaOption: Class for vanilla options.
DigitalOption: Class for digital options.
BarrierOption: Class for barrier options.
AsianOption: Class for asian options.
SpotPriceModel: Abstract base class for spot price models.
GeometricBrownianMotion: Class for Geometric Brownian Motion model for spot price.
JumpDiffusion: Class for Jump-Diffusion model of spot price.
PricingModel: Abstract base class for pricing models.
BlackScholesModel: Black-Scholes pricing model.
BinomialTreeModel: Binomial Tree pricing model.
MonteCarloModel: Monte Carlo pricing model.
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
            Type of option to create ('vanilla', 'digital', 'barrier', or 'asian').
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
        elif option_type == 'asian':
            return AsianOption(*args, **kwargs)
        else:
            raise ValueError("Invalid option type. Choose from 'vanilla', 'digial', 'barrier' or 'asian'.")

           
class UnderlyingAsset():
    """
    Represents the underlying asset of an option.

    Attributes
    ----------
    S : np.ndarray
        Spot price(s) of the underlying asset.
    vol : np.ndarray
        Volatility of the underlying asset.
    r : float
        Risk-free interest rate.

    Properties
    ----------
    is_vectorised : bool
        True if the asset has multiple spot prices or volatilities.
    
    """
    
    def __init__(self, S, vol, r):
        """
        Initialise the UnderlyingAsset.

        Parameters
        ----------
        S : float or array-like
            Spot price(s) of the underlying asset.
        vol : float or array-like
            Volatility of the underlying asset.
        r : float
            Risk-free interest rate.
        
        """
        
        self.S = np.atleast_1d(S)
        self.vol = np.atleast_1d(vol)
        self.r = r # risk-free rate
        
    @property 
    def is_vectorised(self):
        """
        Check if the asset has multiple spot prices or volatilities.

        Returns
        -------
        bool
            True if the asset has multiple spot prices or volatilities, False otherwise.
        
        """
        
        return len(self.S) > 1 or len(self.vol) > 1

class Option(ABC):
    """
    Abstract base class for options.
    
    Attributes
    ----------
    underlying : UnderlyingAssset
        Instance of UnderlyingAsset containing the spot price and volatility of the underlying asset and the risk-free rate.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
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
    mc_payoff(price_paths):
        Compute the payoff for Monte Carlo simulation.

    """

    def __init__(self, underlying, K, T, call = True, american = False):
        """
        Initialise the Option.

        Parameters
        ----------
        underlying : UnderlyingAssset
            Instance of UnderlyingAsset containing the spot price and volatility of the underlying asset and the risk-free rate.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        call : bool, optional
            True for a Call option, False for a Put option. Default is True.
        american : bool, optional
            True for an American option, False for a European option. Default is False.
        
        """
        
        self.underlying = underlying
        self.K = K
        self.T = T
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
        float or np.ndarray
            The price(s) of the option.
        
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
        float or np.ndarray
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
        float or np.ndarray
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
        np.ndarray
            Updated option values after applying option-specific logic.
        
        """
        
        return option_values
    
    def mc_payoff(self, price_paths):
        """
        Compute the payoff for Monte Carlo simulation.

        Parameters
        ----------
        price_paths : np.ndarray
            Array of simulated price paths.

        Returns
        -------
        npy.ndarray
            Array of payoffs for the simulated price paths.
        
        """
        
        return self.payoff(price_paths)


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
        spot_price : float or np.ndarray
            The spot price(s) of the underlying asset.

        Returns
        -------
        float or np.ndarray
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
        spot_price : float or npnd.array
            The spot price(s) of the underlying asset.

        Returns
        -------
        int or np.ndarray
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
    knocked_out_bt(spot_price):
        Check if the option has been knocked out for binomial tree method.
    knocked_in_bt(spot_price):
        Check if the option has been knocked in for binomial tree method.
    knocked_out_mc(price_paths):
        Check if the option has been knocked out for Monte Carlo method.
    knocked_in_mc(price_paths):
        Check if the option has been knocked in for Monte Carlo method.
    option_specific_logic(option_values, spot_prices):
        Apply barrier option-specific logic in the Binomial Tree pricing process.
    mc_payoff(price_paths):
        Compute the payoff for Monte Carlo simulation.
    
    """
    
    def __init__(self, underlying, K, T, barrier, barrier_type, call = True, american = False):
        """
        Initialise the BarrierOption.

        Parameters
        ----------
        underlying : UnderlyingAssset
            Instance of UnderlyingAsset containing the spot price and volatility of the underlying asset and the risk-free rate.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        barrier : float
            The barrier level.
        barrier_type : str
            The type of barrier option ('up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in').
        call : bool, optional
            True for a call option, False for a put option. Default is True.
        american : bool, optional
            True for an American option, False for a European option. Default is False.
        
        """
        
        super().__init__(underlying, K, T, call, american)
        self.barrier = barrier
        self.barrier_type = barrier_type
        
    def payoff(self, spot_price):
        """
        Compute the payoff of the barrier option.

        Parameters
        ----------
        spot_price : float or np.ndarray
            The spot price(s) of the underlying asset.

        Returns
        -------
        float or np.ndarray
            The payoff(s) of the barrier option.
        
        """
        
        vanilla_payoff = self.vanilla_payoff(spot_price, self.K, self.call)
        
        if self.barrier_type.endswith('-out'):
            return np.where(self.knocked_out_bt(spot_price), 0, vanilla_payoff)
        else: # '-in'
            return np.where(self.knocked_in_bt(spot_price), vanilla_payoff, 0)
   
    def knocked_out_bt(self, spot_price):
        """
        Check if the option has been knocked out for binomial tree method.

        Parameters
        ----------
        spot_price : float or np.ndarray
            The spot price(s) of the underlying asset.

        Returns
        -------
        bool or np.ndarray
            True if the option has been knocked out, False otherwise.
        
        """
        
        if self.barrier_type.startswith('up'):
            return spot_price >= self.barrier
        else:  # 'down'
            return spot_price <= self.barrier

    def knocked_in_bt(self, spot_price):
        """
        Check if the option has been knocked in for binomial tree method.

        Parameters
        ----------
        spot_price : float or np.ndarray
            The spot price(s) of the underlying asset.

        Returns
        -------
        bool or np.ndarray
            True if the option has been knocked in, False otherwise.
        
        """
        
        return ~self.knocked_out_bt(spot_price)

    def knocked_out_mc(self, price_paths):
        """
        Check if the option has been knocked out for Monte Carlo method.

        Parameters
        ----------
        price_paths : np.ndarray
            Array of simulated price paths.

        Returns
        -------
        np.ndarray
            Boolean array indicating whether each path has been knocked out.
        
        """
        
        if self.barrier_type.startswith('up'):
            return np.any(price_paths >= self.barrier, axis=-2)
        else:  # 'down'
            return np.any(price_paths <= self.barrier, axis=-2)

    def knocked_in_mc(self, price_paths):
        """
        Check if the option has been knocked in for Monte Carlo method.

        Parameters
        ----------
        price_paths : np.ndarray
            Array of simulated price paths.

        Returns
        -------
        np.ndarray
            Boolean array indicating whether each path has been knocked in.
        
        """
        
        if self.barrier_type.startswith('up'):
            return np.any(price_paths >= self.barrier, axis=-2)
        else:  # 'down'
            return np.any(price_paths <= self.barrier, axis=-2)
    
    def option_specific_logic(self, option_values, spot_prices):
        """
        Apply barrier option-specific logic in the Binomial Tree pricing process.

        Parameters
        ----------
        option_values : np.ndarray
            Option values at a given step in the pricing process.
        spot_prices : np.ndarray
            Spot prices at a given step in the pricing process.

        Returns
        -------
        np.ndarray
            Updated option values after applying barrier option-specific logic.

        Raises
        ------
        ValueError
            If attempting to price an in-barrier option using the binomial tree model.
        
        """
        
        if self.barrier_type.endswith('-out'):
            return np.where(self.knocked_out_bt(spot_prices), 0, option_values)
        else: # '-in'
            raise ValueError("In-barrier options not implemented for binomial tree model.")
            
    def mc_payoff(self, price_paths):
        """
        Compute the payoff of the barrier option for Monte Carlo simulation.

        Parameters
        ----------
        price_paths : np.ndarray
            Array of simulated price paths.

        Returns
        -------
        np.ndarray
            Array of payoffs for the simulated price paths.
        
        """
        
        final_prices = price_paths[..., -1, :]
        vanilla_payoff = self.vanilla_payoff(final_prices, self.K, self.call)
        
        if self.barrier_type.endswith('-out'):
            knocked_out = self.knocked_out_mc(price_paths)
            return np.where(knocked_out, 0, vanilla_payoff)
        else:  # '-in'
            knocked_in = self.knocked_in_mc(price_paths)
            return np.where(knocked_in, vanilla_payoff, 0)

class AsianOption(Option):
    """
    Class representing an Asian option.
    
    Attributes
    ----------
    underlying : UnderlyingAsset
        The underlying asset of the option.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    call : bool
        True for a call option, False for a put option.
    american : bool
        True for an American option, False for a European option.
    averaging_type : str
        Type of averaging: 'arithmetic' or 'geometric'.
    averaging_points : np.ndarray
        Array of time points for averaging.
        
    """
    
    def __init__(self, underlying, K, T, averaging_type='arithmetic', averaging_freq='monthly', averaging_points=None, call=True, american=False):
        """
        Initialise the AsianOption.

        Parameters
        ----------
        underlying : UnderlyingAsset
            The underlying asset of the option.
        K : float
            Strike price of the option.
        T : float
            Time to maturity in years.
        averaging_type : str, optional
            Type of averaging: 'arithmetic' or 'geometric'. Default is 'arithmetic'.
        averaging_freq : str, optional
            Frequency of averaging: 'monthly', 'weekly', or 'daily'. Default is 'monthly'.
        averaging_points : int or array-like, optional
            Number of equally spaced averaging points or specific averaging points.
        call : bool, optional
            True for a call option, False for a put option. Default is True.
        american : bool, optional
            True for an American option, False for a European option. Default is False.
        """
        
        super().__init__(underlying, K, T, call, american)
        self.averaging_type = averaging_type
        
        if averaging_points is None:
            if averaging_freq == 'monthly':
                num_months = int(np.ceil(T * 12))
                self.averaging_points = np.linspace(0, T, num_months + 1)[1:]
            elif averaging_freq == 'weekly':
                num_weeks = int(np.ceil(T * 52))
                self.averaging_points = np.linspace(0, T, num_weeks + 1)[1:]
            elif averaging_freq == 'daily':
                num_days = int(np.ceil(T * 252))
                self.averaging_points = np.linspace(0, T, num_days + 1)[1:]
            else:
                raise ValueError("Invalid averaging_freq. Use 'monthly', 'weekly', 'daily', or specify averaging_points.")
        elif isinstance(averaging_points, int):
            self.averaging_points = np.linspace(0, T, averaging_points + 1)[1:]
        else:
            self.averaging_points = np.array(averaging_points)
    
    def payoff(self, price_paths):
        """
        Compute the payoff of the Asian option.

        Parameters
        ----------
        price_paths : np.ndarray
            Array of price paths. Shape: (..., num_time_steps, num_simulations)

        Returns
        -------
        np.ndarray
            Array of payoffs. Shape: (..., num_simulations)
            
        """
        
        num_steps = price_paths.shape[-2] - 1  # Subtract 1 because we include t=0
        time_points = np.linspace(0, self.T, num_steps + 1)
        
        # Find the indices closest to our averaging points
        indices = np.searchsorted(time_points, self.averaging_points)
        
        if self.averaging_type == 'arithmetic':
            average_prices = np.mean(price_paths[..., indices, :], axis=-2)
        elif self.averaging_type == 'geometric':
            average_prices = np.exp(np.mean(np.log(price_paths[..., indices, :]), axis=-2))
        else:
            raise ValueError("Invalid averaging type. Use 'arithmetic' or 'geometric'.")
        
        return self.vanilla_payoff(average_prices, self.K, self.call)
        

class SpotPriceModel(ABC):
    """
    Abstract base class for spot price models.
    
    Attributes
    ----------
    underlying : UnderlyingAsset
        The underlying asset of the option.
    S : np.ndarray
        Spot price(s) of the underlying asset.
    vol : np.ndarray
        Volatility of the underlying asset.
    r : float
        Risk-free interest rate.
    
    """
    
    def __init__(self, underlying):
        """
        Initialise the SpotPriceModel.

        Parameters
        ----------
        underlying : UnderlyingAssset
            Instance of UnderlyingAsset containing the spot price and volatility of the underlying asset and the risk-free rate.

        Returns
        -------
        None.

        """

        self.underlying = underlying
        self.S = self.underlying.S
        self.vol = self.underlying.vol
        self.r = self.underlying.r
        
    @abstractmethod 
    def generate_paths(self, T, num_simulations, num_steps, antithetic = False, is_path_dependent = False):
        """
        Generate asset price paths for Monte Carlo simulation.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        num_simulations : int
            Number of simulations to run.
        num_steps : int
            Number of time steps in each simulation.
        antithetic : bool, optional
            Whether to use antithetic variates for variance reduction. Default is False.
        is_path_dependent : bool, optional
            Whether the option is path-dependent. Default is False.

        Returns
        -------
        np.ndarray
            Array of simulated asset price paths.
        
        """
        
        pass
    
    @abstractmethod
    def compute_price(self, i, j, dt):
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
        
        pass
    
    
class GeometricBrownianMotion(SpotPriceModel):
    """
    Geometric Brownian Motion model for spot price evolution.
    
    Methods
    -------
    generate_paths(T, num_simulations, num_steps, antithetic = False, is_path_dependent = False):
        Generate asset price paths for Monte Carlo simulation.
    adjust_paths(unit_vol_paths, T):
        Adjust the generated paths for path-dependent options.
    compute_price(i, j, dt):
        Compute the stock price at a given node in the binomial tree.
        
    """
    
    def __init__(self, underlying):
        """
        Initialise the GeometricBrownianMotion model.

        Parameters
        ----------
        underlying : UnderlyingAsset
            The underlying asset of the option.
        
        """
        
        super().__init__(underlying)
    
    def generate_paths(self, T, num_simulations, num_steps, antithetic = False, is_path_dependent = False):
        """
        Generate asset price paths for Monte Carlo simulation.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        num_simulations : int
            Number of simulations to run.
        num_steps : int
            Number of time steps in each simulation.
        antithetic : bool, optional
            Whether to use antithetic variates for variance reduction. Default is False.
        is_path_dependent : bool, optional
            Whether the option is path-dependent. Default is False.

        Returns
        -------
        np.ndarray
            Array of simulated asset price paths.
        
        """
        
        if not is_path_dependent:
            S = self.S[:, np.newaxis, np.newaxis]
            vol = self.vol[:, np.newaxis, np.newaxis]
            
            nu_T = (self.r - 0.5 * vol**2) * T
            vol_sqrt_T = vol*np.sqrt(T)
            
            Z = np.random.standard_normal(num_simulations)
            if antithetic:
                Z = np.concatenate([Z, -Z], axis = -1)
        
            S_paths = S * np.exp(nu_T + vol_sqrt_T * Z)
        
            return S_paths
        
        else:
            dt = T / num_steps
            
            Z = np.random.standard_normal((num_steps, num_simulations))
            if antithetic:
                Z = np.concatenate([Z, -Z], axis = 1)
                
            # Generate paths with unit volatility and no drift which will be adjusted later to allow for fast vectorisation
            unit_vol_paths = np.exp(np.cumsum(np.sqrt(dt) * Z, axis = 0))
            unit_vol_paths = np.insert(unit_vol_paths, 0, 1, axis = 0)
            
            return unit_vol_paths
    
    def adjust_paths(self, unit_vol_paths, T):
        """
        Adjust the generated paths for path-dependent options.

        Parameters
        ----------
        unit_vol_paths : np.ndarray
            Array of price paths with unit volatility and no drift.
        T : float
            Time to maturity in years.

        Returns
        -------
        np.ndarray
            Array of adjusted price paths.
        
        """
        
        num_steps = unit_vol_paths.shape[0] - 1
        dt = T / num_steps
        
        S = self.S[:, np.newaxis, np.newaxis]
        vol = self.vol[:, np.newaxis, np.newaxis]
        
        stochastic_part = unit_vol_paths[np.newaxis, :, :] ** vol
        drift_part = np.exp((self.r - 0.5 * vol**2) * np.arange(num_steps + 1)[np.newaxis, :, np.newaxis] * dt)
        
        return S * drift_part * stochastic_part
    
    def compute_price(self, i, j, dt):
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
        np.ndarray
            Stock price(s) at the specified node.
        
        """
        
        return self.S * np.exp(self.vol * np.sqrt(dt) * (2 * j - i))
    
class JumpDiffusionModel(SpotPriceModel):
    """
    Merton Jump Diffusion model for spot price evolution.

    Attributes
    ----------
    lambda_jump : float
        Average number of jumps per year.
    mu_jump : float
        Average jump size.
    sigma_jump : float
        Volatility of jump size.

    Methods
    -------
    generate_paths(T, num_simulations, num_steps, antithetic=False):
        Generate asset price paths for Monte Carlo simulation.
    compute_price(i, j, dt):
        Compute the stock price at a given node in the binomial tree.
    
    """
    
    def __init__(self, underlying, lambda_jump, mu_jump, sigma_jump):
        """
        Initialise the JumpDiffusionModel.

        Parameters
        ----------
        underlying : UnderlyingAsset
            The underlying asset of the option.
        lambda_jump : float
            Average number of jumps per year.
        mu_jump : float
            Average jump size.
        sigma_jump : float
            Volatility of jump size.
        
        """
        
        super().__init__(underlying)
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        
    def generate_paths(self, T, num_simulations, num_steps, antithetic = False):
        """
        Generate asset price paths for Monte Carlo simulation.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        num_simulations : int
            Number of simulations to run.
        num_steps : int
            Number of time steps in each simulation.
        antithetic : bool, optional
            Whether to use antithetic variates for variance reduction. Default is False.

        Returns
        -------
        np.ndarray
            Array of simulated asset price paths.
        
        """
        
        dt = T / num_steps
        nu_dt = (self.r - 0.5 * self.vol**2 - self.lambda_jump * (np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1)) * dt
        vol_sqrt_dt = self.vol * np.sqrt(dt)
        
        Z = np.random.standard_normal((num_steps, num_simulations))
        N = np.random.possion(self.lambda_jump * dt, (num_steps, num_simulations))
        J = np.random.normal(self.mu_jump, self.sigma_jump, (num_steps, num_simulations))
        
        if antithetic:
            Z = np.concatenate([Z, -Z], axis = 1)
            N = np.concatenate([N, N], axis = 1)
            J = np.concatenate([J, J], axis = 1)
        
        increments = nu_dt + vol_sqrt_dt * Z + N * J
        log_returns = np.cumsum(increments, axis = 0)
        
        S_paths = self.S * np.exp(log_returns)
        S_paths = np.insert(S_paths, 0, self.S, axis = 0)
        
        return S_paths
    
    def compute_price(self, i, j, dt):
        """
        Compute the stock price at a given node in the binomial tree.

        This method approximates the jump process by adjusting volatility to include the jump component.

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
        np.ndarray
            Stock price(s) at the specified node.
        
        """
        
        # Approximate jump process by adjusting volatility to include the jump component
        total_vol = np.sqrt(self.vol**2 + self.lambda_jump * (self.mu_jump ** 2 + self.sigma_jump ** 2) / dt)
        return self.S * np.exp(total_vol * np.sqrt(dt) * (2 * j - i))
    

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
        float or np.ndarray
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
    _price_geometric_asian(option):
        Price a geometric Asian option using the Black-Scholes model.
    
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
        float or np.ndarray
            The price of the option.
        
        """
        
        if option.american:
            raise ValueError("Black-Scholes model cannot price American options. Use a different method instead.")
        
        if isinstance(option, VanillaOption):
            return BlackScholesModel._price_vanilla(option)
        elif isinstance(option, DigitalOption):
            return BlackScholesModel._price_digital(option)
        elif isinstance(option, BarrierOption):
            return BlackScholesModel._price_barrier(option)
        elif isinstance(option, AsianOption) and option.averaging_type == 'geometric':
            return BlackScholesModel._price_geometric_asian(option)
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
        float or np.ndarray
            The price(s) of the vanilla option.
        
        """
        
        S = option.underlying.S
        K = option.K
        vol = option.underlying.vol
        T = option.T
        r = option.underlying.r
        call = option.call
        
        d1 = (np.log(S/K) + (r + vol**2/2)*T)/(vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        if call:
            price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        
        if not option.underlying.is_vectorised:
            return price.item()
        
        return price
            
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
       float or np.ndarray
           The price(s) of the digital option.
       
        """
        
        S = option.underlying.S
        K = option.K
        vol = option.underlying.vol
        T = option.T
        r = option.underlying.r
        call = option.call

        d1 = (np.log(S/K) + (r + vol**2/2)*T)/(vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        if call:
            price = np.exp(-r*T)*norm.cdf(d2, 0, 1)
        else:
            price = np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        
        if not option.underlying.is_vectorised:
            return price.item()
        
        return price

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
        float or np.ndarray
            The price(s) of the barrier option.

        """
        
        S = option.underlying.S
        K = option.K
        vol = option.underlying.vol
        T = option.T
        r = option.underlying.r
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
                
        if not option.underlying.is_vectorised:
            return result.item()
        
        return result
    
    @staticmethod
    def _price_geometric_asian(option):
        """
        Price a geometric Asian option using the Black-Scholes model.

        Parameters
        ----------
        option : AsianOption
            The geometric Asian option to price.

        Returns
        -------
        float or np.ndarray
            The price(s) of the geometric Asian option.
        
        """
        
        S = option.underlying.S
        K = option.K
        vol = option.underlying.vol
        T = option.T
        r = option.underlying.r
        n = len(option.averaging_points)
        
        sigma_adj = vol * np.sqrt((2*n+1) / (6*(n+1)))
        mu_adj = (r - 0.5*vol**2) * (n+1)/(2*n) + vol**2/6
        
        d1 = (np.log(S/K) + (mu_adj + 0.5*sigma_adj**2)*T) / (sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)
        
        if option.call:
            price = np.exp(-r*T) * (S*np.exp(mu_adj*T)*norm.cdf(d1) - K*norm.cdf(d2))
        else:
            price = np.exp(-r*T) * (K*norm.cdf(-d2) - S*np.exp(mu_adj*T)*norm.cdf(-d1))
        
        return price.item() if not option.underlying.is_vectorised else price
    
class BinomialTreeModel(PricingModel):
    """
    Class implementing the Binomial Tree pricing model for options.
    
    Attributes
    ----------
    num_steps : int
        Number of steps in the binomial tree.
    spot_model : SpotPriceModel
        Spot price model for the asset price evolution in the tree.

    Methods
    -------
    price(option):
        Price an option using the Binomial Tree model.
    _price_standard(option):
        Price a standard option using the Binomial Tree model.
    _price_in_barrier(option):
        Price an in-barrier option using the Binomial Tree model and in-out parity.
    
    """
    
    def __init__(self, num_steps = DEFAULT_NUM_STEPS, spot_model_class = GeometricBrownianMotion):
        """
        Initialise the BinomialTreeModel.

        Parameters
        ----------
        num_steps : int, optional
            Number of steps in the binomial tree. Default is DEFAULT_NUM_STEPS.
        spot_model : class, optional
            Model for spot price evolution in the tree. Default is GeometricBrownianMotion.
        
        """
        
        self.num_steps = num_steps
        self.spot_model_class = spot_model_class
        
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
                
        num_steps = self.num_steps
        
        dt = option.T / num_steps
        disc_factor = np.exp(-option.underlying.r*dt)
        
        spot_model = self.spot_model_class(option.underlying)
        S_expiry = spot_model.compute_price(num_steps + 1, np.arange(0, num_steps + 1, 1), dt)
        
        value = option.payoff(S_expiry)
        
        for layer in np.arange(num_steps -1, -1, -1):
            price_at_node = spot_model.compute_price(layer, np.arange(0, layer + 1, 1), dt)
            price_up = S_expiry[1 : layer + 2]
            price_down = S_expiry[0 : layer + 1]
            prob_up = ((price_at_node / disc_factor) - price_down) / (price_up - price_down)
            prob_down = 1 - prob_up
            value = disc_factor * (prob_up * value[1 : layer + 2] + prob_down * value[0 : layer + 1])
            
            value = option.option_specific_logic(value, price_at_node)
            
            if option.american:
                exercise_value = option.payoff(price_at_node)
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
        
        vanilla_option = VanillaOption(option.underlying, option.K, option.T, option.call)
        out_barrier_type = option.barrier_type.replace('-in', '-out')
        out_barrier_option = BarrierOption(option.underlying, option.K, option.T, option.barrier, out_barrier_type, option.call)
        
        vanilla_price = BlackScholesModel.price(vanilla_option)
        out_barrier_price = self._price_standard(out_barrier_option)
        
        in_barrier_price = vanilla_price - out_barrier_price
        
        return in_barrier_price

class MonteCarloModel(PricingModel):
    """
    Class implementing the Monte Carlo pricing model for options.
    
    Attributes
    ----------
    num_simulations : int
        Number of simulations to run in the Monte Carlo method.
    num_steps : int
        Number of time steps in each simulation.
    variance_reduction : str or None
        Type of variance reduction technique to use ('antithetic', 'control_variate', or None).
    spot_model_class : class
        Class of the spot price model to use for generating price paths.

    Methods
    -------
    price(option):
        Price an option using the Monte Carlo method.
    _calculate_payoffs(option, S_paths):
        Calculate the payoffs for an option given the simulated price paths.
    _control_variate(option, payoffs, S_T):
        Apply the control variate technique for variance reduction.
    """
    
    def __init__(self, num_simulations = 10_000, num_steps = 252, variance_reduction = None, spot_model_class = GeometricBrownianMotion):
        """
        Initialise the MonteCarloModel.

        Parameters
        ----------
        num_simulations : int, optional
            Number of simulations to run. Default is 10,000.
        num_steps : int, optional
            Number of time steps in each simulation. Default is 252.
        variance_reduction : str or None, optional
            Type of variance reduction technique to use. Default is None.
        spot_model_class : class, optional
            Class of the spot price model to use. Default is GeometricBrownianMotion.
            
        Returns
        -------
        None.
        
        """
        
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.variance_reduction = variance_reduction
        self.spot_model_class = spot_model_class
        
    def price(self, option):
        """
        Price an option using the Monte Carlo method.

        Parameters
        ----------
        option : Option
            The option to price.

        Raises
        ------
        ValueError
            If attempting to price an American option.
            
        Returns
        -------
        float or np.ndarray
            The price(s) of the option.
            
        """
        
        if option.american:
            raise ValueError("Monte Carlo Method not valid for American options.")
            
        spot_model = self.spot_model_class(option.underlying)
        
        is_path_dependent = isinstance(option, (BarrierOption, AsianOption))
        
        if self.variance_reduction == 'antithetic':
            S_paths = spot_model.generate_paths(option.T, self.num_simulations // 2, self.num_steps, antithetic = True, is_path_dependent = is_path_dependent)
        else:
            S_paths = spot_model.generate_paths(option.T, self.num_simulations, self.num_steps, is_path_dependent = is_path_dependent)
        
        if is_path_dependent:
            S_paths = spot_model.adjust_paths(S_paths, option.T)
        
        payoffs = self._calculate_payoffs(option, S_paths)
        
        if self.variance_reduction == 'control_variate':
            if not isinstance(option, VanillaOption):
                return self._control_variate(option, payoffs, S_paths)
        
        option_price = np.exp(-option.underlying.r * option.T) * np.mean(payoffs, axis = -1)
        
        if not option.underlying.is_vectorised:
            return option_price.item()
        
        else:
            return option_price
    
    def _calculate_payoffs(self, option, S_paths):
        """
        Calculate the payoffs for an option given the simulated price paths.

        Parameters
        ----------
        option : Option
            The option for which to calculate payoffs.
        S_paths : np.ndarray
            Array of simulated price paths.

        Returns
        -------
        np.ndarray
            Array of payoffs for the option.
        
        """
        
        if isinstance(option, (BarrierOption, AsianOption)):
            return option.mc_payoff(S_paths)
        else:  # VanillaOption or DigitalOption
            return option.mc_payoff(S_paths[..., -1, :])
    
    def _control_variate(self, option, payoffs, S_paths):
        """
        Apply the control variate technique for variance reduction.

        Parameters
        ----------
        option : Option
            The option being priced.
        payoffs : np.ndarray
            Array of payoffs from the Monte Carlo simulation.
        S_paths : np.ndarray
            Array of simulated asset prices.

        Returns
        -------
        option_price : float or np.ndarray
            The option price(s) after applying the control variate technique.
        
        """
        
        S_T = S_paths[..., -1, :]
        
        if isinstance(option, DigitalOption):
            cv_payoffs, cv_price = self._vanilla_control_variate(option, S_T)
        elif isinstance(option, BarrierOption):
            cv_payoffs, cv_price = self._vanilla_control_variate(option, S_T)
        elif isinstance(option, AsianOption):
            if option.averaging_type == 'arithmetic':
                cv_payoffs, cv_price = self._geometric_asian_control_variate(option, S_paths)
            else:  # geometric
                cv_payoffs, cv_price = self._vanilla_control_variate(option, S_T)
        else:
            raise ValueError("Unsupported option type for control variate")
            
        payoffs = np.atleast_2d(payoffs)
        cv_payoffs = np.atleast_2d(cv_payoffs)
        cv_price = np.atleast_1d(cv_price)
            
        # Initialize the mask for valid prices
        valid_mask = np.ones_like(cv_price, dtype=bool)

        # Check for invalid cv_price
        invalid_cv_price = np.isnan(cv_price) | np.isinf(cv_price)
        if np.any(invalid_cv_price):
            print(f"Warning: Invalid control variate price(s) for {option.__class__.__name__}. Falling back to standard Monte Carlo for these cases.")
            valid_mask[invalid_cv_price] = False

        cov_matrices = np.array([np.cov(payoffs[i], cv_payoffs[i]) for i in range(payoffs.shape[0])])
        
        # Check for numerical instability
        unstable_cov = np.isclose(cov_matrices[:, 1, 1], 0, atol=1e-15)
        if np.any(unstable_cov):
            print(f"Warning: Unstable control variate(s) for {option.__class__.__name__}. Falling back to standard Monte Carlo for these cases.")
            valid_mask[unstable_cov] = False

        beta = np.where(valid_mask, cov_matrices[..., 0, 1] / cov_matrices[..., 1, 1], 0)

        controlled_payoffs = payoffs - beta[..., np.newaxis] * (cv_payoffs - cv_price[..., np.newaxis])

        option_price = np.exp(-option.underlying.r * option.T) * np.mean(controlled_payoffs, axis=1)

        # Check for invalid option prices
        invalid_option_price = np.isnan(option_price) | np.isinf(option_price)
        if np.any(invalid_option_price):
            print(f"Warning: Invalid price(s) computed for {option.__class__.__name__}. Falling back to standard Monte Carlo for these cases.")
            valid_mask[invalid_option_price] = False

        # Fall back to standard Monte Carlo for invalid cases
        standard_mc_price = np.exp(-option.underlying.r * option.T) * np.mean(payoffs, axis=1)
        final_price = np.where(valid_mask, option_price, standard_mc_price)

        if final_price.size == 1:
            return final_price.item()
        else:
            return final_price
            
    def _vanilla_control_variate(self, option, S_T):     
        vanilla_option = VanillaOption(option.underlying, option.K, option.T, option.call)
        bs_price = BlackScholesModel.price(vanilla_option)
        cv_payoffs = vanilla_option.payoff(S_T)
        return cv_payoffs, bs_price
        
    def _geometric_asian_control_variate(self, option, S_paths):
        geo_asian_option = AsianOption(option.underlying, option.K, option.T,
                                       averaging_type = 'geometric',
                                       averaging_points = option.averaging_points,
                                       call = option.call)
        bs_price = BlackScholesModel.price(geo_asian_option)
        cv_payoffs = geo_asian_option.payoff(S_paths)
        return cv_payoffs, bs_price