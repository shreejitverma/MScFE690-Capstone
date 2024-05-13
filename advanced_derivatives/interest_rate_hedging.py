# interest rate hedging

import numpy as np
import matplotlib.pyplot as plt

class InterestRateSwap:
    """
    This class indicates the swap interest rate. This will be utilized to in derivative to manage interest rate.
    Attributes:
        principal: Is the swap principal and it is in floating.
        fixed_rate: It indicated the fixed interest rate and it is in floating.
        float_rate_ind: Indicate the floating rate index and it is string (LIBOR, SOFIA, SOFR).
        mat: Is the swap maturity in years and it is in integer.
    """
    def __init__(self, principal, fixed_rate, float_rate_ind, mat):
        self.principal = principal
        self.fixed_rate = fixed_rate
        self.float_rate_ind = float_rate_ind
        self.mat = mat

    def value(self, int_rate):
        # The return will provide the payoff, which calculated by finding the fixed rate and floating rates difference and multiplied by principal 
        return (self.fixed_rate - int_rate) * self.principal


class InterestRateFuture:
    """
    This class indicate the future interest rate of the contact.
    Attributes:
        cont_size: This indicates the each contract size in terms of monetary valueand in this floating.
        set_price: This the settelment price that sttles the future contract and it is in floating. 
        mat: This is the maturity and provide the future expiration in years and it is in integer.
    """
    def __init__(self, cont_size, set_price, mat): 
        self.cont_size = cont_size
        self.set_price = set_price
        self.mat = mat

    def value(self, interest_rate):
        # The return will provide the payoff, which calculated by finding settlement price differences
        return (self.set_price - int_rate) * self.cont_size


class InterestRateOption:
    """
    This class indicates the interest rates option.
    Attributes:
        str_price: This strike price, the option can be excersized at this rate and it is floating.
        pre: The premium is the price to purchase the option and it is floating.
        type: This is type which indicate the of option type, call or put and it is string.
        mat: This is the maturity and provide the option expiration in years and it is integer.
    """
    def __init__(self, str_price, pre, type, mat):
        self.str_price = str_price
        self.pre = pre
        self.type = type
        self.mat = mat

    def value(self, interest_rate):
        # The return will provide the payoff, which calculated on call or put option
        if self.type == 'call':
            return max(int_rate - self.str_price, 0) * self.pre
        else:
            return max(self.str_price - int_rate, 0) * self.pre

 

def vasicek(r0, a, b, sigma, T, dt=1/252.):
    """
    This function by using Vasicek model evalute interest rate.
    Args:
        r0: This is the initial interest rate, and it is floating.
        a: Thisi is the rate of reversion to the mean, and it is floating.
        b: This is long-term mean rate, and it is floating.
        sigma: This is the interest rate volatility nad it is floating.
        T (int): This is time in days, and it is in integer .
        dt: This the step of time in years, and it is floating.
    Returns:
        It evaluare the interest rate.
    """
    N = int(T / dt)
    rates = np.zeros(N)
    rates[0] = r0
    for t in range(1, N):
        dr = a * (b - rates[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        rates[t] = rates[t-1] + dr
    return rates


def assess_interest_rate_risk(rates):
    """
    This function based on the evaluated interest rate variance evaluating the risk.
    Args:
        rates: This is evaluated  interest rates array.
    Returns:
        This will return the rate variance in floating. 
    """
    return np.var(rates)

def plot_rates(rates):
    """
    This function will plot the evaluated interest rate.
    Args:
        rates: This is evaluated  interest rates array.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rates)
    plt.title("Evaluated Interest Rate")
    plt.xlabel("Time Steps")
    plt.ylabel("Interest Rate")
    plt.grid(True)
    plt.show()

    

def select_hedging_instrument(risk, insts, risk_tol):
    """
    This function based on evaluated risk will select the instrument that required to be hedged.
    Arguments:
        risk: This is the evaluated risk and it is floating.
        inst: This indicates a list of instruments available for hedging and it is a list.
    Returns:
        list: Based on the evaluated risk it will indicat the list of the appropriate instruments for hedging and it is a list.
    """

    if risk > risk_tol:
        return [inst for inst in insts if isinstance(inst, InterestRateSwap)]
    else:
        return [inst for inst in insts if isinstance(inst, InterestRateOption)]



def plot_payoffs(rates, insts):
    """
    This will plot for each hedging the payoff time series.
    Args:
        rates: This is an array of evaluated interest rates.
        instruments: This is the hedging list.
    """
    plt.figure(figsize=(14, 7))
    for inst in insts:
        payoffs = [inst.value(rate) for rate in rates]
        plt.plot(payoffs, label=type(inst).__name__)

    plt.title("Hedging Instrument Payoffs Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.show()


