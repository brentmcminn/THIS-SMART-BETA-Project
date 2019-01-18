# -*- coding: utf-8 -*-
"""
Smart Beta portfolio implementation on Tiingo Dataset

@author: Brent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cvxpy as cvx

wdir = '/Users/brentmcminn/Documents/GitHub/THIS-SMART-BETA-Project'

close = pd.read_csv(wdir+'/close.csv',
                    index_col = 0,
                    parse_dates = True)

volume = pd.read_csv(wdir+'/volume.csv',
                    index_col = 0,
                    parse_dates = True)
 
def generate_dollar_volume_weights(close, volume):
    """
    Generate dollar volume weights.

    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    volume : str
        Volume for each ticker and date

    Returns
    -------
    dollar_volume_weights : DataFrame
        The dollar volume weights for each ticker and date
    """
    assert close.index.equals(volume.index)
    assert close.columns.equals(volume.columns)
    
    
    product_close_volume = volume.values * close.values
    
    df_close_volume = pd.DataFrame(product_close_volume, 
                                   columns = volume.columns, 
                                   index = volume.index)
    
    df_total_turnover = df_close_volume.sum(axis=1)
    dollar_volume_weights = df_close_volume.div(df_total_turnover, 
                                                axis=0) 
    
    return dollar_volume_weights

index_weights = generate_dollar_volume_weights(close, volume)


def calculate_momentum_weights(close):
    """
    Calculate dividend weights.

    Parameters
    ----------
    close : DataFrame
        closing prices dataframe

    Returns
    -------
    momentum_weights : DataFrame
        Weights for each stock and date, based on 12m-1m momentum
    """
    
    dfmomentum = np.log(close.pct_change(252).shift(20) + 1)
    
    #force 0 score for stocks that are down yoy
    dfmomentum[dfmomentum < 0] = 0

    mom_sum = dfmomentum.sum(axis=1)
    
    # calculating the momentum weights
    mom_weights = dfmomentum.div(mom_sum, axis =0)

    return mom_weights

etf_weights = calculate_momentum_weights(close)

#Quick plot to check maximum weight by day

plt.plot(index_weights.max(axis = 1), label = 'Index')
plt.plot(etf_weights.max(axis = 1), label = 'ETF')
plt.legend()
plt.title('Max Index,ETF Weights by Day')
plt.show()

returns = close.pct_change()
returns = returns[returns < 0.5]

def generate_weighted_returns(returns, weights):
    """
    Generate weighted returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    weights : DataFrame
        Weights for each ticker and date

    Returns
    -------
    weighted_returns : DataFrame
        Weighted returns for each ticker and date
    """
    assert returns.index.equals(weights.index)
    assert returns.columns.equals(weights.columns)
    
    weighted_returns = pd.DataFrame(weights.values
                                    *(1 + returns.values),
                                    columns = returns.columns, 
                                    index = returns.index)
      
    return weighted_returns

etf_weighted_returns = generate_weighted_returns(returns, etf_weights)

#Get rid of nans in the first year of the data

etf_weighted_returns = etf_weighted_returns[~(etf_weighted_returns.isnull(
                                                ).all(axis = 1))]
index_weighted_returns = generate_weighted_returns(returns, index_weights)
#reindex index weighted returns to match ETF weighted returns
index_weighted_returns = index_weighted_returns.reindex(etf_weighted_returns.index)

def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date

    Returns
    -------
    cumulative_returns : Pandas Series
        Cumulative returns for each date
    """

    # comulative product returns for the returns
    cumulative_returns = (returns).sum(axis = 1)
    cumulative_returns[cumulative_returns == 0] = 1
    cumulative_returns = np.cumprod(cumulative_returns)
    return cumulative_returns

index_weighted_cumulative_returns = calculate_cumulative_returns(index_weighted_returns)
etf_weighted_cumulative_returns = calculate_cumulative_returns(etf_weighted_returns)

plt.plot(index_weighted_cumulative_returns, label = 'Index')
plt.plot(etf_weighted_cumulative_returns, label = 'ETF')
plt.legend()
plt.show()

def tracking_error(benchmark_returns_by_date, etf_returns_by_date):
    """
    Calculate the tracking error.

    Parameters
    ----------
    benchmark_returns_by_date : Pandas Series
        The benchmark returns for each date
    etf_returns_by_date : Pandas Series
        The ETF returns for each date

    Returns
    -------
    tracking_error : float
        The tracking error
    """
    assert benchmark_returns_by_date.index.equals(etf_returns_by_date.index)
    
    
    # difference between the ETF and benchmark returns
    std = etf_returns_by_date.subtract(benchmark_returns_by_date).std(ddof = 1)
    
    # annual tracking error
    tracking_error = np.sqrt(252) * std

    return tracking_error

smart_beta_tracking_error = tracking_error(np.sum(index_weighted_returns, 1), 
                                           np.sum(etf_weighted_returns, 1))

def get_covariance_returns(returns):
    """
    Calculate covariance matrices.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date

    Returns
    -------
    returns_covariance  : 2 dimensional Ndarray
        The covariance of the returns
    """
    
    # calculating covariance
    returns_covariance = np.cov(returns.fillna(0), rowvar=False)
    
    return returns_covariance

covariance_returns = get_covariance_returns(returns)
covariance_returns = pd.DataFrame(covariance_returns, 
                                  returns.columns, 
                                  returns.columns)

covariance_returns_correlation = np.linalg.pinv(np.diag(np.sqrt(np.diag(covariance_returns))))
covariance_returns_correlation = pd.DataFrame(
    covariance_returns_correlation.dot(covariance_returns).dot(covariance_returns_correlation),
    covariance_returns.index,
    covariance_returns.columns)

def get_optimal_weights(covariance_returns, index_weights, scale=2.0):
#    covariance_returns = covariance_returns.values
#    scale=2.0
#    index_weights = index_weights.iloc[-1]
    """
    Find the optimal weights.

    Parameters
    ----------
    covariance_returns : 2 dimensional Ndarray
        The covariance of the returns
    index_weights : Pandas Series
        Index weights for all tickers at a period in time
    scale : int
        The penalty factor for weights the deviate from the index 
    Returns
    -------
    x : 1 dimensional Ndarray
        The solution for x
    """
    x_values = np.zeros(len(index_weights))
    try:
        valid_stocks = np.where(~(index_weights.isnull()))[0]
    except AttributeError:
        valid_stocks = np.where(~(np.isnan(index_weights)))[0]        
    covariance_returns = covariance_returns[np.ix_(valid_stocks,valid_stocks)]
    index_weights = index_weights[valid_stocks]
    assert len(covariance_returns.shape) == 2
    assert len(index_weights.shape) == 1
    assert (covariance_returns.shape[0]
            == covariance_returns.shape[1]
            == index_weights.shape[0])
    
    m = covariance_returns.shape[0]
    
    # x variables (to be found with optimization)
    x = cvx.Variable(m)
    
    #portfolio variance, in quadratic form
    portfolio_variance = cvx.quad_form(x,covariance_returns)
    
    distance_to_index = cvx.norm(x - index_weights)
    objective = cvx.Minimize(portfolio_variance + scale * distance_to_index)
    
    #constraints
    constraints =  [x >= 0, sum(x) == 1]

    #use cvxpy to solve the objective
    problem = cvx.Problem(objective, constraints)
    min_value = problem.solve()
    
    #retrieve the weights of the optimized portfolio
    x_values[valid_stocks] = x.value
    
    return x_values

raw_optimal_single_rebalance_etf_weights = get_optimal_weights(covariance_returns.values, index_weights.iloc[-1],1)
optimal_single_rebalance_etf_weights = pd.DataFrame(
    np.tile(raw_optimal_single_rebalance_etf_weights, (len(etf_weighted_returns.index), 1)),
    etf_weighted_returns.index,
    etf_weighted_returns.columns)

optim_etf_returns = generate_weighted_returns(returns.reindex(etf_weighted_returns.index), 
                                              optimal_single_rebalance_etf_weights)
optim_etf_cumulative_returns = calculate_cumulative_returns(optim_etf_returns)

plt.plot(index_weighted_cumulative_returns, label = 'Index')
plt.plot(etf_weighted_cumulative_returns, label = 'ETF')
plt.legend()
plt.title('Optimized ETF vs Index')
plt.show()


optim_etf_tracking_error = tracking_error(np.sum(index_weighted_returns, 1), np.sum(optim_etf_returns, 1))
print('Optimized ETF Tracking Error: {}'.format(optim_etf_tracking_error))


def rebalance_portfolio(returns, index_weights, shift_size, chunk_size):
    """
    Get weights for each rebalancing of the portfolio.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    index_weights : DataFrame
        Index weight for each ticker and date
    shift_size : int
        The number of days between each rebalance
    chunk_size : int
        The number of days to look in the past for rebalancing

    Returns
    -------
    all_rebalance_weights  : list of Ndarrays
        The ETF weights for each point they are rebalanced
    """
    assert returns.index.equals(index_weights.index)
    assert returns.columns.equals(index_weights.columns)
    assert shift_size > 0
    assert chunk_size >= 0

    #Use ndarray for quick indexing 
    ##see %timeit npreturns[size-chunk_size:size]
    ##see %timeit returns.iloc[size-chunk_size:size]
    
    npreturns = returns.values
    npindex_weights = index_weights.values
    
    #assign list for output, set to zeros for first chunk_size days
    all_rebalance_weights = [np.zeros(returns.shape[1]) 
                             for j in range(chunk_size)] 
    
    for k in range(chunk_size, len(returns)):
        if k % shift_size == 0:
##Code that will allow previous functions to work
#            returns_chunk = returns.iloc[k-chunk_size:k]
#            covariance_chunk = get_covariance_returns(returns_chunk)
#            optimal_w = get_optimal_weights(covariance_chunk, 
#                                            index_weights.iloc[k-1])
##Code using ndarrays
            returns_chunk = npreturns[k-chunk_size:k]
            #remove nans from returns chunk:
            covariance_chunk = np.cov(np.nan_to_num(returns_chunk), 
                                      rowvar=False)
            optimal_w = get_optimal_weights(covariance_chunk, 
                                            npindex_weights[k-1,:])
            all_rebalance_weights.append(optimal_w)
        else:
            all_rebalance_weights.append(all_rebalance_weights[-1])
    return all_rebalance_weights


chunk_size = 250
shift_size = 5
all_rebalance_weights = rebalance_portfolio(returns,
                                            index_weights, 
                                            shift_size, 
                                            chunk_size)

min_var_rolling_weights = pd.DataFrame(all_rebalance_weights,
                                       index = returns.index,
                                       columns = returns.columns)

min_var_returns = generate_weighted_returns(returns, 
                                            min_var_rolling_weights)
min_var_cumulative_returns = calculate_cumulative_returns(min_var_returns)

plt.plot(index_weighted_cumulative_returns, label = 'Index')
plt.plot(min_var_cumulative_returns, label = 'ETF')
plt.legend()
plt.title('Min Var vs Index')
plt.show()
