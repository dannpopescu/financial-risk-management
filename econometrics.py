import pandas as pd
import numpy as np


def log_return(prices: pd.Series) -> pd.Series:
    log_ret = np.log(prices / prices.shift(1))
    log_ret.name = 'log_ret'
    return log_ret


def drop_consecutive_duplicates(df: pd.DataFrame, prices_column: str) -> pd.DataFrame:
    not_duplicate = df[prices_column] != df[prices_column].shift(1)
    return df.loc[not_duplicate]


def risk_metrics_variance(returns: pd.Series) -> pd.Series:
    variance = pd.Series(index=returns.index, dtype='float64', name='RM Variance')

    if np.isnan(returns[0]):
        variance[0] = np.nan
        variance[1] = returns.var()
        start_idx = 2  # starting index for computing volatility
    else:
        variance[0] = returns.var()
        start_idx = 1

    for i in range(start_idx, len(returns)):
        variance[i] = 0.94 * variance[i - 1] + 0.06 * returns[i - 1] ** 2

    return variance


def var_historical_simulation(rets, start_date, end_date, m, confidence=0.99):
    """
    Compute Value-at-Risk using the Historical Simulation approach

    Parameters
    ----------
    rets : pd.Series
        Series of returns indexed by datetime sorted in ascending order
    start_date : str
        Start date of the period for which to compute VaR
    end_date : str
        End date of the period for which to compute VaR (inclusive)
    m : int
        Number of past observations used in historical simulation
    confidence : float
        Level of confidence (default: 0.99)

    Returns
    -------
    var : pd.Series
        1-day Value-at-Risk for the specified period
    """
    var = pd.Series(dtype='float64', name='VaR HS')

    for day in rets[start_date: end_date].index:
        day_idx = rets.index.get_loc(day)
        day_var = -rets.iloc[day_idx - m: day_idx].quantile(1 - confidence)
        var[day] = day_var
    return var


def var_weighted_historical_simulation(rets, start_date, end_date, m, weights, confidence=0.99):
    """
    Compute Value-at-Risk using the Weighted Historical Simulation approach
    Parameters
    ----------
    rets : pd.Series
        Series of returns indexed by datetime sorted in ascending order
    start_date : str
        Start date of the period for which to compute VaR
    end_date : str
        End date of the period for which to compute VaR (inclusive)
    m : int
        Number of past observations used in historical simulation
    weights : pd.Series
        Series of weights sorted in ascending order
    confidence : float
        Level of confidence (default: 0.99)

    Returns
    -------
    var : pd.Series
        Series of 1-day Value-at-Risk for the specified period
    """
    var = pd.Series(dtype='float64', name='VaR WHS')

    for day in rets[start_date: end_date].index:
        day_idx = rets.index.get_loc(day)
        rets_and_weights = pd.concat([rets.iloc[day_idx - m: day_idx].reset_index(drop=True), weights], axis=1)
        rets_and_weights.columns = ['ret', 'weight']
        rets_and_weights.sort_values(by='ret', inplace=True, ignore_index=True)

        cum_weights = rets_and_weights['weight'].cumsum()
        for i in range(len(rets_and_weights)):
            if cum_weights[i] >= (1 - confidence):
                var[day] = -rets_and_weights.loc[i, 'ret']
                break
    return var
