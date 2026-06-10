import numpy as np
import pandas as pd
from typing import Sequence


def result_processing(log_file: str, quantile_list: list):
    """
    Process result
    :param log_file: The log file path
    :param quantile_list
    """
    data = pd.read_csv(log_file)
    pnl = data[['episode', 'step_pnl']].groupby('episode').aggregate('sum').values

    gamma_hedge_ratio = ((data['gamma_after_hedge'] - data['gamma_before_hedge'])
                         / data['gamma_before_hedge']).abs().mean()
    vega_hedge_ratio = ((data['vega_after_hedge'] - data['vega_before_hedge']) / data['vega_before_hedge']).abs().mean()

    result = {'mean': pnl.mean(),
              'std': pnl.std(),
              'Mean-SD': pnl.mean() - 1.645 * pnl.std()}

    for q in quantile_list:
        var = np.quantile(pnl, 1 - q)
        result[f'var{q * 100:.0f}'] = var
        result[f'cvar{q * 100:.0f}'] = pnl[pnl < var].mean()

    result['gamma hedge ratio'] = gamma_hedge_ratio
    result['vega hedge ratio'] = vega_hedge_ratio

    return pd.DataFrame([result]), data, pnl


class HedgingDatesGenerator:
    """
    Generate hedging dates, Monthly or daily hedging
    """
    def __init__(self, issue_date: str, valuation_dates: Sequence):
        """
        :param valuation_dates: The list of valuation dates
        """
        self._issue_date = pd.Timestamp(issue_date)
        self._valuation_dates = [pd.Timestamp(d) for d in valuation_dates]
        self._dates = None
        self._max_i = None
        self._i = None

    def days_before_coupon_hedge(self, days_before_coupon: int = 1):
        """
        Hedging before coupon date
        :param days_before_coupon: Hedging date is how many days before the coupon date
               1 means one day before the coupon date.
        :return:
        """
        if days_before_coupon < 1:
            raise ValueError('days_before_coupon should be a positive number!')

        self._dates = [d - pd.offsets.BusinessDay(days_before_coupon) for d in self._valuation_dates]

        return self._hedging_date_generator()

    def fixed_date_every_month_hedge(self, fixed_date: int = 1):
        """
        Hedging on a fixed date every month
        :param fixed_date: 1 means hedging on the first of every month
        :return:
        """
        if fixed_date < 1 or fixed_date > 31:
            raise ValueError('fixed_date should between 1 and 31!')

        dates = pd.date_range(self._issue_date, self._valuation_dates[-1])
        self._dates = []
        for date in dates:
            if date.day == fixed_date:
                # Adjust the weekend to near business day
                self._dates.append(date if date.day_of_week not in [5, 6] else date + pd.offsets.BusinessDay(1))

        return self._hedging_date_generator()

    def fixed_interval_hedge(self, interval_days: int = 1, jump_weekend: bool = True):
        """
        Hedging on a fixed interval
        :param interval_days: 1 means hedging every business date
        :param jump_weekend: Whether skip weekend
        :return:
        """
        if interval_days < 1:
            raise ValueError('interval_days should be a positive number!')

        self._dates = []
        date = self._issue_date

        while date < self._valuation_dates[-1]:
            if jump_weekend:
                date += pd.offsets.BusinessDay(interval_days)
            else:
                date += pd.Timedelta(days=interval_days)
            self._dates.append(date)

        return self._hedging_date_generator()

    def _hedging_date_generator(self) -> pd.Timestamp:
        """
        Return the next hedging date
        :return:
        """
        self._max_i = len(self._dates)
        self._i = 0

        while True:
            date = self._dates[self._i] if self._i < self._max_i else self._valuation_dates[-1]
            self._i += 1
            yield date

    def reset(self):
        """
        Reset the index to 0
        :return:
        """
        self._i = 0


def generate_random_cash(vol: float = 1) -> float:
    """
    Generate a random cashflow with normal distribution
    :param vol: The standard deviation
    :return
    """

    return np.random.normal(scale=vol)
