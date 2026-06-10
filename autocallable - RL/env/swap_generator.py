from typing import Union
import pandas as pd
from inception.instruments.synthetic_swap import SwapDatabaseReader, SyntheticSwap
from inception.instruments.stock import Stock


def generate_synthetic_swap(hedge_ratio: float,
                            rho_amount: float,
                            stock: Stock,
                            swap_data_reader: SwapDatabaseReader) -> list:
    """
    Generate option that can hedge input delta
    :param hedge_ratio: The ratio of hedge, from 0 to 1
    :param rho_amount: The Rho amount we want to hedge
    :param stock: Stock object
    :param swap_data_reader: The object which stores the swap numerical data
    """

    if hedge_ratio == 0:
        return []

    hedge_rho = rho_amount * hedge_ratio
    swap_data = swap_data_reader.get(stock.date)
    swap = SyntheticSwap(stock.date, swap_data)
    stock.attach(swap)
    hedge_position = -1 * hedge_rho / swap.query('rho')
    swap.modify_volume(hedge_position)

    return [swap]
