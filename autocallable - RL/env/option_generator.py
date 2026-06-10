"""
This file defines functions that simulate client option behavior, and function that create hedge options
"""
import numpy as np

from inception.instruments.stock import Stock
from inception.instruments.option_interpolation import OptionNumerical
import pickle
import os


def load_option_value_database(risk_free_rate: float,
                               file_dir: str = 'C:/Users/TY482JM/FreemanWorkplace/rl-hedging/database') -> dict:
    """
    Loading the option value and Greeks database
    :param risk_free_rate: Risk-free rate
    :param file_dir: File directory
    :return:
    """
    data_dict = dict()
    for path, directories, files in os.walk(file_dir):
        for file in files:
            file = os.path.join(file_dir, file)
            with open(file, 'rb') as f:
                option_db = pickle.load(f)
                if option_db['risk free rate'] != risk_free_rate:
                    continue

                data_dict[option_db['option style']] = option_db['data']

    return data_dict


def helper_generate_poisson_position(beta: list, poisson_rate: int) -> list:
    """
    Total position comes with poisson distribution
    :param beta: probability of each option
    :param poisson_rate: Poisson rate
    """
    probability = [b / sum(beta) for b in beta]
    num_of_style = len(beta)

    total_position = np.random.poisson(poisson_rate)
    style_numbers = np.random.choice(range(num_of_style), total_position, p=probability)
    # Aggregate each option
    style_position = [sum(style_numbers == i) for i in range(num_of_style)]
    
    # Both of them have an equal probability of being long or short
    position = []
    for pos in style_position:
        buy_sell_pos = np.random.choice([-1.0, 1.0], pos)
        position.append([sum(buy_sell_pos == 1), sum(buy_sell_pos == -1) * -1])
         
    return position


def generate_poisson_arrive_options(styles: list,
                                    beta: list,
                                    poisson_rate: int,
                                    stock: Stock,
                                    risk_free_rate: float,
                                    option_interpolator: list,
                                    tenor: int = 30,
                                    contract_size: int = 100,
                                    strike_constraint_function=lambda stock_price: stock_price
                                    ):
    """
    Generate Poisson arrival options
    :param styles: The option style: 'European Call', 'European Put', 'Binary Call', 'Binary Put', 'American Call',
                  'American Put', 'European Call UofT'....
    :param beta: The probability of option style, if style = ['American Call', 'American put'], beta = [1, 1] meaning
                 50% of probability the option will be American Call, and the 50% of probability the option will be
                 American Put.
    :param poisson_rate: Poisson rate
    :param stock: Stock object
    :param option_interpolator: The interpolator contains option value and Greeks
    :param risk_free_rate: The risk-free rate
    :param tenor: Option tenor
    :param contract_size: The contract size
    :param strike_constraint_function: The constraint function of option strike
    """
    if len(styles) != len(beta) or len(styles) != len(option_interpolator):
        raise ValueError("The dimension of style, alpha and option_interpolator should be equal!")

    i = 0
    while True:
        if i == 0:
            # Step 0 - always underwrite one option
            new_position = [[0, -1]] + [[0, 0] for _ in styles[1:]]
        else:
            # The New client option comes with poisson distribution and an equal probability of being long or short
            new_position = helper_generate_poisson_position(beta, poisson_rate)

        # At The Money option
        options = []
        # For the first and second option
        for i, pos_list in enumerate(new_position):
            # For the long and short option
            for pos in pos_list:
                if pos == 0:
                    continue

                # We can change the Option to any other style: American option, Digital option, ...
                option = OptionNumerical(direction=styles[i],
                                         evaluate_date=stock.date,
                                         exercise_date=tenor,
                                         stock_price=stock.price,
                                         strike=strike_constraint_function(stock.price),
                                         vol=stock.vol,
                                         free_rate=risk_free_rate,
                                         volume=pos * contract_size,
                                         option_interpolator=option_interpolator[i])
                # link stock and option
                stock.attach(option)
                options.append(option)

        # Increase step marker
        i += 1

        yield options


def generate_gamma_vega_hedge_options(hedge_ratio: float,
                                      gamma_amount: float,
                                      vega_amount: float,
                                      styles: list,
                                      weights: list,
                                      stock: Stock,
                                      risk_free_rate: float,
                                      option_interpolator: list,
                                      tenor: int = 30,
                                      transaction_cost: float = 0.005,
                                      is_rl_env: bool = False,
                                      strike_constraint_function=lambda stock_price: stock_price
                                      ) -> list:
    """
    Generate option that can hedge input gamma and vega
    :param hedge_ratio: The ratio of hedge, from 0 to 1
    :param gamma_amount: The Gamma amount we want to hedge
    :param vega_amount: The Vega amount we want to hedge
    :param styles:  The option style list: 'European Call', 'European Put', 'Binary Call', 'Binary Put',
          'American Call', 'American Put', 'European Call UofT'....
    :param weights: Proportion of hedging with different types of options, for example,
           if style = ['American Call', 'American Put'], alpha=[1, 1] means using 50% of American Call and
           50% of American Put to do the hedging.
    :param stock: Stock object
    :param risk_free_rate: The risk-free rate
    :param option_interpolator: The interpolator contains option value and Greeks
    :param tenor: Option tenor
    :param transaction_cost: The transaction cost
    :param is_rl_env: Is reinforcement learning environment
    :param strike_constraint_function: The constraint function of option strike
    """

    if hedge_ratio == 0:
        return []

    if len(styles) != len(weights) or len(styles) != len(option_interpolator):
        raise ValueError("The dimension of style, alpha and option_interpolator should be equal!")

    if is_rl_env:
        # Action constraints, UofT code do this.
        # Refer: https://github.com/rotmanfinhub/gamma-vega-rl-hedging/blob/main/environment/Environment.py
        action_low = [0, gamma_amount]
        action_high = [0, gamma_amount]
        low_val = np.min(action_low)
        high_val = np.max(action_high)
        hedge_gamma = low_val + hedge_ratio * (high_val - low_val)
    else:
        hedge_gamma = gamma_amount * hedge_ratio

    # At The Money option, the volume is 1
    options = [
        OptionNumerical(direction=s,
                        evaluate_date=stock.date,
                        exercise_date=tenor,
                        stock_price=stock.price, 
                        strike=strike_constraint_function(stock.price),
                        vol=stock.vol,
                        free_rate=risk_free_rate,
                        volume=1, 
                        transaction_cost=transaction_cost, 
                        option_interpolator=option_interpolator[i])
        for i, s in enumerate(styles)
        ]
    # link stock and option
    [stock.attach(option) for option in options]

    _weights = [a / sum(weights) for a in weights]
    # Assign the hedging gamma to each option
    hedge_positions = [-1 * hedge_gamma * _weights[i] / option.query('gamma')
                       for i, option in enumerate(options)]

    # Modify the option volume, so the total gamma of these options will equal gamma_amount * hedge_ratio
    [option.modify_volume(hedge_positions[i]) for i, option in enumerate(options)]

    return options


def generate_delta_hedge_options(hedge_ratio: float,
                                 delta_amount: float,
                                 styles: list,
                                 weights: list,
                                 stock: Stock,
                                 risk_free_rate: float,
                                 option_interpolator: list,
                                 tenor: int = 30,
                                 transaction_cost: float = 0.005,
                                 strike_constraint_function=lambda stock_price: stock_price
                                 ) -> list:
    """
    Generate option that can hedge input delta
    :param hedge_ratio: The ratio of hedge, from 0 to 1
    :param delta_amount: The Delta amount we want to hedge
    :param styles:  The option style list: 'European Call', 'European Put', 'Binary Call', 'Binary Put',
          'American Call', 'American Put', 'European Call UofT'....
    :param weights: Proportion of hedging with different types of options, for example,
           if style = ['American Call', 'American Put'], alpha=[1, 1] means using 50% of American Call and
           50% of American Put to do the hedging.
    :param stock: Stock object
    :param risk_free_rate: The risk-free rate
    :param option_interpolator: The interpolator contains option value and Greeks
    :param tenor: Option tenor
    :param transaction_cost: The transaction cost
    :param strike_constraint_function: The constraint function of option strike
    """

    if hedge_ratio == 0:
        return []

    if len(styles) != len(weights) or len(styles) != len(option_interpolator):
        raise ValueError("The dimension of style, alpha and option_interpolator should be equal!")

    hedge_delta = delta_amount * hedge_ratio

    # At The Money option, the volume is 1
    options = [
        OptionNumerical(direction=s,
                        evaluate_date=stock.date,
                        exercise_date=tenor,
                        stock_price=stock.price,
                        strike=strike_constraint_function(stock.price),
                        vol=stock.vol,
                        free_rate=risk_free_rate,
                        volume=1,
                        transaction_cost=transaction_cost,
                        option_interpolator=option_interpolator[i])
        for i, s in enumerate(styles)
        ]
    # link stock and option
    [stock.attach(option) for option in options]

    _weights = [a / sum(weights) for a in weights]
    # Assign the hedging gamma to each option
    hedge_positions = [-1 * hedge_delta * _weights[i] / option.query('delta')
                       for i, option in enumerate(options)]

    # Modify the option volume, so the total gamma of these options will equal gamma_amount * hedge_ratio
    [option.modify_volume(hedge_positions[i]) for i, option in enumerate(options)]

    return options


if __name__ == '__main__':
    data = load_option_value_database(0.0)
    print(data.keys())
