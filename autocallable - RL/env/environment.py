"""A trading environment for Auto-Callable Notes: Action for Gamma hedging, keeping Delta neutral"""

from typing import Optional, Sequence
import dataclasses

import gym
from gym import spaces
from acme.utils import loggers
import numpy as np

from inception.instruments import FullTensorInterpolator, AutoCallableNoteChebyshev
from inception.instruments import Stock, Portfolio

from env.option_generator import generate_gamma_vega_hedge_options
from env.option_generator import load_option_value_database
from env.step_result import StepResult
from env.utils import HedgingDatesGenerator


class TradingEnv(gym.Env):
    """
    This is the Gamma & Vega Trading Environment.
    """
    def __init__(self,
                 stock_mean: list,
                 stock_vol: list,
                 stock_price: list,
                 stock_name: list,
                 init_date: str,
                 risk_free_rate: float,
                 hedge_option_style: list,
                 alpha: list,
                 hedge_option_tenor: int,
                 valuation_dates: Sequence[str],
                 coupon_payment_dates: Sequence[str],
                 call_dates: Sequence[str],
                 coupon_barrier: float = -0.35,
                 principal_barrier: float = -0.3,
                 call_barrier: float = 0,
                 coupon_amount: float = 0.95,
                 hedging_date: int = 5,
                 stock_model: str = 'GBM',
                 sabr_alpha: float = 0.3,
                 sabr_beta: float = 1,
                 sabr_rho: float = -0.7,
                 transaction_cost: float = 0.005,
                 option_database_dir: str = './database',
                 notes_database_dir: str = './autocallable_database',
                 logger: Optional[loggers.Logger] = None,
                 gamma_multiple: float = 1,
                 is_rl_env: bool = False,
                 seed: int = 0):
        """
        :param stock_mean: The stock model mean
        :param stock_vol: The stock model volatility
        :param stock_price: The stock model initial price
        :param init_date: The fist date
        :param risk_free_rate: The risk-free rate
        :param hedge_option_style: The hedging option style, like: 'European Call', 'European Put', 'Binary Call',
               'Binary Put', 'American Call', 'American Put', 'European Call UofT'....
        :param alpha: Proportion of hedging with different types of options, for example,
               if hedge_option_style = ['American Call', 'American Put'],
               alpha = [1, 1] means using 50% of American Call and 50% of American Put to do the hedging.
        :param hedging_date: It refers to the day on which each month the heading can be done
               And 5 means heading can be done on the fifth day of each month
        :param hedge_option_tenor: The tenor of hedging option, days
        :param valuation_dates: List of observation dates
        :param coupon_payment_dates: List of coupon payment dates
        :param call_dates: List of call dates
        :param coupon_barrier: Coupon barrier level
        :param call_barrier: Auto-call threshold
        :param principal_barrier:
        :param coupon_amount:
        :param stock_model: The stock model: like 'GBM', 'SABR', or 'Heston'
        :param sabr_alpha: The SABR model parameter alpha
        :param sabr_beta: The SABR model parameter beta
        :param sabr_rho: The SABR model parameter rho
        :param transaction_cost: The percentage of transaction cose
        :param option_database_dir: The directory of database containing option value and Greeks
        :param notes_database_dir: The directory of database containing Auto-Callable Notes value and Greeks
        :param logger:
        :param gamma_multiple: A multiple that gamma hedge. Greater than one will no longer be gamma neutral.
        :param is_rl_env:
        :param seed: The random seed
        """
        super(TradingEnv, self).__init__()
        self.logger = logger

        self._stock_mean = stock_mean
        self._stock_vol = stock_vol
        self._stock_price = stock_price
        self._stock_name = stock_name
        self._init_date = init_date
        self._risk_free_rate = risk_free_rate
        self._hedge_option_style = hedge_option_style
        self._hedge_option_tenor = hedge_option_tenor
        self._valuation_dates = valuation_dates
        self._coupon_dates = coupon_payment_dates
        self._call_dates = call_dates
        self._coupon_barrier = coupon_barrier
        self._call_barrier = call_barrier
        self._principal_barrier = principal_barrier
        self._coupon_amount = coupon_amount
        self._transaction_cost = transaction_cost
        self._notes_hedging_dates = HedgingDatesGenerator(init_date, valuation_dates)
        self.hedging_dates_generator = self._notes_hedging_dates.fixed_interval_hedge()
        self._alpha = alpha
        self._is_rl_env = is_rl_env
        self._stock_model = stock_model
        self._sabr_alpha = sabr_alpha
        self._sabr_beta = sabr_beta
        self._sabr_rho = sabr_rho
        self._gamma_multiple = gamma_multiple
        self._seed = seed

        self.portfolio = None
        self.hedge_portfolio = None
        self.client_portfolio = None
        self.stock_A = None
        self.stock_B = None
        self.stock_C = None
        self.stock = None
        self.client_option_generator = None
        self.notes = None
        self.result_log = None

        # Load option database
        self._client_interpolator = FullTensorInterpolator(notes_database_dir, verbose=False)
        # self._option_database = load_option_value_database(risk_free_rate, option_database_dir)
        # self._hedge_option_interpolator = [OptionInterpolator(self._option_database[style])
        #                                    for style in hedge_option_style]

        # track time step within an episode (it's step)
        self.t = None
        self.episode_num = 0

        # Observation space
        obs_low_bound = np.array([np.float32(0), -np.float32("inf"), -np.float32("inf")])
        obs_high_bound = np.array([np.float32("inf"), np.float32("inf"), np.float32("inf")])
        self.observation_space = spaces.Box(low=obs_low_bound, high=obs_high_bound)

        # Action space: HIGH value has to be adjusted with respect to the option used for hedging
        self.action_space = spaces.Box(low=np.array([0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))

        # Initializing the state values
        self.num_state = 3

        # self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> np.array:
        """
        reset function which is used for each episode
        """
        self.t = 0
        self.episode_num += 1
        self._seed += 1
        self.stock_A = Stock(self._stock_price[0], self._stock_mean[0], self._stock_vol[0], self._init_date,
                             jump_weekend=True, name=self._stock_name[0], seed=self._seed)
        self.stock_B = Stock(self._stock_price[1], self._stock_mean[1], self._stock_vol[1], self._init_date,
                             jump_weekend=True, name=self._stock_name[1], seed=self._seed + 1_000_000, correlation=0.96)
        self.stock_C = Stock(self._stock_price[2], self._stock_mean[2], self._stock_vol[2], self._init_date,
                             jump_weekend=True, name=self._stock_name[2], seed=self._seed + 2_000_000, correlation=0.82)
        self.stock = dict(zip(self._stock_name, [self.stock_A, self.stock_B, self.stock_C]))

        self.portfolio = Portfolio(self._init_date)
        self.hedge_portfolio = Portfolio(self._init_date)
        self.client_portfolio = Portfolio(self._init_date)

        self._notes_hedging_dates.reset()
        # Client portfolio with an Auto-Callable Notes
        self.notes = AutoCallableNoteChebyshev(start_date=self._init_date,
                                               evaluate_date=self._init_date,
                                               underlying_names=self._stock_name,
                                               stock_price=self._stock_price,
                                               vol=self._stock_vol,
                                               issue_price=self._stock_price,
                                               note_interpolator=self._client_interpolator,
                                               valuation_dates=self._valuation_dates,
                                               coupon_payment_dates=self._coupon_dates,
                                               call_dates=self._call_dates,
                                               coupon_barrier=self._coupon_barrier,
                                               principal_barrier=self._principal_barrier,
                                               call_barrier=self._call_barrier,
                                               coupon_amount=self._coupon_amount,
                                               free_rate=self._risk_free_rate,
                                               volume=-1)

        [self.stock_A.attach(stock) for stock in [self.stock_B, self.stock_C]]
        [stock.attach(self.notes) for stock in [self.stock_A, self.stock_B, self.stock_C]]
        [stock.attach(self.portfolio) for stock in [self.stock_A, self.stock_B, self.stock_C]]

        self.notes.attach(self.portfolio)
        # self.notes.attach(self.client_portfolio)

        return self.get_states()

    def get_states(self) -> np.array:
        """
        Return the states
        :return:
        """
        states = [self.stock_A.price, self.portfolio.gamma.max[1], self.notes.days_before_call()]

        return np.array(states)

    def _create_step_logger(self,
                            action: float,
                            delta_before_hedge: float,
                            gamma_before_hedge: float,
                            vega_before_hedge: float,
                            hedge_options: list) -> StepResult:
        """
        Crete the step result object
        :param action:
        :param delta_before_hedge:
        :param gamma_before_hedge:
        :param vega_before_hedge:
        :param hedge_options:
        :return:
        """
        result = StepResult(
            episode=self.episode_num,
            t=self.t,
            hed_action=action,
        )
        # Delta, Gamma, and Vega before hedge
        result.delta_before_hedge = delta_before_hedge
        result.gamma_before_hedge = gamma_before_hedge
        result.vega_before_hedge = vega_before_hedge

        # Delta, Gamma, and Vega after hedge
        result.delta_after_hedge = self.portfolio.delta.max[1]
        result.gamma_after_hedge = self.portfolio.gamma.max[1]
        result.vega_after_hedge = self.portfolio.vega

        result.stock_price = self.notes.stock_price
        result.hed_cost = self.hedge_portfolio.transaction_fee
        result.stock_position = [self.stock[s].net_position for s in self.stock]

        result.liab_port_delta = self.client_portfolio.delta.max[1]
        result.liab_port_gamma = self.client_portfolio.gamma.max[1]
        result.liab_port_vega = self.client_portfolio.vega
        result.liab_port_cash = self.client_portfolio.cash

        result.hed_port_delta = self.hedge_portfolio.delta.max[1]
        result.hed_port_gamma = self.hedge_portfolio.gamma.max[1]
        result.hed_port_vega = self.hedge_portfolio.vega
        result.hed_port_cash = self.hedge_portfolio.cash

        result.daily_cash = self.portfolio.daily_cash
        result.total_cash = self.portfolio.cash
        result.transaction_fee = self.portfolio.transaction_fee
        result.hedge_option_dollar_amount = float(sum([option.value for option in hedge_options]))

        return result

    def _update_step_logger(self,
                            result: StepResult,
                            reward: float,
                            hedge_cost: float,
                            states: np.array,
                            done: bool):
        """
        Update the step log object
        :param result: StepResult object
        :param reward: Step reward
        :param hedge_cost:
        :param states: Environment state
        :param done: Flag indicates whether it is the last step
        :return:
        """
        result.step_pnl = reward
        result.liab_port_pnl = self.client_portfolio.pnl
        result.hed_port_pnl = self.hedge_portfolio.pnl
        result.stock_pnl = [self.stock[s].pnl for s in self.stock]
        result.hed_cost = hedge_cost
        result.state_price, result.state_gamma, result.days_before_call = states[:3]

        if done:
            result.hed_port_final_exercise_cash = self.hedge_portfolio.intrinsic_value
            result.liab_port_final_exercise_cash = self.client_portfolio.intrinsic_value
            result.final_exercise_cash = self.portfolio.intrinsic_value

    def step(self, action):
        """
        profit and loss period reward
        """
        # Delta, Gamma, and Vega before hedge
        portfolio_delta = self.portfolio.delta
        portfolio_gamma = self.portfolio.gamma
        portfolio_vega = self.portfolio.vega

        # # Gamma hedge; Create hedge options
        # hedge_options = generate_gamma_vega_hedge_options(action[0],
        #                                                   portfolio_gamma * self._gamma_multiple,
        #                                                   portfolio_vega,
        #                                                   self._hedge_option_style,
        #                                                   self._alpha,
        #                                                   self.stock,
        #                                                   self._risk_free_rate,
        #                                                   self._hedge_option_interpolator,
        #                                                   self._hedge_option_tenor,
        #                                                   self._transaction_cost,
        #                                                   self._is_rl_env,
        #                                                   self._strike_constraint_function)
        # Link the portfolio
        hedge_options = []
        [option.attach(self.portfolio) for option in hedge_options]
        # Link the hedge portfolio
        [option.attach(self.hedge_portfolio) for option in hedge_options]
        hedge_cost = sum([option.transaction_fee for option in hedge_options])

        # Using the stock to hedge the rest Delta
        [self.stock[s].add_position(-self.portfolio.delta[s]) for s in self.stock]
        hedge_cost += -1 * sum([self.portfolio.delta[s] * self.stock[s].price * self.stock[s].transaction_cost
                                for s in self.stock])

        # Record Delta, Gamma, and Vega after hedge
        self.result_log = self._create_step_logger(action[0],
                                                   portfolio_delta,
                                                   portfolio_gamma,
                                                   portfolio_vega,
                                                   hedge_options)

        # Update stock price
        reward = 0
        hedge_date = next(self.hedging_dates_generator)
        while self.stock_A.date <= hedge_date:
            self.stock_A.update()
            reward += self.portfolio.daily_pnl
            if self.notes.is_expired:
                break

        self.t = self.t + 1

        # The reward is the next step PnL with current position
        reward += hedge_cost
        states = self.get_states()

        if self.notes.is_expired:
            done = True
            states[1:] = 0
            [self.stock[s].unwind_position() for s in self.stock]
        else:
            done = False

        self._update_step_logger(self.result_log, reward, hedge_cost, states, done)

        # for other info later
        info = {"path_row": self.episode_num}
        if self.logger:
            self.logger.write(dataclasses.asdict(self.result_log))

        return states, reward, done, info
