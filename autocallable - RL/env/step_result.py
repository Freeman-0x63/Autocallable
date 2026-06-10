"""
Step result class
"""


import dataclasses


@dataclasses.dataclass
class StepResult:
    """Logging step metrics for analysis
    """
    episode: int = 0
    t: int = 0
    hed_action: float = 0.
    hed_share: float = 0.
    stock_price: float = 0.
    stock_position: float = 0.
    stock_pnl: float = 0.
    liab_port_delta: float = 0.
    liab_port_gamma: float = 0.
    liab_port_vega: float = 0.
    liab_port_pnl: float = 0.
    liab_port_cash: float = 0.
    liab_port_final_exercise_cash: float = 0.
    hed_cost: float = 0.
    hed_port_delta: float = 0.
    hed_port_gamma: float = 0.
    hed_port_vega: float = 0.
    hed_port_pnl: float = 0.
    hed_port_cash: float = 0.
    hed_port_final_exercise_cash: float = 0.
    delta_before_hedge: float = 0.
    delta_after_hedge: float = 0.
    gamma_before_hedge: float = 0.
    gamma_after_hedge: float = 0.
    vega_before_hedge: float = 0.
    vega_after_hedge: float = 0.
    step_pnl: float = 0.
    transaction_fee: float = 0.
    state_price: float = 0.
    state_gamma: float = 0.
    state_vega: float = 0.
    state_hed_gamma: float = 0.
    state_hed_vega: float = 0.
    daily_cash: float = 0.
    total_cash: float = 0.
    final_exercise_cash: float = 0.
    hedge_option_dollar_amount: float = 0.
    days_before_call: float = 0.
