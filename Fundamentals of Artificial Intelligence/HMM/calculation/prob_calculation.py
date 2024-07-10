import numpy as np


def weibull_probability(a, b, k, lambda_):
    """
    Calculate the probability of a Weibull-distributed random variable falling within a given interval.

    Parameters:
    a (float): Lower bound of the interval.
    b (float): Upper bound of the interval.
    k (float): Shape parameter of the Weibull distribution.
    lambda_ (float): Scale parameter of the Weibull distribution.

    Returns:
    float: Probability of the variable falling within the interval [a, b].
    """
    F_a = 1 - np.exp(-((a / lambda_) ** k))
    F_b = 1 - np.exp(-((b / lambda_) ** k))
    return F_b - F_a


def cal_prob(low_interval, high_interval, matric):
    """
    Calculate the probability of 3 different matrics in low or high interval based on Weibull distribution.

    Parameters:
    low_interval (list): Lower bound and upper bound of the low interval.
    high_interval (list): Lower bound and upper bound of the high interval.
    matric (dictionary): given the parameter of Weibull distribution in different state

    Returns:
    float,float,float,float,float,float: Probabilities of the variable falling within the low and high intervals regarding the matric
    """
    prob_flapping_l = weibull_probability(
        low_interval[0],
        low_interval[1],
        matric["shape_flapping"],
        matric["scale_flapping"],
    )
    prob_soaring_l = weibull_probability(
        low_interval[0],
        low_interval[1],
        matric["shape_soaring"],
        matric["scale_soaring"],
    )
    prob_water_l = weibull_probability(
        low_interval[0], low_interval[1], matric["shape_water"], matric["scale_water"]
    )
    prob_flapping_h = weibull_probability(
        high_interval[0],
        high_interval[1],
        matric["shape_flapping"],
        matric["scale_flapping"],
    )
    prob_soaring_h = weibull_probability(
        high_interval[0],
        high_interval[1],
        matric["shape_soaring"],
        matric["scale_soaring"],
    )
    prob_water_h = weibull_probability(
        high_interval[0], high_interval[1], matric["shape_water"], matric["scale_water"]
    )
    return (
        prob_flapping_l,
        prob_soaring_l,
        prob_water_l,
        prob_flapping_h,
        prob_soaring_h,
        prob_water_h,
    )
