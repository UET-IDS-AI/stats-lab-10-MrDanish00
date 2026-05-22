import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(
    x,
    y,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6
):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).
    """

    q = (
        ((x - mu_x) ** 2) / (sigma_x ** 2)
        - (
            2
            * rho
            * (x - mu_x)
            * (y - mu_y)
        ) / (sigma_x * sigma_y)
        + ((y - mu_y) ** 2) / (sigma_y ** 2)
    )

    coefficient = (
        1 /
        (
            2
            * np.pi
            * sigma_x
            * sigma_y
            * np.sqrt(1 - rho ** 2)
        )
    )

    exponent = np.exp(
        -q / (2 * (1 - rho ** 2))
    )

    return coefficient * exponent


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    """

    coefficient = 1 / (
        np.sqrt(2 * np.pi) * sigma_x
    )

    exponent = np.exp(
        -((x - mu_x) ** 2) /
        (2 * sigma_x ** 2)
    )

    return coefficient * exponent


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    """

    coefficient = 1 / (
        np.sqrt(2 * np.pi) * sigma_y
    )

    exponent = np.exp(
        -((y - mu_y) ** 2) /
        (2 * sigma_y ** 2)
    )

    return coefficient * exponent


def covariance_matrix(
    sigma_x=2,
    sigma_y=3,
    rho=0.6
):
    """
    Return covariance matrix.
    """

    covariance = rho * sigma_x * sigma_y

    return np.array([
        [sigma_x ** 2, covariance],
        [covariance, sigma_y ** 2]
    ])


def joint_pdf_grid_integral(
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    n=250
):
    """
    Numerically approximate integral of joint Gaussian PDF.
    """

    x_values = np.linspace(
        mu_x - 4 * sigma_x,
        mu_x + 4 * sigma_x,
        n
    )

    y_values = np.linspace(
        mu_y - 4 * sigma_y,
        mu_y + 4 * sigma_y,
        n
    )

    dx = x_values[1] - x_values[0]
    dy = y_values[1] - y_values[0]

    total = 0.0

    for x in x_values:
        for y in y_values:

            total += (
                joint_gaussian_pdf(
                    x,
                    y,
                    mu_x,
                    mu_y,
                    sigma_x,
                    sigma_y,
                    rho
                )
                * dx
                * dy
            )

    return total


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    """
    Generate n samples from a jointly Gaussian distribution.
    """

    np.random.seed(seed)

    mean = [mu_x, mu_y]

    cov = covariance_matrix(
        sigma_x,
        sigma_y,
        rho
    )

    samples = np.random.multivariate_normal(
        mean,
        cov,
        size=n
    )

    x_samples = samples[:, 0]
    y_samples = samples[:, 1]

    return x_samples, y_samples


def sample_means(x_samples, y_samples):
    """
    Return sample means of X and Y.
    """

    mx = np.mean(x_samples)
    my = np.mean(y_samples)

    return mx, my


def sample_covariance_matrix(
    x_samples,
    y_samples
):
    """
    Return 2 by 2 sample covariance matrix.
    """

    return np.cov(
        x_samples,
        y_samples,
        ddof=1
    )


def sample_correlation(
    x_samples,
    y_samples
):
    """
    Return sample correlation coefficient.
    """

    return np.corrcoef(
        x_samples,
        y_samples
    )[0, 1]


def gaussian_independence_check(rho):
    """
    For jointly Gaussian variables:
    return True if rho is zero.
    """

    return rho == 0


def zero_rho_covariance_check(n=100000):
    """
    Check that covariance is approximately zero
    when rho = 0.
    """

    x, y = generate_joint_gaussian_samples(
        n=n,
        rho=0,
        seed=0
    )

    cov_matrix = sample_covariance_matrix(
        x,
        y
    )

    sample_cov = cov_matrix[0, 1]

    return bool(abs(sample_cov) < 0.05)


def nonzero_rho_covariance_check(n=100000):
    """
    Check that covariance is nonzero when rho = 0.6.
    """

    rho = 0.6
    sigma_x = 2
    sigma_y = 3

    expected_cov = (
        rho
        * sigma_x
        * sigma_y
    )

    x, y = generate_joint_gaussian_samples(
        n=n,
        rho=rho,
        seed=0
    )

    cov_matrix = sample_covariance_matrix(
        x,
        y
    )

    sample_cov = cov_matrix[0, 1]

    return bool(abs(
        sample_cov - expected_cov
    ) < 0.2)
