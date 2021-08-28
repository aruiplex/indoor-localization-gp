import numpy as np
import matplotlib.pyplot as plt
import GPy


def gp_ttl():
    X = np.random.uniform(-3., 3., (20, 1))
    Y = np.sin(X) + np.random.randn(20, 1)*0.05
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X, Y, kernel)
    m.optimize(messages=True)

    m.plot()
    print(
        m.predict(
            Xnew=np.array(
                [
                    [1.],
                ]
            )
        )
    )
    plt.show()
    # GPy.plotting.show(fig, filename='basic_gp_regression_notebook')


def draw_gaussian_dis():
    mu, sigma = 0, 0.1  # mean and standard deviation
    s = np.random.normal(mu, sigma, 30)
    count, bins, ignored = plt.hist(s, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu)**2 / (2 * sigma**2)),
             linewidth=4, color='r')
    plt.show()


def draw_2d():
    sigma = [[1, 0.8], [0.8, 1]]
    mu = [0, 0]
    x1, x2 = np.random.multivariate_normal(mu, sigma, (30, 30)).T
    plt.plot(x1, x2, "x")
    plt.axis("equal")
    plt.show()


def kernal(a, b):
    sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1)-2*np.dot(a, b.T)
    return np.exp(-.5*sqdist)


def pf():
    n = 500
    Xtest = np.linspace(-5, 5, n).reshape(-1, 1)
    K_ = kernal(Xtest, Xtest)

    L = np.linalg.cholesky(K_+1e-6*np.eye(n))
    f_prior = np.dot(L, np.random.normal(size=(n, 100)))

    plt.plot(Xtest, f_prior)
    plt.show()


# draw_2d()
gp_ttl()
