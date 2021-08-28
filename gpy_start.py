import numpy as np
import matplotlib.pyplot as plt
import GPy


def gp_ttl():
    X = np.random.uniform(-3., 3., (20, 1))
    Y = np.sin(X) + np.random.randn(20, 1)*0.05
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(X, Y, kernel)
    m.optimize(messages=False)

    x = [1.]
    y0, y1 = m.predict(
        Xnew=np.array(
            [
                x
            ]
        )
    )
    m.plot()
    plt.plot(x, y0, y1, "r")

    plt.show()
    # GPy.plotting.show(fig, filename='basic_gp_regression_notebook')


if __name__ == "__main__":
    gp_ttl()
