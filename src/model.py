from cmath import log
import os
import pickle as pk
from config import cfg
import GPy
from loguru import logger


class Model:

    def __init__(self, position, rss, num_outputs):
        self.num_outputs = num_outputs
        self._kernal(3)
        # Gaussian process regression model
        self.m = GPy.models.GPCoregionalizedRegression(
            [position], [rss], kernel=self.kernel)
        self.m = self._dump_and_load_model(self.m)
        logger.info(f"Done Regression\n{self.m}")

    def _kernal(self, dim):
        """Gaussian process regression kernel
        """
        if dim == 3:
            # kernel = RBF * matern52
            self.kernel = GPy.util.multioutput.ICM(
                input_dim=3, num_outputs=self.num_outputs, kernel=GPy.kern.Matern52(3))
            logger.info("This is the kernel for TUT dataset")

        elif dim == 2:
            self.kernel = GPy.util.multioutput.LCM(
                input_dim=2, num_outputs=520, kernels_list=[GPy.kern.Matern52(2)])
            logger.info("This is the kernel for UJI dataset")
            
        logger.info(f"Done Kernel\n{self.kernel}")

    def _dump_and_load_model(self, m):
        """If the model file exists, load the model.
        If the model file does not exist, train the model.
        
        Store the Gaussian Process model to the model file. Reduce the training time.

        Args:
            m (model): the train model

        Returns:
            model: model from file or new model trained.
        """
        filename = cfg["file"]["model_file"]
        if os.path.exists(filename):
            logger.info(f"Loading model from {filename}")
            with open(filename, "rb") as _f:
                return pk.load(_f)
        else:
            logger.info(f"Dumping model to {filename}")
            with open(filename, "wb") as _f:
                m.optimize(messages=True)
                m.optimize_restarts(num_restarts=10)
                pk.dump(m, _f)
                return m

    def rss_pred(self, position):
        """Use Gaussian process regression model to predict the z value.

        Args:
            position (numpy array): 2d numpy array, LONGITUDE and LATITUDE.

        Returns:
            numpy array: the predicted z value.
        """
        Y_metadata = {"output_index": position[:, -1].astype(int)}
        rss_pred_raw = self.m.predict(position, Y_metadata=Y_metadata)
        z_pred = rss_pred_raw[0]
        logger.info(f"Done Prediction\n{z_pred.shape}")
        return z_pred
