import os
import pickle as pk
from config import cfg
import GPy
from loguru import logger


class Model:

    def __init__(self, xy, z):
        self._kernal()
        # Gaussian process regression model
        self.m = GPy.models.GPCoregionalizedRegression(
            [xy], [z], kernel=self.kernel)
        self.m = self._dump_and_load_model(self.m)
        logger.info(f"Done Regression\n{self.m}")

    def _kernal(self):
        """Gaussian process regression kernel
        """
        # kernel = RBF * matern52
        self.kernel = GPy.util.multioutput.LCM(
            input_dim=2, num_outputs=520, kernels_list=[GPy.kern.Matern52(2)])
        logger.info(f"Done Kernel\n{self.kernel}")

    def _dump_and_load_model(self, m):
        """If the model file exists, load the model.
        If the model file does not exist, train the model.

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

    def z_pred(self, xy_pred):
        Y_metadata = {"output_index": xy_pred[:, -1].astype(int)}
        z_pred_raw = self.m.predict(xy_pred, Y_metadata=Y_metadata)
        z_pred = z_pred_raw[0]
        logger.info(f"Done Prediction\n{z_pred.shape}")
        return z_pred
