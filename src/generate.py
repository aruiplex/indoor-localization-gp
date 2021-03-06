import random

from loguru import logger

from config import cfg


class Generate:

    def __init__(self, xy):
        """Get the range of longitude and latitude

        Args:
            xy (numpy array): 2d array, longitude and latitude
        """
        self.x_min = xy[:, 0].min()
        self.x_max = xy[:, 0].max()
        self.y_min = xy[:, 1].min()
        self.y_max = xy[:, 1].max()
        if xy.shape[1] == 3:
            self.z = xy[:, 2][0]
        self.n = xy.shape[0]

    def position_pred(self):
        """generate the fake longitude and latitude into python list

        Returns:
            list: the xy_pred
        """
        data_set = cfg["model"]["data_set"]
        l = []
        real_fake_ratio = cfg["generate"]["real_fake_ratio"]
        fake_num = int(real_fake_ratio * self.n)
        logger.info(f"fake_num: {fake_num}")
        # 2d dataset
        if not hasattr(self, "z"):
            for _ in range(fake_num):
                l.append([random.uniform(self.x_min, self.x_max),
                          random.uniform(self.y_min, self.y_max), 
                          data_set])

        # 3d dataset
        else:
            for _ in range(fake_num):
                l.append([random.uniform(self.x_min, self.x_max),
                          random.uniform(self.y_min, self.y_max),
                          self.z,
                          data_set])
        return l
