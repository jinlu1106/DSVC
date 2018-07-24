# -*-coding:utf-8-*-
from DSVC.classifiers.logistic_regression import *


class SubLogisticRegression(LogisticRegression):
    def __init__(self):
        super(SubLogisticRegression, self).__init__()
        self.wi = None

    # 多分类器
    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True):

        num_train, dim = X.shape
        loss_history = {}
        if self.wi is None:
            self.wi = np.random.randn(dim, 10)

        # 训练10个二分类器
        for i in xrange(10):

            loss_history[i] = []
            if self.w is None:
                self.w = 0.001 * np.random.randn(dim)

            # 如果 y 中值与所要识别的数字一致, 则y_batch 中标签值为1，否则为0
            y_train = []
            for label in y:
                if label == i:
                    y_train.append(1)
                else:
                    y_train.append(0)

            y_train = np.array(y_train)

            loss_history[i] = self.train(X, y_train, learning_rate, num_iters, batch_size)

            self.wi[:, i] = self.w
            self.w = None

        return loss_history

    # 预测函数
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        z_num = X.dot(self.wi)
        h_theta = (1. / (1. + np.exp(-z_num)))

        for i in xrange(h_theta.shape[0]):
            y_pred[i] = np.argmax(h_theta[i])

        return y_pred