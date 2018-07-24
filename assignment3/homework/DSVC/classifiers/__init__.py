# -*-coding:utf-8-*-
from DSVC.classifiers.logistic_regression import *


class SubLogisticRegression(LogisticRegression):
    def __init__(self):
        super(SubLogisticRegression, self).__init__()
        self.wi = None

    # 重写父类损失函数
    def loss(self, X_batch, y_batch):

        N = X_batch.shape[0]

        # sigmoid 函数 h_theta = 1 / (1 + e^(-z))
        z = np.dot(X_batch, self.wi)
        h_theta = 1. / (1. + np.exp(-z))

        # 似然函数 l_theta = y * (log(h_theta)) + (1 - y) * (log(1 - h_theta))
        log_h = np.log(h_theta)
        log1_h = np.log(1 - h_theta)

        l_theta = np.sum(y_batch * log_h + (1 - y_batch) * log1_h) / N

        # 损失函数 -- 求解 loss = -y * (log(h_theta)) - (1 - y) * (log(1 - h_theta))
        loss = -l_theta

        # 梯度 gradient -- grad = (y - h_theta) * X.T
        # grad = np.sum(np.dot(X_batch.T, y_batch - h_theta)) / N  # 错误，grad是一个一维向量，不需要求和
        grad = X_batch.T.dot(h_theta - y_batch) / N

        return loss, grad

    # 多分类器
    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True):

        num_train, dim = X.shape

        if self.w is None:
            self.w = np.random.randn(dim, 10)

        loss_history = {}

        # 训练10个二分类器
        for i in xrange(10):

            loss_history[i] = []

            if self.wi is None:
                self.wi = 0.0001 * np.random.randn(dim)

            # 如果 y 中值与所要识别的数字一致, 则y_batch 中标签值为1，否则为0
            # y_batch = np.array(y_batch == i).astype(int)
            for j in xrange(len(y)):
                if y[j] != i:
                    y[j] = 1
                # else:
                #     y[j] = 1

            for it in xrange(num_iters):

                sample_index = np.random.choice(num_train, batch_size, replace=False)
                X_batch = X[sample_index]
                y_batch = y[sample_index]

                loss, grad = self.loss(X_batch, y_batch)
                loss_history[i].append(loss)

                if loss < 0.2:
                    learning_rate = 1e-8

                self.wi -= learning_rate * grad
                self.w[:, i] = self.wi

                if verbose and it % 1000 == 0:
                    print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

            self.wi = None

        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        z_num = X.dot(self.w)
        h_theta = (1. / (1. + np.exp(-z_num)))

        for i in xrange(h_theta.shape[0]):
            y_pred[i] = np.argmax(h_theta[i])

        # y_pred = [np.argmax(h_theta[i]) for i in range(len(h_theta))]

        return y_pred