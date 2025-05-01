import numpy as np

# mini-batch линейная регрессия для квадратичной ошибки (MSE)
class LinearRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size

        self._coef = None
        self._intercept = None

    def get_penalty_grad(self):
        if self.penalty == "l2":
            return self.alpha * self._coef * 2
        elif self.penalty == "l1":
            return self.alpha * np.sign(self._coef)
        else:
            return 0

    def fit(self, x, y):
        if self.random_state:
            np.random.seed(self.random_state)
        x = np.hstack([np.ones((x.shape[0], 1)), x])

        n_features = x.shape[1]
        self._coef = np.random.normal(scale=0.01, size=n_features)
        if self.early_stopping:
            n_val = int(x.shape[0] * self.validation_fraction)
            x_val, y_val = x[:n_val], y[:n_val]
            x_train, y_train = x[n_val:], y[n_val:]
        else:
            x_train, y_train = x, y

        best_loss = np.inf
        no_improvement = 0

        for epoch in range(self.max_iter):
            if self.shuffle:
                indices = np.random.permutation(len(x_train))
                x_train = x_train[indices]
                y_train = y_train[indices]
            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                pred = x_batch.dot(self._coef)
                error = pred - y_batch

                grad = (x_batch.T.dot(error) / len(y_batch)) + self.get_penalty_grad()

                self._coef -= self.eta0 * grad

                if np.linalg.norm(grad) < self.tol:
                    return

            if self.early_stopping:
                val_loss = np.mean((x_val.dot(self._coef) - y_val)**2)

                if val_loss < best_loss - self.tol:
                    best_loss = val_loss
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= self.n_iter_no_change:
                    break

        self._intercept = self._coef[0]

    def predict(self, x):
        print(self._coef.shape)
        print(x.shape)
        print(self._coef)
        return np.c_[np.ones(x.shape[0]), x].dot(self._coef)

    @property
    def coef_(self):
        return self._coef[1:]

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
