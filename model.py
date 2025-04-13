import numpy as np

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.params = {}
        self.params['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.activation = activation

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        h1 = np.dot(X, W1) + b1
        if self.activation == 'relu':
            a1 = np.maximum(0, h1)
        elif self.activation == 'sigmoid':
            a1 = 1 / (1 + np.exp(-h1))
        scores = np.dot(a1, W2) + b2
        return scores, a1

    def loss(self, X, y, reg):
        scores, a1 = self.forward(X)
        num_examples = X.shape[0]
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs) / num_examples
        W1, W2 = self.params['W1'], self.params['W2']
        reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples
        dW2 = np.dot(a1.T, dscores) + reg * W2
        db2 = np.sum(dscores, axis=0)
        if self.activation == 'relu':
            da1 = np.dot(dscores, W2.T)
            da1[a1 <= 0] = 0
        elif self.activation == 'sigmoid':
            da1 = np.dot(dscores, W2.T) * a1 * (1 - a1)
        dW1 = np.dot(X.T, da1) + reg * W1
        db1 = np.sum(da1, axis=0)
        grads = {}
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=5e-3, learning_rate_decay=0.99,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        best_val_acc = 0
        best_params = {}
        for it in range(num_iters):
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]
            loss, grads = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            if verbose and it % 100 == 0:
                print(f'iteration {it} / {num_iters}: loss {loss}')
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = self.params.copy()
                learning_rate *= learning_rate_decay
        self.params = best_params
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def predict(self, X):
        scores, _ = self.forward(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
