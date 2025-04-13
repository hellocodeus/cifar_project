from model import ThreeLayerNet

def hyperparameter_search(x_train, y_train, x_val, y_val):
    results = {}
    best_val = -1
    best_net = None
    learning_rates = [1e-3, 5e-3, 1e-2]
    hidden_sizes = [128, 256, 512, 1024]
    regs = [1e-3, 5e-3, 1e-2]
    for lr in learning_rates:
        for hs in hidden_sizes:
            for reg in regs:
                net = ThreeLayerNet(x_train.shape[1], hs, 10)
                stats = net.train(x_train, y_train, x_val, y_val,
                                  learning_rate=lr, reg=reg,
                                  num_iters=1000, verbose=False)
                val_acc = (net.predict(x_val) == y_val).mean()
                results[(lr, hs, reg)] = val_acc
                if val_acc > best_val:
                    best_val = val_acc
                    best_net = net
    for lr, hs, reg in sorted(results):
        val_acc = results[(lr, hs, reg)]
        print(f'lr {lr}, hidden_size {hs}, reg {reg}: validation accuracy: {val_acc}')
    print(f'Best validation accuracy achieved during search: {best_val}')
    return best_net