import train_model

if __name__ == '__main__':
    config = {
        'beta': 2,
        'num_prune': 1
    }
    train_model.train_model(config)