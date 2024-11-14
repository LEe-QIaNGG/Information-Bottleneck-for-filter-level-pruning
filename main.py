import train_model
config = {
    'beta': 2,
    'num_prune': 1
}
if __name__ == '__main__':
    train_model.train_model(config)