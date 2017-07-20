class Config(object):

    def __init__(self):

        self.mode = 'mlp'  # 'cnn' or 'mlp'
        self.parse_seed = 9999
        self.torch_seed = 9372

        self.mnist_path = '/home/khshim/data/mnist/'
        self.save_path = './best_model_' + self.mode + '.pt'
        self.num_valid = 10000
        self.batch_size = 200
        self.eval_batch_size = 1000
        self.num_workers = 4

        self.max_epoch = 1000
        self.max_change = 4
        self.max_patience = 5

        self.initial_lr = 0.001
