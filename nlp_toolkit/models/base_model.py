import json


class Base_Model(object):
    """
    Base Keras model for all SOTA models
    """
    def __init__(self):
        self.model = None

    def save(self, weights_file, params_file):
        self.save_weights(weights_file)
        self.save_params(params_file)

    def save_params(self, file_path, invalid_params={}):
        with open(file_path, 'w') as f:
            invalid_params = {'_loss', '_acc', 'model', 'invalid_params', 'token_embeddings'}.union(invalid_params)
            params = {name.lstrip('_'): val for name, val in vars(self).items()
                      if name not in invalid_params}
            print('model hyperparameters:\n', params)
            json.dump(params, f, sort_keys=True, indent=4)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    @classmethod
    def load(cls, weights_file, params_file):
        params = cls.load_params(params_file)
        self = cls(**params)
        self.forward()
        self.model.load_weights(weights_file)
        print('model loaded')
        return self

    @classmethod
    def load_params(cls, file_path):
        with open(file_path) as f:
            params = json.load(f)
        return params
