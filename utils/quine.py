from abc import ABC, abstractmethod


class Quine(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def get_param(self, idx):
        assert idx < self.num_params
        subtract = 0
        param = None
        normalized_idx = None
        for i, n_params in enumerate(self.cum_params_arr):
            if idx < n_params:
                param = self.param_list[i]
                normalized_idx = idx - subtract
                break
            else:
                subtract = n_params
        return param.view(-1)[normalized_idx]


class Auxiliary(Quine):
    def __init__(self, config, model):
        super(Auxiliary, self).__init__()
        self.config = config
        self.model = model
        self.aux_model = self.get_aux()

    def get_aux(self):
        exit()

        return model


class Vanilla(Quine):
    def __init__(self, config, model):
        super(Vanilla, self).__init__()
        self.config = config
        self.model = model
        self.van_model = self.van_aux()

    def get_van(self):

        return model
