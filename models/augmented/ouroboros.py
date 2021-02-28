from abc import ABC, abstractmethod


class Ouroboros(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def get_param(self, idx):
        pass


class Jung(Ouroboros):
    def __init__(self, config, model):
        super(Jung, self).__init__()
        self.config = config
        self.model = model
        self.aux_model = self.get_aux()

    def get_aux(self):

        return model


class Kekule(Ouroboros):
    def __init__(self, config, model):
        super(Kekule, self).__init__()
        self.config = config
        self.model = model
        self.van_model = self.van_aux()

    def get_van(self):

        return model


class Godel():
    pass


class Escher():
    pass


class Bach():
    pass
