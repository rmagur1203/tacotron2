import yaml


class HParam:
    def __init__(self, path):
        self.path = path
        self.load()

    def load(self):
        with open(self.path, "r") as f:
            self.hparams = DotDict(yaml.load(f, Loader=yaml.FullLoader))

    def save(self):
        with open(self.path, "w") as f:
            yaml.dump(self.hparams, f)

    def __getattr__(self, item):
        return self.hparams[item]

    def __getitem__(self, item):
        return self.hparams[item]


class DotDict:
    def __init__(self, _dict):
        for key, value in _dict.items():
            if isinstance(value, dict):
                value = DotDict(value)
            self.__dict__[key] = value

    def __getattr__(self, item):
        return self.__dict__[item]

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
