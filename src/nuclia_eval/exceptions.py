class ModelException(Exception):
    pass


class NoOutputException(ModelException):
    pass


class InvalidToolCallException(ModelException):
    pass
