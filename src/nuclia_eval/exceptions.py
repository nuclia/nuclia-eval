class ModelException(Exception):
    """Generic exception for model errors."""

    pass


class InvalidToolCallException(ModelException):
    """Exception for when a model does not generate an output that can be mapped to the desired metric."""

    pass
