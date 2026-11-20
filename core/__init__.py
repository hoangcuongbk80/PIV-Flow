__all__ = ["PIVFlow"]


def __getattr__(name):
    if name == "PIVFlow":
        from .piv_flow import PIVFlow
        return PIVFlow
    raise AttributeError(name)
