import functools


def rsetattr(obj, attr, val):
    """
        Sets a nested attribute, e.g. model.encoder. Code based on:
        
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """
    pre, _, post = attr.rpartition('.')

    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
        Gets a nested attribute, e.g. model.encoder. Code based on:
        
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
