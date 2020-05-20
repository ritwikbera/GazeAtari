
def get_instance(module, name, config, *args):
    try:
        return getattr(module, config[name]['type'])(*args, **config[name]['args'])
    except KeyError:
        return getattr(module, config[name]['type'])()