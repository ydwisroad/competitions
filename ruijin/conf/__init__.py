from configparser import ConfigParser

conf = None


def get_config(conf_path="conf/config.conf"):
    global conf
    if not conf:
        conf = ConfigParser()
        conf.read(conf_path)
    return conf
