import os
import configparser


class DbConfig:
    user = ''
    pwd = ''
    host = ''
    port = ''
    db = ''

    def __init__(self):
        path = os.path.dirname(os.path.realpath(__file__))
        cf = configparser.ConfigParser()
        cf.read(path + '/../../../resources/config/dbconfig.ini')
        self.user = cf.get("baseconf", "user")
        self.pwd = cf.get("baseconf", "pwd")
        self.host = cf.get("baseconf", "host")
        self.port = cf.get("baseconf", "port")
        self.db = cf.get("baseconf", "db")


if __name__ == '__main__':
    config = DbConfig()
