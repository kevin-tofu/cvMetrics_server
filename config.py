import os

VERSION = os.getenv('VERSION', 'v0')
AUTHOR = os.getenv('AUTHOR', 'kevin')

APP_PORT = os.getenv('APP_PORT', 80)

PATH_DATA = os.getenv('PATH_DATA', './temp/')
PATH_ANNOTATION = os.getenv('PATH_ANNOTATION', './annotations/instances_val2017.json')