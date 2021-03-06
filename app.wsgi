activate_this = '/var/www/app/bin/activate_this.py'
exec(compile(open(activate_this, "rb").read(), activate_this, 'exec'), dict(__file__=activate_this))

import sys
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/var/www/html/app')
sys.path.insert(0,'/home/ubuntu/app')
sys.path.insert(1,'/home/ubuntu')

from app import app as application
