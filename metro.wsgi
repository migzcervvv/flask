# This file contains the WSGI configuration required to serve up your
# web application at http://yourusername.pythonanywhere.com/
# It works by setting the variable 'application' to a WSGI handler of some
# description.
#
# +++++++++++ GENERAL DEBUGGING TIPS +++++++++++
# getting imports and sys.path right can be fiddly!
# We've tried to collect some useful advice here:
# https://help.pythonanywhere.com/pages/DebuggingImportError

# +++++++++++ FLASK +++++++++++
# Flask works using this file with no changes, just import your app instead of
# flask_app and set application to your app instead of flask_app.
#
# If your flask app is not called flask_app.py or your variable is not named
# app, you will need to edit this to match your code.
import sys

# Add your project directory to the sys.path
project_home = '/home/yourusername/mysite'
if project_home not in sys.path:
    sys.path.append(project_home)

# Import your application modules
from app import app as application  # Adjust the from import if necessary

# Also, you can import other components/modules that your app depends on
# Here we import the forecasting module if it needs to be initialized
import forecasting