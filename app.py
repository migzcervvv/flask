from flask import Flask 
from flask import (Flask, g, render_template, request, flash, url_for, redirect, session, json, jsonify)
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2 import service_account

app = Flask(__name__)

sa_dict =  {
  "type": "service_account",
  "project_id": "metrobreathe",
  "private_key_id": "a9c826609e3f425cbe6a9d5e68abbc4d1afa3c3e",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC24G9UyiKoFksK\no7HU8VBz+K6m+hTcemjd8Li9XiW8rnmAZwNsfbE+LoN55L2li4JlawP9Yw+8wSKM\nZFNKwQIVLxjaDNfPF2A2myXl2dDXLCaIKEw5aS6FcvKPyAgDPndj2LQ5fX/68VWB\nuUwnB+pBK7ziJYolwfOpBzPwCEZQyaw6gUctlTzKx8Xz1Y1u/7woagWkqXrIljw1\nh3C2BgZGsyS3WkYtKEusT9gs9JlOcNH8+yDC5A5hHhwnA8UnTiDZJ2MQTwr4xzsA\nDCE1TN+UStiExO5YJIcSG84PLNP3/mV+ikG2mrPWaoPxIdu5WTtf8Xuv8yUlBJkc\n1oWWn2PRAgMBAAECggEAQbCKrSVSn5gqam70cO4hnRbF/bq1BaF+y8ItSfBok8cE\nY+gc5bqckR56Ia3VpYJgW3e+XiVYQNALTey3spFz4xIg1ipo1r2p1NOUIGVGTBRS\n3vPCtZifmlX45UbQAgJNNnNgAYqgDw7jTQ63WQnRzokcdwtO/VMW6C9rK5x9TYUX\nxqvk3LyxwzCMZA27g5psGfAV+YFW8ZzbKCvGsDyWz5W7xt4sPQIiY0oz7nMp2aOE\nvwjw4JmHvg6E8uEoJeFLhcAr/6SkeHUL6lUvZ3VSBY/00H5/2KsA4RJuH805z0Ww\n+7FGwD8GX2mCld51lL4ZRiU6hSlrl9zjZ+cmt/7z9QKBgQDqOjs2Z+KGKGOZRFw2\nTrgWoGEjK4V/XtJIkoe6lsDJlRw3k4yF9s9ZtSJ2TEFCqGtLGSVueMWrT8PrzgEF\nlDKtqoRHWhyD4m+ABQSfhzUUOhQr2aaOUxGmpWb6MV6YmJrV1XNT4hNXEoupYPeN\ng97JbhKAlK27wumlGF/D4jN1fwKBgQDH4D2WnxOA0SYeFcrLzrAnt/1m6LjL9Oto\nTLLL7PjivJxLYkSp9fcqvMjuS3/YASs5oWeq8XmcU8hcHSJAe3X/q9RRmPsp4/c5\niRr/LWT/xPsnSTjA1b+5IjvCD+a50UucyDL4S/pxJnuqiV4857eamjoPRlXbVeYR\nHskMmMrurwKBgBbIuU/OhSt1rFKRYsLpn2CcEzbfmenM6Hmkv004MuKo/YiucqHp\nYzwfsY1+V27LgTxZ4tk/KENEzBMZA+JuCwv3pUxniJSPpzb1xgBan6ArEiE918TK\ncdIbDsxRlxkS5yTb0Y8cU7NJm4pLY2lBpZ9EdMpLLCcyi5XCrDYav9SzAoGBAKUo\nJJv2HD6AE9geN7FKu4JGZQwI1tIpMe/AGKmqyUlJgnhD3er2xGK7FejZ1+yboqT/\nOtMkr1E+Zbu/kxLnMWyvBkTafQdzSFBxey5Jy0AQ+1rOBShKNx76K5jCXOtEBw+o\n2X0UAwBGRT94PLdk3PDR6ZG/k6gmhXG2F3jiNCUfAoGBALlakuR0rqCXZlXX78Ov\nwuHmPebH5WcYfkP7wl1ydG4Wuw36+RDAwZGbigJAlsPaQHPzigFOdG692xyKdXOo\n/DamCF8Trru0DmEi8cSRHnrnDZ1U4vJXXRxI+sL95DMbzmmhOxzySBaee5PqTEXu\nKkJKaBZtG5rj21Pk4MW3sWrT\n-----END PRIVATE KEY-----\n",
  "client_email": "migzcerv@metrobreathe.iam.gserviceaccount.com",
  "client_id": "104861993626947114754",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/migzcerv%40metrobreathe.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

creds = service_account.Credentials.from_service_account_info(sa_dict)
service = build("sheets", "v4", credentials = creds)

@app.route("/")

def home():
    return "Hello world, from flask"

@app.route("/readSheets", methods=['POST'])
def readSheets_API():
    param = json.loads(request.data.decode())
    range = param ['range']
    result = readGsheet(range)
    return result

def readGsheet(range):
    SPREADSHEET_ID = "1QY_I_7ci1pZkraeUopbVrYgI-kphveqrOKsGq890Z_w"
    SHEETRANGE = range
    service = build("sheets", "v4", credentials = creds)
    sheet = service.spreadsheets()
    sheetread = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=SHEETRANGE).execute()
    values = sheetread.get('values',[])
    items = []

    for item in values:
        items.append(item[0])

    result={"result:": items}

    return jsonify(result)