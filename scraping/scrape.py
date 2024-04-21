from twitter import Twitter
import time
import json
from datetime import datetime

twitter = Twitter()

twitter.login()
time.sleep(2)
twitter.historical_search("bitcoin")
time.sleep(5)
twitter.quit()
