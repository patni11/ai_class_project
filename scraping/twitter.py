from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from datetime import datetime, date, timedelta
import json
import random


class Twitter:
    def __init__(self, options: list = []) -> None:
        driver_options = webdriver.FirefoxOptions()
        if "headless" in options:
            driver_options.add_argument("-headless")
        self.driver = webdriver.Firefox(options=driver_options)

        with open(".cred", "r") as f:
            cred = json.load(f)
            self.username = cred["username"]
            self.password = cred["password"]

        pass

    def quit(self) -> None:
        self.driver.quit()

        pass

    def login(self) -> None:
        self.driver.get("https://twitter.com/login")
        self.driver.implicitly_wait(3)

        username = self.driver.find_element(
            By.XPATH,
            "/html/body/div[1]/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input",
        )
        username.send_keys(self.username)
        username.send_keys(Keys.RETURN)

        password = self.driver.find_element(
            By.XPATH,
            "/html/body/div[1]/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input",
        )
        password.send_keys(self.password)
        password.send_keys(Keys.RETURN)

        pass

    def search(self, term: str | list, count: int = 10) -> dict[str, list[str]]:
        if isinstance(term, list):
            return dict(zip(term, map(lambda t: self.__search_term(t, count), term)))
        else:
            return {term: self.__search_term(term, count)}

    def __search_term(self, term: str, count: int) -> list[str]:
        self.driver.get(
            f"https://twitter.com/search?q={term}"  #  min_faves%3A5000 until%3A2024-03-31 since%3A2024-03-30"
        )
        self.driver.implicitly_wait(3)

        scroll = self.driver.find_element(
            By.XPATH, "/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div"
        )
        tweets = []
        timeout = time.time() + 30
        while len(tweets) < count and time.time() < timeout:
            try:
                tweet = self.driver.find_element(
                    By.XPATH,
                    "/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/section/div/div/div[1]/div/div/article/div/div/div[2]/div[2]/div[2]",
                ).text
                if tweet not in tweets:
                    tweets.append(tweet)
                else:
                    scroll.send_keys(Keys.UP)
            except:
                scroll.send_keys(Keys.DOWN)

        print(tweets)

        return tweets

    def historical_search(
        self, term: str, start: datetime.date = datetime.today().date()
    ):
        with open("historical.json", "r") as f:
            historical = json.load(f)

        date = start
        genesis = datetime(2019, 1, 1).date()
        while date > genesis:
            print(date)
            if historical.get(str(date), []) == []:
                historical[str(date)] = self.__search_term_date(term, date, 1)
                with open("historical.json", "w") as f:
                    json.dump(historical, f)
            date -= timedelta(days=1)

    def __search_term_date(self, term: str, date: date, count: int) -> list[str]:
        prev = date - timedelta(days=1)
        self.driver.get(
            f"https://twitter.com/search?q={term} min_faves%3A1000 until%3A{date} since%3A{prev} lang:en"
        )
        self.driver.implicitly_wait(3)

        scroll = self.driver.find_element(
            By.XPATH, "/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div"
        )
        tweets = []
        timeout = time.time() + 10
        i = 1
        while len(tweets) < count and time.time() < timeout:
            time.sleep(1)
            scroll.send_keys(Keys.UP)
            try:
                tweet = self.driver.find_element(
                    By.XPATH,
                    f"/html/body/div[1]/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/section/div/div/div[{i}]/div/div/article/div/div/div[2]/div[2]/div[2]",
                ).text
                if tweet not in tweets:
                    tweets.append(tweet)
                else:
                    scroll.send_keys(Keys.UP)
            except:
                # if random.random() < 0.002:
                # i += 1
                scroll.send_keys(Keys.DOWN)

        return tweets
