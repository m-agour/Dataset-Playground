import json
import pickle
import re
import sys

import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
import time
import html

imgs = []

with open('../input/insert/urls.json') as f:
    pros = json.load(f)

with open('../input/notebook0c89399df5/images_urls2.json') as f:
    d = json.load(f)

with open('../input/unwanted1/urls1.json') as f:
    data = json.load(f)

out = []


def get_url(i):
    try:
        return requests.get(f'https://housing.com/in/buy/projects/page/{i}')
    except:
        return None


def get_data(url):
    try:

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/90.0.4430.93 Safari/537.36",
        }
        c = requests.get('https://housing.com' + url, headers=headers)

        if c.status_code != 200:
            print('error loading page!')

    except:
        print('https://housing.com' + url)

        return None
    soup = BeautifulSoup(c.text, 'lxml')
    data_script = soup.find('script', string=re.compile("INITIAL_STATE"))
    js = data_script.text.split('window.__INITIAL_STATE__ = JSON.parse("')[1][:-2]
    js = js.replace('\\"', '"')

    data = json.loads(js)
    # [1]["data"]
    data = data["propertyDetails"]["details"]["details"]["config"]["propertyConfig"]
    return data
    sec = soup.find("section", {'id': 'floorPlan'})
    try:
        imgs = sec.find_all("img")
        imgs = [i['src'] for i in imgs if 'fs.' in i['src']]
        sizes = sec.find_all("div", {'class': "css-lo3e7n"})
    except:
        return None
    if len(imgs) == 0:
        return None
    return [imgs, [i.text for i in sec.find_all("div", {'class': "css-lo3e7n"})]]


dd = 8
data = list(set(data).difference(set(pros)))

count = len(data) // 8
assert dd * count < min(len(data), (dd + 1) * count)

print(len(data))
data = data[dd * count: min(len(data), (dd + 1) * count)]

start = 0

end = len(data)
print(start, end, count)

# print(get_data('/in/buy/projects/page/50276'))

with Pool(12) as p:
    while start < end:
        t = time.perf_counter()

        final = list(p.imap(get_data, data[start: min(start + 400, end)]))
        with open(f'{start}.json', 'w') as f:
            json.dump(final, f)
        start += 400
        print('----------------', start, '-------------------------')
        # time.sleep(10 * 60)
        print((time.perf_counter() - t) / 60 / 60 * end / 400)
