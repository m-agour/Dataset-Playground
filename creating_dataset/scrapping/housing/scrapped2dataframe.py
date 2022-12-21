import os.path
import pickle

import pandas as pd

with open('./housing/data_scrapped_dict.pickle', 'rb') as handle:
    data_lst = pickle.load(handle)
    df = pd.DataFrame(data_lst)

with open('./housing/floor_plans_urls.pickle', 'rb') as handle:
    urls = pickle.load(handle)


def has_2d_plan(urls):
    for url in urls:
        print()


print(df[df.plans2d.apply(lambda x: bool(x))].reset_index()['plans2d'][60000])
