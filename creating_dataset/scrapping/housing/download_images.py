import pickle

with open('./floor_plans_urls.pickle', 'rb') as handle:
    urls = pickle.load(handle)

print(len(urls))
