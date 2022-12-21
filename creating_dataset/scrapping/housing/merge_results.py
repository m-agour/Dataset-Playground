import json
import os
import pickle

import pandas as pd

files = os.listdir('results')


def clean_scrapped_data(data):
    plans = []
    rooms_labels = set()
    urls = []
    for collection in data:
        for record in collection:
            if not record:
                continue
            for entity in record:
                label = entity.get("label")
                key = entity.get("label")
                typ = entity.get("propertyTypeName")

                data = entity.get("data", [])
                for plan in data:
                    area = {}
                    # Area extraction
                    area_data = plan.get('areaConfig', [])
                    for ad in area_data:
                        info = ad.get('areaInfo', {})
                        name = ad.get('name', 'Builtup Area')
                        value = info.get('value')
                        unit = info.get('unit', 'sq.ft')
                        area[name] = {'value': value, 'unit': unit}

                    # Plan Images extraction
                    # ['id', 'price', 'unitsAvailable', 'areaConfig', 'emi', 'averagePrice', 'floorPlan',
                    # 'projectAttributes', 'features', 'additionalFields']
                    price = plan.get('price', {}).get('value')
                    average_price = plan.get('averagePrice', {})
                    if '₹' not in average_price:
                        average_price = None
                    average_price = plan.get('averagePrice', {})

                    # Get rooms counts (at least listed ones)
                    listed_rooms_counts = {}
                    additional = plan.get('additionalFields', [])
                    for room in additional:
                        listed_rooms_counts[room.get('id', 'bedroom')] = room.get('value')
                        rooms_labels.add(room.get('label'))

                    # Getting floor plans
                    plans_imgs = {'2d': [], '3d': []}
                    plan_imgs_2d = []
                    floor_plans = plan.get('floorPlan', [])
                    if floor_plans:
                        for floor_plan in floor_plans:
                            url = floor_plan.get('src', '').replace('version', 'fs-large')
                            mode = floor_plan.get('tag', '2d')
                            if not mode:
                                mode = 'other'
                            plans_imgs.get(mode, []).append(url)
                            if mode != '3d':
                                plan_imgs_2d.append(url)

                    for plan_img in plan_imgs_2d:
                        img_id = plan_img.split('/')[4]
                        plan_data = {
                            'img_id': img_id,
                            'label': label,
                            'key': key,
                            'type': typ,
                            'area': area,
                            'price': price,
                            'average_price': average_price,
                            'listed_rooms_count': listed_rooms_counts,
                            'plans': plans_imgs,
                            'plans2d': plan_img,
                        }
                        urls.append(plan_img)
                        plans.append(plan_data)

    return plans, rooms_labels, urls


if __name__ == '__main__':
    scrapped_data_collection = []
    for root, subdirs, files in os.walk('results'):
        for fname in files:
            fname = os.path.join(root, fname)
            with open(f'{fname}') as f:
                scrapped_data_collection.append(json.load(f))

    data, labels, urls = clean_scrapped_data(scrapped_data_collection)

    with open('data_scrapped_dict.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=4)

    with open('rooms_labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=4)

    with open('floor_plans_urls.pickle', 'wb') as handle:
        pickle.dump(list(set(urls)), handle, protocol=4)

    # print(labels)
    # print(scrapped_data_collection[17][120])

