

from multiprocessing import Pool
import tqdm
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    images = [f for f in listdir(IMG_PATH) if isfile(join(IMG_PATH, f))]

    dataframe = None

    for i, img_p in tqdm(enumerate(images)):
        plan_data = vectorize_plan(img_p)

        if not dataframe:
            dataframe = pd.DataFrame([plan_data])
        else:
            dataframe.loc[len(dataframe.index)] = plan_data

        if i % 100 == 0:
            dataframe.to_pickle('dataset.pkl')

    with Pool(12) as p:
        data = list([tqdm.tqdm(p.imap(vectorize_plan, images[:2]), total=4)])
        dataframe = pd.DataFrame(data)
        dataframe.to_pickle('dataset.pkl')
        p.close()
p.to_pickle('dataset.pkl')
df = pd.read_pickle('dataset_1.pkl')
print(df.dtypes)
print(df.head(1))

# doors = draw_multi_polygons(df['bedroom'][26], df['size'][30])
# general = draw_multi_polygons(df['general'][26], df['size'][30])
# wall = draw_multi_polygons(df['bathroom'][26], df['size'][30])

# new = cv2.merge([doors , general, wall])

# imshow(wall)
# imshow(new)
# print(p['window'])
# print(p['servant'])

# a = wkt.loads("MULTIPOLYGON (((231 682, 231 682, 231 693, 231 693, 231 693, 231 693, 231 693, 231 693, 231 693, 231 693, 231 708, 231 708, 231 708, 231 742, 231 742, 231 742, 231 742, 231 742, 231 742, 231 839, 231 839, 231 840, 231 840, 189 840, 189 840, 189 840, 189 961, 327 961, 327 961, 337 961, 337 961, 337 961, 337 961, 350 961, 350 961, 564 961, 564 961, 564 892, 564 892, 564 892, 564 892, 564 892, 576 892, 576 892, 576 892, 576 892, 576 892, 576 840, 576 840, 564 840, 564 839, 564 839, 564 839, 564 839, 564 693, 564 693, 564 693, 564 693, 564 693, 564 693, 564 693, 564 693, 564 682, 564 682, 231 682)), ((864 682, 864 961, 1186 961, 1186 907, 1186 907, 1186 907, 1186 907, 1186 907, 1198 907, 1198 907, 1198 854, 1198 854, 1198 854, 1198 854, 1198 854, 1198 854, 1186 854, 1186 854, 1186 715, 1186 715, 1186 715, 1186 708, 1186 693, 1186 693, 1186 693, 1186 693, 1186 693, 1186 682, 900 682, 900 682, 900 693, 875 693, 875 682, 875 682, 875 682, 875 682, 875 682, 875 682, 875 682, 864 682)), ((1092 186, 1092 236, 1092 236, 1014 236, 1014 236, 1000 236, 1000 236, 989 236, 989 236, 983 236, 983 236, 934 236, 934 236, 920 236, 920 236, 825 236, 825 381, 825 381, 825 381, 864 381, 864 381, 864 495, 875 495, 875 495, 875 495, 875 495, 875 495, 875 495, 896 495, 896 495, 896 495, 1186 495, 1186 495, 1186 495, 1198 495, 1198 495, 1198 495, 1198 495, 1198 440, 1198 440, 1198 440, 1186 440, 1186 440, 1186 344, 1186 344, 1186 344, 1186 344, 1198 344, 1198 344, 1198 344, 1198 344, 1198 344, 1198 344, 1198 344, 1198 284, 1186 284, 1186 284, 1186 284, 1186 284, 1186 284, 1186 186, 1092 186)))")

print(df['wall_lines'][0])

# wall = draw_multi_polygons(a, (2000, 2000))
# imshow(wall)

print(type(df['wall_lines'][0][0]))

BaseGeometry
