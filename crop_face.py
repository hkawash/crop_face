# %%
import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

# 画像ファイル名では，日本語入りのファイル名はだめかも

src_dir = 'input_imgs'  # 入力フォルダ
dst_dir = 'crop_imgs'  # 出力フォルダ
input_types = ('*.jpg', '*.png')  # 検索する画像拡張子
output_ext = 'jpg'  # 出力フォーマット
output_size = (64, 64)  # 出力サイズ

margin = 20  # 検出矩形の上下左右にどれだけマージンを入れるか
show_image = True   # True: 画像を検出矩形と共に表示


print('src_dir:', src_dir)

os.makedirs(dst_dir, exist_ok=True)
print('dst_dir:', dst_dir)

# create file list
img_list = []
for t in input_types:
    img_list.extend(glob.glob(src_dir + '/' + t))
print('len(img_list):', len(img_list))
print(img_list)


detector = cv2.CascadeClassifier('./haar_xml/haarcascade_frontalface_default.xml')

# for each image
for img_id, img_fname in enumerate(img_list):
    # print('img: ', img_fname)
    try:
        img = cv2.imread(img_fname)
        img_rect = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for visualization
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(str(e))
    if img_gray is None:
        continue
    rects = detector.detectMultiScale(img_gray, scaleFactor=1.02, minNeighbors=4, minSize=(64, 64))

    basename = os.path.splitext(os.path.basename(img_fname))[0]
    print(f'{img_fname} .... detected {len(rects)} regions')

    # for each detected region
    for (i, (x, y, w, h)) in enumerate(rects):
        ul = [x - margin, y - margin]  # upper left
        br = [x + w + margin, y + h + margin]  # bottom right
        face = img[ul[1]:br[1], ul[0]:br[0]].astype('float32')
        if np.size(face) > 0:
            try:
                # ここで検出された各矩形領域を画像として保存
                face = cv2.resize(face, output_size)
                dst_fname = f'{basename}_{i}.{output_ext}'
                cv2.imwrite(f'{dst_dir}/{dst_fname}', face)
                print(dst_fname)
            except Exception as e:
                print(str(e))
        
        cv2.rectangle(img_rect, (ul[0], ul[1]), (br[0], br[1]), (0, 255, 0), 2)
        cv2.rectangle(img_rect, (x, y), (x + w, y + h), (0, 128, 0), 2)

    # Display detected regions
    # if show_image:
    #     cv2.imshow(f'img:{img_id} - detected regions', img_rect)
    #     cv2.waitKey(0)  # wait

    if show_image:
        plt.imshow(img_rect)
        plt.title(img_fname)
        plt.show()
# %%
