import os
import time
import imageio
import cv2
from shutil import copy2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from image_slicer import slice, save_tiles
from PIL import Image
from matplotlib import pyplot as plt

'''
MERGED ALL PREPROCESSES
1) Cut images and segmaps into tiles (8x8 whole image,7x7 cropped by 256 on each side) --> Results in 512x512px imgs
2) Rotate each tile by 90, 180, 270 degrees
-> Keep corresponding images and segmaps naming convention
-> This process results in 14 916 samples
'''


def reduce_dims(segmap):
    # INFO ABOUT INPUT IM:
    # type <class 'numpy.ndarray'>
    # (4096, 4096, 3)
    # shape type <class 'tuple'>
    # 4
    # [[0, 0, 0], [0, 128, 128], [0, 0, 128], [0, 128, 0]]

    #seg_maps = load_images('seg/')
    #key = 'img5.png'
    #segmap = seg_maps[key]
    new_segmap = np.zeros((segmap.shape[0], segmap.shape[1]), dtype=np.int32)

    # print(type(segmap[2500,1100,:]), type(np.asarray([0,128,0])))
    # print(np.asarray([0,128,0]))
    # print(segmap[2500,1100,:])
    # print(np.array_equal(segmap[2500,1100,:], np.asarray([0,128,0])))

    for i in range(segmap.shape[0]):
        for j in range(segmap.shape[1]):
            if np.array_equal(segmap[i,j,:], np.asarray([0,0,0])):
                continue
            elif np.array_equal(segmap[i,j,:], np.asarray([0, 128, 128])):
                new_segmap[i,j] = 1
            elif np.array_equal(segmap[i,j,:], np.asarray([0, 0, 128])):
                new_segmap[i,j] = 2
            elif np.array_equal(segmap[i,j,:], np.asarray([0, 128, 0])):
                new_segmap[i,j] = 3
    return new_segmap


def load_images(subfolder):

    abs_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(abs_path, 'images/' + subfolder)
    #curpath = 'images/' + subfolder

    # theimg = os.path.join(orig_path, 'img1.png')
    # theseg = os.path.join(seg_path, 'img1.png')

    images = {}
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images[filename] = img
    return images


def im_augment():

    ia.seed(1)

    directory = 'images/cut/orig/'
    counter = 1
    for file in os.listdir(directory):
        image = cv2.imread('images/cut/orig/' + file)
        segmap = cv2.imread('images/cut/seg/' + file)

        # merge the three channels to a single one
        # segmap1 = np.argmax(segmap, axis=2)
        segmap = reduce_dims(segmap)

        #segmap1 = SegmentationMapsOnImage(segmap1, shape=image.shape)
        segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

        print(f'\n{counter}\t\tCopying original {file} ...\n')
        imageio.imwrite('images/augmented/orig/' + file[:-4] + str(1) + '.png', image)
        imageio.imwrite('images/augmented/seg/' + file[:-4] + str(1) + '.png', segmap.draw()[0])
        #
        # aug = iaa.Rot90(1)
        # img_aug, segmaps_aug = aug(image=image, segmentation_maps=segmap)
        # print('writing 90 degrees rotation...\n')
        # imageio.imwrite('images/augmented/orig/' + file[:-4] + str(2) + '.png', img_aug)
        # imageio.imwrite('images/augmented/seg/' + file[:-4] + str(2) + '.png', segmaps_aug.draw()[0])
        #
        # aug = iaa.Rot90(2)
        # img_aug, segmaps_aug = aug(image=image, segmentation_maps=segmap)
        # print('writing 180 degrees rotation...\n')
        # imageio.imwrite('images/augmented/orig/' + file[:-4] + str(3) + '.png', img_aug)
        # imageio.imwrite('images/augmented/seg/' + file[:-4] + str(3) + '.png', segmaps_aug.draw()[0])
        #
        # aug = iaa.Rot90(3)
        # img_aug, segmaps_aug = aug(image=image, segmentation_maps=segmap)
        # print('writing 270 degrees rotation...\n')
        # imageio.imwrite('images/augmented/orig/' + file[:-4] + str(4) + '.png', img_aug)
        # imageio.imwrite('images/augmented/seg/' + file[:-4] + str(4) + '.png', segmaps_aug.draw()[0])

        counter += 1


def segmaps_rgb_to_greyscale():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    directory = cur_dir + '/images/final_ds_notaugmented/'
    # destination = cur_dir + '/images/final_ds/'

    counter = 1
    for img in os.listdir(directory + 'seg/'):
        segmap = imageio.imread(directory + 'seg/' + img)
        newseg = np.zeros((segmap.shape[0], segmap.shape[1]), dtype=np.int32)
        print(segmap[0, 0])
        for i in range(segmap.shape[0]):
            for j in range(segmap.shape[1]):
                # background
                if np.array_equal(segmap[i, j, :], np.asarray([0, 0, 0])):
                    continue
                # type 1 - red
                elif np.array_equal(segmap[i, j, :], np.asarray([230, 25, 75])):
                    newseg[i, j] = 1
                # type 2 - green
                elif np.array_equal(segmap[i, j, :], np.asarray([60, 180, 75])):
                    newseg[i, j] = 2
                # type 3 - yellow
                elif np.array_equal(segmap[i, j, :], np.asarray([255, 225, 25])):
                    newseg[i, j] = 3

        #imageio.imwrite(directory + 'seg_greyscale/' + img, newseg) # y u not workin m8
        im = Image.fromarray(newseg)
        im.save(directory+'seg_greyscale/' + img)

        if counter % 200 == 0:
            print(f'RGB -> Greyscale converted: {counter} ...')
        counter += 1


def rename_data():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    directory = cur_dir + '/images/augmented/'
    destination = cur_dir + '/images/final_ds_notaugmented/'

    counter = 1
    for img in os.listdir(directory + 'orig/'):
        copy2(directory + 'orig/' + img, destination + 'orig/img_' + str(counter) + '.png')
        copy2(directory + 'seg/' + img, destination + 'seg/img_' + str(counter) + '.png')
        counter += 1


def cut_images8x8(orig_images):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    for img in orig_images:
        sliced_img = slice('images/orig/' + img, 64, save=False)
        save_tiles(tiles=sliced_img, directory=cur_dir+'/images/cut/orig', prefix=img[:-4]+'_8', format='png')

        sliced_segmap = slice('images/seg/' + img, 64, save=False)
        save_tiles(tiles=sliced_segmap, directory=cur_dir+'/images/cut/seg', prefix=img[:-4]+'_8', format='png')


def cut_images7x7(orig_images):
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    left = 256
    top = 256
    right = 3840
    bottom = 3840

    for img in orig_images:
        # crop image
        im = Image.open(cur_dir + '/images/orig/' + img)
        new_im = im.crop((left, top, right, bottom))
        # save cropped img
        imageio.imwrite('images/orig/' + img[:-4] + '_cropped' + '.png', new_im)
        # slice cropped img
        sliced_img = slice('images/orig/' + img[:-4] + '_cropped' + '.png', 49, save=False)
        # save all tiles
        save_tiles(tiles=sliced_img, directory=cur_dir+'/images/cut/orig', prefix=img[:-4]+'_7', format='png')

        # same for segmap
        sm = Image.open(cur_dir + '/images/seg/' + img)
        new_sm = sm.crop((left, top, right, bottom))
        # save cropped segmap
        # imageio.imwrite('images/seg/' + img[:-4] + '_cropped' + '.png', new_sm)
        new_sm.save("images/seg/" + img[:-4] + "_cropped.png", "PNG")  # USED DIFFERENT SAVING METHOD SINCE IMAGEIO COULDNT SAVE CROPPED SEGMAP
        # slice cropped segmap
        sliced_segmap = slice('images/seg/' + img[:-4] + '_cropped' + '.png', 49, save=False)
        # save all tiles
        save_tiles(tiles=sliced_segmap, directory=cur_dir+'/images/cut/seg', prefix=img[:-4]+'_7', format='png')
        # todo ADD DELETION OF _CROPPED.png


def aug_check():
    ia.seed(1)

    sometimes6 = lambda aug: iaa.Sometimes(0.6, aug)
    sometimes8 = lambda aug: iaa.Sometimes(0.8, aug)

    seq = iaa.Sequential([
        iaa.OneOf([
            sometimes6(iaa.CropAndPad(percent=(0, 0.2), pad_mode="constant", pad_cval=160)),
            sometimes6(iaa.GaussianBlur(sigma=(0.0, 3.0)))
        ]),
        sometimes8(iaa.Affine(rotate=(-180, 180), mode='constant', cval=160),),
        iaa.Fliplr(0.25),
        iaa.Flipud(0.25)
    ], random_order=True)

    directory = 'images/final_ds_notaugmented/orig/'
    counter = 1
    for file in os.listdir(directory):
        image = cv2.imread('images/final_ds_notaugmented/orig/' + file)
        segmap = cv2.imread('images/final_ds_notaugmented/seg/' + file)

        # merge the three channels to a single one
        # segmap1 = np.argmax(segmap, axis=2)
        # segmap = reduce_dims(segmap)

        # segmap1 = SegmentationMapsOnImage(segmap1, shape=image.shape)
        segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

        images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)

        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('original_img')
        plt.imshow(image)
        plt.subplot(232)
        plt.title('aug_img')
        plt.imshow(images_aug_i)
        plt.subplot(233)
        plt.title('aug_mask')
        plt.imshow(segmaps_aug_i.get_arr())
        plt.show()
        print("hi")

        counter += 1


if __name__ == '__main__':
    start = time.time()
    # # Load and cut images [orig,seg --> cut/orig/, cut/seg/]
    # orig_imgs = load_images('orig/')
    # orig_segmaps = load_images('seg/')
    # cut_images8x8(orig_imgs)
    # print('Img cutting 8x8 ***DONE***')
    # cut_images7x7(orig_imgs)
    # print('Img cutting 7x7 ***DONE***')

    # # augment images [cut/ -> augmented/]
    # im_augment()
    #
    # # # rename images [augmented/ -> final_ds/] (copy n rename to preserve whole pipeline in case of a mistake)
    # rename_data()

    # # to work with segmaps: I need NxN segmaps with 0,1,...,n pixel values where n=numof classes-1
    # # THEN IN CODE use "one hot encoding", where we create NxNxn segmaps, where in each of n  is BINARY image with
    # ones where there is corresponding object of n-th type
    # rgb to greyscale [final_ds/seg -> final_ds/seg_greyscale]
    # segmaps_rgb_to_greyscale()


    # just check and try out of augmentation techniques that will be used in training
    aug_check()
    end = time.time()
    print(f'Total time: {end - start} seconds')
