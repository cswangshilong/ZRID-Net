import os
import cv2

# Code for generating dehaze images by real (in order to calculate Loss_CLAHE)
if __name__ == '__main__':

    # for dataset_name in ['BeDDE']:
    # # for dataset_name in ['outdoor','indoor','BeDDE','IO']:
    # # for dataset_name in ['IO-haze']:
    #
    #     data_path = '{}/train/haze/'.format(dataset_name)
    #     out_path = '{}/train/CLAHE/'.format(dataset_name)
    #
    #     if not os.path.exists(out_path):
    #         os.makedirs(out_path)
    #
    #     name_list = list(os.walk(data_path))[0][2]
    #     # print(name_list)
    #     for i, name in enumerate(name_list):
    #         img0 = cv2.imread(data_path + name)
    #
    #         b,g,r = cv2.split(img0)
    #         img_rgb = cv2.merge([r,g,b])
    #         img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    #         clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(16,16))
    #         img_rgb2 = clahe.apply(img_rgb.reshape(-1)).reshape(img_rgb.shape)
    #         r, g, b = cv2.split(img_rgb2)
    #         img_out = cv2.merge([b, g, r])
    #         # print(out_path+name)
    #         cv2.imwrite(out_path+name,img_out)
    #
    #     print("{} train dataset done!".format(dataset_name))
    #
    #
    # for dataset_name in ['outdoor','indoor','BeDDE','IO']:
    # # for dataset_name in ['IO-haze']:
    #
    #     data_path = '{}/test/haze/'.format(dataset_name)
    #     out_path = '{}/test/CLAHE/'.format(dataset_name)
    #
    #     if not os.path.exists(out_path):
    #         os.makedirs(out_path)
    #
    #     name_list = list(os.walk(data_path))[0][2]
    #     # print(name_list)
    #     for i, name in enumerate(name_list):
    #         img0 = cv2.imread(data_path + name)
    #
    #         b,g,r = cv2.split(img0)
    #         img_rgb = cv2.merge([r,g,b])
    #         img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    #         clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(16,16))
    #         img_rgb2 = clahe.apply(img_rgb.reshape(-1)).reshape(img_rgb.shape)
    #         r, g, b = cv2.split(img_rgb2)
    #         img_out = cv2.merge([b, g, r])
    #         # print(out_path+name)
    #         cv2.imwrite(out_path+name,img_out)
    #
    #     print("{} test dataset done!".format(dataset_name))

    for dataset_name in ['outdoor']:

        data_path = '{}/train/haze/'.format(dataset_name)
        out_path = '{}/train/CLAHE/'.format(dataset_name)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        name_list = list(os.walk(data_path))[0][2]
        # print(name_list)
        for i, name in enumerate(name_list):
            img0 = cv2.imread(data_path + name)

            b,g,r = cv2.split(img0)
            img_rgb = cv2.merge([r,g,b])
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(16,16))
            img_rgb2 = clahe.apply(img_rgb.reshape(-1)).reshape(img_rgb.shape)
            r, g, b = cv2.split(img_rgb2)
            img_out = cv2.merge([b, g, r])
            # print(out_path+name)
            cv2.imwrite(out_path+name,img_out)

        print("{} dataset done!".format(dataset_name))

