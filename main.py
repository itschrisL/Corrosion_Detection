from pathlib import Path
from scipy._lib.six import xrange
from Corrosion_Detection_Model import CorrosionDetectionModel
from os import listdir
from os.path import isfile, join
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# Helper method that will get all the files in a directory and return it as a list
def load_data_from_folder(img_dir, label_img_dir):
    # try to get the files from the directory
    try:
        img_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
        label_img_files = [f for f in listdir(label_img_dir) if isfile(join(label_img_dir, f))]
    except NotADirectoryError:
        print("Could not load files from " + str(img_dir) + " directory")
        return None

    rtrn_data = []
    for i in range(0, len(img_files)):
        rtrn_data.append((
            Image.open(img_dir/img_files[i]),
            Image.open(label_img_dir/label_img_files[i])
        ))

    # Shuffle the data here to make it easier in the future
    np.random.shuffle(rtrn_data)
    return rtrn_data


# Helper method that gets the rgb values of an image and returns a 2D array of tuples of rgb values
def get_rgb_of_img(img):
    pixels = list(img.getdata())
    width, height = img.size
    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
    return pixels


def img_to_black_white(img):
    thresh = 200
    fn = lambda x: 255 if x > thresh else 0
    new_img = img.convert('L').point(fn, mode='1')
    return new_img


def load_data_to_file(images):
    f = open(RGB_DATA_FILE_NAME, 'w')
    for im in images:
        f.write(get_rgb_of_img(im[0]))
        f.write(" || ")
        f.write(get_rgb_of_img(im[1]))
        f.write("\n")
    f.close()


# Helper method to read the data from the file
def read_data_from_file():
    rtn_list = []
    f = open(RGB_DATA_FILE_NAME, 'r')
    lines = f.readlines()
    for line in lines:
        line.replace("\n", "")
        x = line.split(" || ")
        rtn_list.append((eval(x[0]), eval(x[1])))


def show_image(img):
    plt.imshow(img)
    plt.show()


def show_both_images(img1, img2, img1_title="", img2_title="", subplot_title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img1)
    ax1.set_title(img1_title)
    ax2.imshow(img2)
    ax2.set_title(img2_title)
    fig.suptitle(subplot_title)
    plt.show()


def show_three_images(img_list, title_list):
    fig, (ax1, ax2) = plt.subplots(2, 2, sharey=True)
    ax1[0].imshow(img_list[0])
    ax1[0].set_title(title_list[0])
    ax2[0].imshow(img_list[1])
    ax2[0].set_title(title_list[1])
    ax2[1].imshow(img_list[2])
    ax2[1].set_title(title_list[2])
    plt.show()


# Helper method that resize an image
def reformat_image(img, new_width=256, new_height=256):
    new_img = img.resize((new_width, new_height))
    return new_img


def normalize_img(img):
    new_img = []
    for r in range(0, len(img)):
        new_img.append([])
        for c in range(0, len(img[0])):
            new_img[r].append(tuple([float(p/255) for p in img[r][c]]))
    return new_img


def binary_convert(img):
    copy_img = []
    for r in range(0, len(img)):
        copy_img.append([])
        for c in range(0, len(img[0])):
            x = sum(img[r][c])
            if x > 500:
                # copy_img[r][c] = 1
                copy_img[r].append((1, 1, 1))
            else:
                #copy_img[r][c] = 0
                copy_img[r].append((0, 0, 0))
    return copy_img


def convert_from_binary(img):
    copy_img = []
    for r in range(0, len(img)):
        t = []
        copy_img.append(t)
        for c in range(0, len(img[0])):
            x = sum(img[r][c])
            # y = [int(i*255) for i in img[r][c]]
            if x <= 1:
                # copy_img[r][c] = (1, 1, 1)
                copy_img[r].append((1, 1, 1))
            elif x > 1:
                # copy_img[r][c] = (255, 255, 255)
                copy_img[r].append((255, 255, 255))
            else:
                print("Should not be here")
                return

    return copy_img

def create_tuples(img):
    new_img = []
    for r in range(0, len(img)):
        new_img.append([])
        for c in range(0, len(img[0])):
            new_img[r].append(tuple([int(p*255) for p in img[r][c]]))
    return new_img



def find_biggest_resolution(img_list):
    width_max = 0
    height_max = 0
    for im in img_list:
        w, h = im.size
        if w > width_max:
            width_max = w
        if h > height_max:
            height_max = h
    return width_max, height_max


if __name__ == "__main__":

    RGB_DATA_FILE_NAME = 'rgb_data_file.txt'

    cherryPickedFolderPath = Path("./Images/cherrypicked")
    DATA_FOLDER_PATH = Path()  # Our image file

    image_list = load_data_from_folder(Path("./Images/cherrypicked"), Path("./Images/cherrypicked_gt"))
    print("Done loading images.")

    rgb_values = []
    for d in image_list:
        rgb_values.append((
            get_rgb_of_img(reformat_image(d[0])),
            get_rgb_of_img(reformat_image(d[1]))
        ))

    testing_set = []
    label_set = []
    for i in rgb_values:
        testing_set.append(normalize_img(i[0]))
        x = binary_convert(i[1])
        label_set.append(x)

    corrosion_model = CorrosionDetectionModel()
    corrosion_model.train_model(np.array(testing_set), np.array(label_set), use_cp=True)

    p_img = corrosion_model.predict_img(np.array(testing_set[:4]))

    # show_both_images(p_img[0], label_set[0],
    #                  img1_title="Predicted Img",
    #                  img2_title="Original Label Img",
    #                  subplot_title="Predicted Image and Labeled Image")

    show_three_images([testing_set[0], create_tuples(p_img[0]), rgb_values[0][1]],
                      ["Original Image", "Predicted Image From Model", "Correct Labeled Image"])

    show_three_images([testing_set[1], p_img[1], rgb_values[1][1]],
                      ["Original Image", "Predicted Image From Model", "Correct Labeled Image"])

    show_three_images([testing_set[2], p_img[2], rgb_values[2][1]],
                      ["Original Image", "Predicted Image From Model", "Correct Labeled Image"])





