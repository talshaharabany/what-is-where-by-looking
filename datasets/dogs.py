from __future__ import print_function

from PIL import Image
from os.path import join
import os
import scipy.io
import numpy as np

import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir, list_files
from datasets.tfs import get_dog_transform
from tqdm import tqdm


class ImageLoader(data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'StanfordDogs'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 root,
                 train=True,
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 download=True):

        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self.main_annotations = []
        if self.cropped:
            for annotation, idx in split:
                bbox = self.get_boxes(join(self.annotations_folder, annotation))
                self.main_annotations.append([(annotation, bbox, idx)])
        else:
            self.main_annotations = [[(annotation, idx)] for annotation, idx in split]

            # self._flat_breed_images = self._breed_images

        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]
        self.text_class = ['dog', 'floor', 'sofa', 'couch', 'bed', 'garden', 'wall', 'building', 'person', 'human']
        # self.text_class = ['dog', 'background']


    def __len__(self):
        return len(self.main_annotations)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name = self.main_annotations[index][0][0] + '.jpg'
        image_path = join(self.images_folder, image_name)
        # img = Image.open(image_path)
        img = Image.open(image_path).convert('RGB')
        image_sizes = (img.width, img.height)
        if self.transform:
            image = self.transform(img)
        if not self.train:
            out_box = []
            for box in self.main_annotations[index][0][1]:
                x1, y1, x2, y2 = box
                resize = 256
                shift = 16
                max_value = 223
                image_width, image_height = image_sizes
                left_bottom_x = np.maximum(x1 / image_width * resize - shift, 0)
                left_bottom_y = np.maximum(y1 / image_height * resize - shift, 0)
                right_top_x = np.minimum(x2 / image_width * resize - shift, max_value)
                right_top_y = np.minimum(y2 / image_height * resize - shift, max_value)
                out_box.append([left_bottom_x, left_bottom_y, right_top_x, right_top_y])
            return image, 0, out_box
        else:
            return image, 0

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self._flat_breed_images), len(counts.keys()), float(len(self._flat_breed_images))/float(len(counts.keys()))))

        return counts


def get_dogs_dataset(is_bbox=False, size=224):
    datadir = '/path_to_data/dogs'
    transform_train, transform_test = get_dog_transform(size)
    ds_train = ImageLoader(datadir, train=True, cropped=False, transform=transform_train)
    ds_test = ImageLoader(datadir, train=False, cropped=True, transform=transform_test)
    return ds_train, ds_test


def uni_bbox(buffer):
    out = {}
    for item in tqdm(buffer):
        (annotation, box, _) = item[0]
        bbox_list = []
        if annotation in out.keys():
            continue
        else:
            for item_new in buffer:
                (annotation_new, box_new, _) = item_new[0]
                if annotation == annotation_new:
                    bbox_list.append(box_new)
        bbox_list.append(box)
        out[str(annotation)] = bbox_list
    return out


if __name__ == "__main__":
    import cv2
    import torch
    import numpy as np

    ds_train, ds_test = get_dogs_dataset(is_bbox=True)
    ds = torch.utils.data.DataLoader(ds_train,
                                     batch_size=16,
                                     num_workers=8,
                                     shuffle=False,
                                     drop_last=False)
    pbar = tqdm(ds)
    for i, (real_imgs, labels) in enumerate(pbar):
        pass
    ds = torch.utils.data.DataLoader(ds_test,
                                     batch_size=1,
                                     num_workers=1,
                                     shuffle=False,
                                     drop_last=False)
    pbar = tqdm(ds)
    for i, (real_imgs, labels, bbox_list) in enumerate(pbar):
        pass
        # image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # image = (image - image.min()) / (image.max() - image.min())
        # image = np.array(255*image).copy().astype(np.uint8)
        # for bbox in bbox_list:
        #     gxa = int(bbox[0])
        #     gya = int(bbox[1])
        #     gxb = int(bbox[2])
        #     gyb = int(bbox[3])
        #     image = cv2.rectangle(image, (gxa, gya), (gxb, gyb), (0, 0, 255), 2)
        # cv2.imwrite('check/' + str(i) + '_' + name[0].split('/')[1] + '.jpg', image)

