from io import BytesIO

"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
from training.base_dataset import BaseDataset
import collections
import cv2

# class MultiResolutionDataset(Dataset):
#     def __init__(self, path, transform, resolution=256):
#         self.env = lmdb.open(
#             path,
#             max_readers=32,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )
#
#         if not self.env:
#             raise IOError("Cannot open lmdb dataset", path)
#
#         with self.env.begin(write=False) as txn:
#             self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))
#
#         self.resolution = resolution
#         self.transform = transform
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, index):
#         with self.env.begin(write=False) as txn:
#             key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
#             img_bytes = txn.get(key)
#
#         buffer = BytesIO(img_bytes)
#         img = Image.open(buffer)
#         img = self.transform(img)
#
#         return img


# class GTMaskDataset(Dataset):
#     def __init__(self, dataset_folder, transform, resolution=256):
#
#         self.env = lmdb.open(
#             f"{dataset_folder}/LMDB_test",
#             max_readers=32,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )
#
#         if not self.env:
#             raise IOError("Cannot open lmdb dataset", f"{dataset_folder}/LMDB_test")
#
#         with self.env.begin(write=False) as txn:
#             self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))
#
#         self.resolution = resolution
#         self.transform = transform
#
#         # convert filename to celeba_hq index
#         CelebA_HQ_to_CelebA = (
#             f"{dataset_folder}/local_editing/CelebA-HQ-to-CelebA-mapping.txt"
#         )
#         CelebA_to_CelebA_HQ_dict = {}
#
#         original_test_path = f"{dataset_folder}/raw_images/test/images"
#         mask_label_path = f"{dataset_folder}/local_editing/GT_labels"
#
#         with open(CelebA_HQ_to_CelebA, "r") as fp:
#             read_line = fp.readline()
#             attrs = re.sub(" +", " ", read_line).strip().split(" ")
#             while True:
#                 read_line = fp.readline()
#
#                 if not read_line:
#                     break
#
#                 idx, orig_idx, orig_file = (
#                     re.sub(" +", " ", read_line).strip().split(" ")
#                 )
#
#                 CelebA_to_CelebA_HQ_dict[orig_file] = idx
#
#         self.mask = []
#
#         for filename in os.listdir(original_test_path):
#             CelebA_HQ_filename = CelebA_to_CelebA_HQ_dict[filename]
#             CelebA_HQ_filename = CelebA_HQ_filename + ".png"
#             self.mask.append(os.path.join(mask_label_path, CelebA_HQ_filename))
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, index):
#         with self.env.begin(write=False) as txn:
#             key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
#             img_bytes = txn.get(key)
#
#         buffer = BytesIO(img_bytes)
#         img = Image.open(buffer)
#         img = self.transform(img)
#
#         mask = Image.open(self.mask[index])
#
#         mask = mask.resize((self.resolution, self.resolution), Image.NEAREST)
#         mask = transforms.ToTensor()(mask)
#
#         mask = mask.squeeze()
#         mask *= 255
#         mask = mask.long()
#
#         assert mask.shape == (self.resolution, self.resolution)
#         return img, mask
#
#
# class DataSetFromDir(Dataset):
#     def __init__(self, main_dir, transform):
#         self.main_dir = main_dir
#         self.transform = transform
#         all_imgs = os.listdir(main_dir)
#         self.total_imgs = []
#
#         for img in all_imgs:
#             if ".png" in img:
#                 self.total_imgs.append(img)
#
#     def __len__(self):
#         return len(self.total_imgs)
#
#     def __getitem__(self, idx):
#         img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
#         image = Image.open(img_loc).convert("RGB")
#         tensor_image = self.transform(image)
#         return tensor_image
#
#
# class DataSetTestLocalEditing(Dataset):
#     def __init__(self, main_dir, transform):
#         self.main_dir = main_dir
#         self.transform = transform
#
#         all_imgs = os.listdir(os.path.join(main_dir, "mask"))
#         self.total_imgs = []
#
#         for img in all_imgs:
#             if ".png" in img:
#                 self.total_imgs.append(img)
#
#     def __len__(self):
#         return len(self.total_imgs)
#
#     def __getitem__(self, idx):
#         image_mask = self.transform(
#             Image.open(
#                 os.path.join(self.main_dir, "mask", self.total_imgs[idx])
#             ).convert("RGB")
#         )
#         image_reference = self.transform(
#             Image.open(
#                 os.path.join(self.main_dir, "reference_image", self.total_imgs[idx])
#             ).convert("RGB")
#         )
#         # image_reference_recon = self.transform(Image.open(os.path.join(self.main_dir, 'reference_image', self.total_imgs[idx].replace('.png', '_recon_img.png'))).convert("RGB"))
#
#         image_source = self.transform(
#             Image.open(
#                 os.path.join(self.main_dir, "source_image", self.total_imgs[idx])
#             ).convert("RGB")
#         )
#         # image_source_recon = self.transform(Image.open(os.path.join(self.main_dir, 'source_image', self.total_imgs[idx].replace('.png', '_recon_img.png'))).convert("RGB"))
#
#         image_synthesized = self.transform(
#             Image.open(
#                 os.path.join(self.main_dir, "synthesized_image", self.total_imgs[idx])
#             ).convert("RGB")
#         )
#
#         return image_mask, image_reference, image_source, image_synthesized


class SwapTrainDataset(Dataset):
    def __init__(self, root, transform=None):
        super(SwapTrainDataset, self).__init__()
        self.root = root
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transform = transform

    def __getitem__(self, index):
        l = len(self.files)
        s_idx = index % l
        if index >= 4 * l:
            f_idx = s_idx

        else:
            f_idx = random.randrange(l)

        if f_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.files[f_idx])
        s_img = Image.open(self.files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)

        return f_img, s_img, same

    def __len__(self):
        return len(self.files) * 5


class SwapValDataset(Dataset):
    def __init__(self, root, transform=None):
        super(SwapValDataset, self).__init__()
        self.root = root
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transfrom = transform

    def __getitem__(self, index):
        l = len(self.files)

        f_idx = index // l
        s_idx = index % l

        if f_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.files[f_idx])
        s_img = Image.open(self.files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transfrom is not None:
            f_img = self.transfrom(f_img)
            s_img = self.transfrom(s_img)

        return f_img, s_img, same

    def __len__(self):
        return len(self.files) * len(self.files)


class SwapTestTxtDataset(Dataset):
    def __init__(self, root, root_txt, transform=None, suffix='.png'):
        super(SwapTestTxtDataset, self).__init__()
        self.root = root
        self.txt = root_txt
        f = open(root_txt)
        file_pair = [s.strip() for s in f.readlines()]
        self.file_trg = [root + s.replace(suffix, '').split('_')[0] + suffix for s in file_pair]
        self.file_src = [root + s.replace(suffix, '').split('_')[1] + suffix for s in file_pair]

        self.transform = transform

    def __getitem__(self, index):

        f_img = Image.open(self.file_trg[index])
        s_img = Image.open(self.file_src[index])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        f_img_n = self.file_trg[index].split('/')[-1].split('.')[0]
        s_img_n = self.file_src[index].split('/')[-1].split('.')[0]

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)

        return [f_img, f_img_n], [s_img, s_img_n]

    def __len__(self):
        return len(self.file_trg)


class Dataset_scale_trans(BaseDataset):
    def __init__(self, data_root, train=True, scale=True, transforms=None):
        self.data_root = data_root
        mode = 'test' if train else 'val'
        self.root = os.path.join(self.data_root, mode)

        videos = sorted(os.listdir(self.root))

        self.video_items, self.person_ids = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')

        self.person_ids_woaug = self.person_ids
        self.person_ids = self.person_ids * 100

        self.transforms = transforms
        self.scale = scale

    def default_loader(self, path):
        return Image.open(path).convert('RGB')

    def get_video_index(self, videos):
        video_items = []
        for video in videos:
            video_items.append(self.Video_Item(video))

        person_ids = sorted(list({video.split('#')[0] for video in videos}))
        return video_items, person_ids

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict

    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name
        video_item['person_id'] = video_name.split('#')[0]
        video_item['num_frame'] = [int(float(t[:4])) for t in os.listdir(os.path.join(self.root, video_name))]

        return video_item

    def random_select_frames(self, video_item, k):
        num_frame = video_item['num_frame']
        frame_idx = random.choices(num_frame, k=k)
        return frame_idx

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, index):
        # sample pairs
        person_id_s = self.person_ids[index]
        video_item_s = self.video_items[random.choices(self.idx_by_person_id[person_id_s], k=1)[0]]

        [frame_source_1, frame_source_2] = self.random_select_frames(video_item_s, 2)

        img_s1_path = os.path.join(self.root, video_item_s['video_name'], str(frame_source_1).zfill(4) + '.png')
        img_s2_path = os.path.join(self.root, video_item_s['video_name'], str(frame_source_2).zfill(4) + '.png')

        img_s1, img_s2 = self.default_loader(img_s1_path), self.default_loader(img_s2_path)
        if self.transforms:
            img_s1, img_s2_gt = self.transforms(img_s1), self.transforms(img_s2)

        if self.scale:
            img_s2_scale = self.transforms(self.aug(cv2.imread(img_s2_path), 256))

        return [img_s1, img_s2_gt, img_s2_scale], [img_s1_path, img_s2_path]


class Dataset_for_test(Dataset):
    def __init__(self, data_root, mode='test', root_txt='', suffix='.jpg', transforms=None):
        self.data_root = data_root
        self.root = os.path.join(self.data_root)

        f = open(root_txt)
        file_pair = [s.strip() for s in f.readlines()]
        self.file_pair = file_pair

        self.file_src = [self.root + '/' + s.replace(suffix, '').split('_')[0] + suffix for s in file_pair]
        self.file_drv = [self.root + '/' + s.replace(suffix, '').split('_')[1] + suffix for s in file_pair]

        self.transforms = transforms

    def default_loader(self, path):
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.file_src)

    def __getitem__(self, index):
        # sample pairs
        img_s_path = self.file_src[index]
        img_d_path = self.file_drv[index]

        img_s, img_d = self.default_loader(img_s_path), self.default_loader(img_d_path)
        if self.transforms:
            img_s, img_d = self.transforms(img_s), self.transforms(img_d)

        return [img_s, img_d], [img_s_path, img_d_path], self.file_pair[index]
