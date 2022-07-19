import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def face_augmentation(self, img, crop_size):
        img = self._color_transfer(img)
        img = self._reshape(img, crop_size)
        img = self._blur_and_sharp(img)
        return img

    def aug(self, img, crop_size):
        img = img[None]
        img = self.face_augmentation(img, crop_size)
        img = img[0]
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

    def _blur_and_sharp(self, img):
        blur = np.random.randint(0, 2)
        img2 = img.copy()
        output = []
        for i in range(len(img2)):
            if blur:
                ksize = np.random.choice([3, 5, 7, 9])
                output.append(cv2.medianBlur(img2[i], ksize))
            else:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                output.append(cv2.filter2D(img2[i], -1, kernel))
        output = np.stack(output)
        return output

    def _color_transfer(self, img):

        transfer_c = np.random.uniform(0.3, 1.6)

        start_channel = np.random.randint(0, 2)
        end_channel = np.random.randint(start_channel + 1, 4)

        img2 = img.copy()

        img2[:, :, :, start_channel:end_channel] = np.minimum(np.maximum(img[:, :, :, start_channel:end_channel] * transfer_c,
                                                                         np.zeros(img[:, :, :, start_channel:end_channel].shape)),
                                                                         np.ones(img[:, :, :, start_channel:end_channel].shape) * 255)
        return img2

    def perspective_transform(self, img, crop_size=224, pers_size=10, enlarge_size=-10):
        h, w, _ = img.shape
        dst = np.array([
            [-enlarge_size, -enlarge_size],
            [-enlarge_size + pers_size, w + enlarge_size],
            [h + enlarge_size, -enlarge_size],
            [h + enlarge_size - pers_size, w + enlarge_size], ], dtype=np.float32)
        src = np.array([[-enlarge_size, -enlarge_size], [-enlarge_size, w + enlarge_size],
                        [h + enlarge_size, -enlarge_size], [h + enlarge_size, w + enlarge_size]]).astype(np.float32())
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
        return warped, M

    def _reshape(self, img, crop_size):
        reshape = np.random.randint(0, 2)
        reshape_size = np.random.randint(15, 25)
        extra_padding_size = np.random.randint(0, reshape_size // 2)
        pers_size = np.random.randint(20, 30) * pow(-1, np.random.randint(2))

        enlarge_size = np.random.randint(20, 40) * pow(-1, np.random.randint(2))
        shape = img[0].shape
        img2 = img.copy()
        output = []
        for i in range(len(img2)):
            if reshape:
                im = cv2.resize(img2[i], (shape[0] - reshape_size*2, shape[1] + reshape_size*2))
                im = cv2.copyMakeBorder(im, 0, 0, reshape_size + extra_padding_size, reshape_size + extra_padding_size, cv2.cv2.BORDER_REFLECT)
                im = im[reshape_size - extra_padding_size:shape[0] + reshape_size + extra_padding_size, :, :]
                im, _ = self.perspective_transform(im, crop_size=crop_size, pers_size=pers_size, enlarge_size=enlarge_size)
                output.append(im)
            else:
                im = cv2.resize(img2[i], (shape[0] + reshape_size*2, shape[1] - reshape_size*2))
                im = cv2.copyMakeBorder(im, reshape_size + extra_padding_size, reshape_size + extra_padding_size, 0, 0, cv2.cv2.BORDER_REFLECT)
                im = im[:, reshape_size - extra_padding_size:shape[0] + reshape_size + extra_padding_size, :]
                im, _ = self.perspective_transform(im, crop_size=crop_size, pers_size=pers_size, enlarge_size=enlarge_size)
                output.append(im)
        output = np.stack(output)
        return output
