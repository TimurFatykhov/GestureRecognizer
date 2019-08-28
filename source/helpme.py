# файл со вспомогательными функциями

import PIL
import numpy as np
import torch
import glob
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dataset_dir = os.path.join('source', 'dataset')
x_name = 'X.npy'
y_name = 'Y.npy'

class GestureDataset(Dataset):
    def __init__(self, X, Y, trs=None):
        """
        Параметры:
        ----------
        - X: 

        - y: 

        - trs: torchvision.transforms
            transforms

        """
        self.X = X.copy()
        self.Y = Y.copy()
        self.trs = trs

        if self.trs is None:
            self.trs = transforms.ToTensor()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] # (3, 64, 64) -> (64, 64, 3)
        y = self.Y[idx]

        if len(x.shape) > 2: 
            # если изображение многоканальное
            if x.shape[0] == 3:
                # если первое пространство отведено для каналов,
                # то транспонируем тензор так, чтобы оно стало послденим пространством
                x = np.transpose(x, (1, 2, 0))
            elif x.shape[0] == 1:
                # если изображение черно-белое
                x = np.squeeze(x, 0)

        # конвертируем numpy array в PIL.Image
        img = PIL.Image.fromarray(x)
        x_tensor = self.trs(img)

        return x_tensor, y


def create_loader(X, y, batch_size, num_workers=0, shuffle=True, trs=None):
    gesture_dataset = GestureDataset(X, y, trs=trs)
    gesture_loader = DataLoader(gesture_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return gesture_loader


def get_gesture_dataset(size=(100, 100), shuffle=True, gray_scale=False):
    """
    Возвращает два объекта: тензор с изображениями жестов
    и вектор классов.

    Параметры:
    ----------
    - size: tuple (default: (64, 64))
        Желаемый размер картинок

    - shuffle: bool (default: True)
        Если True, тогда данные вернутся перемешанными

    Возвращает:
    -----------
    - X: numpy ndarray
        Тензор с изображениями

    - y: numpy array 
        Вектор ответов (с классами)
    
    - gray_scale: bool
        Если True, то изображения буду конвертированны в черно-белое
    """
    cwd = os.getcwd()
    to_dataset_path = os.path.join(cwd, dataset_dir)
    to_X = os.path.join(to_dataset_path, x_name)
    to_Y = os.path.join(to_dataset_path, y_name)
    X = np.load(to_X)
    y = np.load(to_Y)
    idxs = np.arange(len(X))

    if shuffle:
        np.random.seed(17)
        np.random.shuffle(idxs)

    # меняем размер если необходимо
    if size != (100, 100):
        _X = []
        for pix in X:
            img = PIL.Image.fromarray(pix)
            img = img.resize(size, PIL.Image.BICUBIC)
            _X.append(np.asarray(img))

        X = np.stack(_X)

    # транспонируем тензор (BS, H, W, C) -> (BS, C, H, W)
    X = np.transpose(X, (0, 3, 1, 2))

    returns = X[idxs], y[idxs]

    if gray_scale:
        returns = X[idxs][:, :1, :, :], y[idxs]
    
    return returns


def load_imgs_from_folder(path, size=(64, 64), gray_scale=False):
    """
    Загружает все изображения из указанной папки и возвращает их в виде
    numpy тензора.

    Параметры:
    ----------
    - path: str
        Полный путь к папке, например: /home/user/Descktop

    - path: tuple (default: (64, 64))
        Размер, до которого требуется ужать изображение, например (64, 64)

    - gray_scale: bool (default: True)
        Если True, тогда вернет черно-белые изображения, иначе в исходдном виде.

    Возвращает:
    -----------
    - X: numpy ndarray
        Тензор с изображениями
    """
    to_imgs = glob.glob(os.path.join(path, '*'))

    pixel_imgs = []
    for to_img in to_imgs:
        img = PIL.Image.open(to_img)
        width, height  = img.size

        diff = height - width
        half = diff // 2


        if diff < 0:
            # width > height
            img = img.crop((-half, 0, width + diff - half, height))
        elif diff > 0:
            # heigth > width
            img = img.crop((0, half, width, height - diff + half))

        img = img.resize(size)
        pix = np.asarray(img)
        if gray_scale:
            pix = pix[..., :1]
            
        pixel_imgs.append(pix)
    
    return np.transpose(np.stack(pixel_imgs), (0, 3, 1, 2))


def calculate_pad(input_size, kernel_size, stride, output_size):
    """
    Вычисляет требуемый размер паддинга для сверточного слоя с
    учетом параметров:

    Параметры:
    ----------
    - input_size: int
        Входной пространственный размер тензора (картинки)

    - kernel_size: int
        Размер ядра свертки

    - stride: int
        Размер шага

    - output_size: int
        Желаемый выходной пространственный размер после слоя
    """
    pad = output_size * stride - input_size + kernel_size - 1
    pad /= 2
    if int(pad) != pad:
        print('ERROR at "calculate_pad"\nС такими параметрами нереально подобрать размер pad-а!')
    else:
        return int(pad)


def show_image(image, figsize=(5,5), title=''):
    """
    Показывает изображение
    
    Параметры:
    ----------
    - img: numpy.array
        массив numpy, с тремя или одним каналом (цветное или ч/б фото)
    """
    img = None

    if type(image) == torch.Tensor:
        # convert to numpy if tensor was passed
        img = image.numpy().copy()
    else:
        img = image.copy()

    if len(img.shape) < 2:
        s = np.sqrt(len(img)).astype(int)
        img = img.reshape((s,s))

    if img.shape[0] == 1:
        img = np.squeeze(img, 0)
    elif img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(str(title))
    plt.axis('off')
    plt.show()
    
def show_history(train_history, valid_history=None, hide_left=0, figsize=None, fontsize=30, title=None, width=4):
    if figsize is not None:
        plt.figure(figsize=figsize)
        
    N = len(train_history)
    plt.plot(np.arange(hide_left, N), train_history[hide_left:], color='blue', label='train', linewidth=width)
    
    if valid_history is not None:
        plt.plot(np.arange(hide_left, N), valid_history[hide_left:], color='green', label='val', linewidth=width)
        
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.show()