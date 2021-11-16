import torchvision.transforms as transforms
import numpy as np

class ToNumpy:
    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8, copy=True)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        # np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        # np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img

def get_inference_scheme(input_size=224, transform_type='crop', val_zoom_factor=0.875,  is_prefetch=False):
    if transform_type == 'crop':
        inference_transform = transforms.Compose(
            [transforms.Resize(int(input_size / val_zoom_factor)),  # 256
             transforms.CenterCrop(input_size)])
    elif transform_type == 'squish':
        inference_transform = transforms.Compose([transforms.Resize((int(input_size),
                                                                     int(input_size)))])
    else:
        print("invalid transform_type {}".format(transform_type))
        exit(-1)

    if is_prefetch:
        inference_transform.transforms.append(ToNumpy())
    else:
        inference_transform.transforms.append(transforms.ToTensor())
    # inference_transform.transforms.append(transforms.ToTensor())

    return inference_transform
