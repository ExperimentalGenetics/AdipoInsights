import numpy as np
from pathlib import Path
from PIL import Image
from skimage import io
from Analyzer.optimize.img_ops import load_mask

class Dataset:
    """
    Container for the image data. Images and segmentation masks are expected
    to be saved in data_path/*.tif and data_path/*_skin.png, respectively.
    """
    def __init__(self, data_path: Path, load_seg: bool = True):
        self.data_path = Path(data_path)
        self.data_files = sorted(self.data_path.glob("*.tif"))
        data = [self.load_data_sample(p) for p in self.data_files]
        if load_seg:
            self.seg_files = sorted(self.data_path.glob("*_skin.png"))
            seg = [self.load_seg_sample(p) for p in self.seg_files]
            self.data = [{'data': d, 'seg': s, 'data_name': str(dp), 'seg_name': str(sp)}
                         for d, s, dp, sp in zip(data, seg, self.data_files, self.seg_files)]
        else:
            self.data = [{'data': d, 'data_name': str(dp)} for d, dp in zip(data, self.data_files)]
    
    def load_data_sample(self, path: Path):
        data = np.array(Image.open(path), dtype=float)
        # Convert H x W x C to C x H x W if needed.
        if data.ndim == 3:
            data = np.transpose(data, (2, 0, 1))
        return data
    
    def load_seg_sample(self, path: Path):
        seg = load_mask(path)
        return seg[np.newaxis, :, :]
    
    def __getitem__(self, index: int):
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)

class LazyDataset:
    """
    A dataset that loads images (and optionally segmentation masks) on demand.
    """
    def __init__(self, data_path: Path, load_seg: bool = True, filestem: str = ''):
        self.data_path = Path(data_path)
        if filestem:
            self.data_files = sorted(self.data_path.glob(f"*{filestem}.tif"))
        else:
            self.data_files = sorted(self.data_path.glob("*.tif"))
        self.load_seg = load_seg
        if load_seg:
            self.seg_files = sorted(self.data_path.glob("*_wat.png"))

    def load_data_sample(self, path: Path) -> np.ndarray:
        data = np.array(Image.open(path), dtype=np.uint8)
        if data.ndim == 3:
            data = np.transpose(data, (2, 0, 1))  # Convert H x W x C to C x H x W
        return data

    def load_seg_sample(self, path: Path) -> np.ndarray:
        seg = load_mask(path)
        return seg[np.newaxis, :, :]

    def __getitem__(self, index: int) -> dict:
        sample = {"data": self.load_data_sample(self.data_files[index]),
                  "data_name": str(self.data_files[index])}
        if self.load_seg:
            sample["seg"] = self.load_seg_sample(self.seg_files[index])
            sample["seg_name"] = str(self.seg_files[index])
        return sample

    def __len__(self) -> int:
        return len(self.data_files)

class LazyDatasetS2(Dataset):
    """
    Dataset for on-demand loading of images (and segmentation masks if required).
    """
    def __init__(self, data_path, load_seg=True, filestem=''):
        self.data_path = Path(data_path)
        if filestem:
            self.data_files = sorted(self.data_path.glob(f"*{filestem}*.tif"))
        else:
            self.data_files = sorted(self.data_path.glob("*.tif"))
        
        self.load_seg = load_seg
        if load_seg:
            if filestem:
                self.seg_files = sorted(self.data_path.glob(f"*{filestem}*_wat.png"))
            else:
                self.seg_files = sorted(self.data_path.glob("*_wat.png"))
            print(f'Number of segmentation files: {len(self.seg_files)}')

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        image_path = self.data_files[index]
        image = io.imread(str(image_path))
        sample = {"data": image, "data_name": image_path.name}
        if self.load_seg:
            sample["seg"] = self.load_seg_sample(image_path.name)
        return sample

    def load_seg_sample(self, filename):
        _id = filename.rsplit('_', 1)[0]
        seg_wat = load_mask(self.data_path / f"{_id}_wat_post.png")
        seg_skin = load_mask(self.data_path / f"{_id}_wat.png")
        seg_bg = np.zeros_like(seg_skin)
        seg = np.argmax(np.stack([seg_bg, seg_wat, seg_skin]), axis=0)[None]
        return seg
