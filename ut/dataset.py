from torch.utils.data import Dataset
import pandas as pd

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SportDataset(Dataset):
    def __init__(self, path, transform = None): 
        df = pd.read_csv(path, sep=',')

        self.img_paths = df['img_path']
        self.img_labels = df['label']

        self.transform = transform
       
        
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths.iloc[idx]
        image = Image.open(img_path).convert('RGB')

        label = self.img_labels.iloc[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
