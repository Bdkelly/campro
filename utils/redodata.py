import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.cleanCall import cleanCall
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import numpy as np


class objectdata(Dataset):
    def __init__(self, images_dir, jsondata, class_names, transform):
        self.images_dir = images_dir
        self.jsondata = jsondata
        self.class_names = ['__background__'] + class_names # Ensure background class is at index 0
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"No image files found in {self.images_dir}. Cannot determine original image size.")

        if transform is None:
            self.transform = A.Compose([
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = transform
    def __len__(self):
        return len(self.jsondata)
    
    def __getitem__(self, idx):
        frame = self.jsondata[idx]
        fimg = frame["frame"]
        img_path = f"{self.images_dir}/{fimg}"

        boxes_for_frame = []
        labels_for_frame = []

        if not os.path.exists(img_path):
            print(f"Error: Image file not found for {img_path}. Returning dummy data.")
            # Create a black dummy image matching the original video's dimensions
            image = Image.new('RGB', (self.oVideo_width, self.oVideo_height), (0,0,0)) 
            # Leave boxes_for_frame and labels_for_frame empty
        else:
            image = Image.open(img_path).convert("RGB")
        
        xmin = frame['x_min']
        ymin = frame['y_min']
        xmax = frame['x_max']
        ymax = frame['y_max']
        label_name = "Ball"
        
        label_id = self.label_to_id.get(label_name)

        boxes_for_frame.append([xmin, ymin, xmax, ymax])
        labels_for_frame.append(label_id)

        boxes_tensor = torch.as_tensor(boxes_for_frame, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels_for_frame, dtype=torch.int64)

        target = {'boxes': boxes_tensor, 'labels': labels_tensor}

        image_np = np.array(image)

        if self.transform:
            transformed = self.transform(
                image=image_np, 
                bboxes=target['boxes'].cpu().numpy(), # Albumentations expects NumPy for bboxes
                labels=target['labels'].cpu().numpy() # Albumentations expects NumPy for labels
            )
            image = transformed['image'] # Transformed image (now a PyTorch tensor)
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        else:
            # If no transform was explicitly provided, apply a basic default to ensure consistency
            # (ToTensorV2 and Normalize are usually essential for model input)
            default_transform_pipeline = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            transformed = default_transform_pipeline(
                image=image_np, 
                bboxes=target['boxes'].cpu().numpy(), 
                labels=target['labels'].cpu().numpy()
            )
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)

        return image, target
    
if __name__ == "__main__":
    bondbox = "/Users/Ben/Documents/dever/python/ptorch/ball_tracking/inferred_images/mlsvideo/jdata/edited_jdata_100.json"
    imgdir = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/mlsvideo/imgs"

    box,width,height = cleanCall(bondbox,imgdir)
    class_name = ['Ball'] 
    
    transform_pipeline = A.Compose([
        A.Resize(640, 640), # Resize images to model input size
        A.HorizontalFlip(p=0.5), # Example augmentation
        A.RandomBrightnessContrast(p=0.2), # Example augmentation
        A.Normalize(mean=(0.485, 0.456, 0.406), # Standard ImageNet means
                    std=(0.229, 0.224, 0.225)),  # Standard ImageNet stds
        ToTensorV2() # Converts image to PyTorch tensor (HWC to CHW)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])) # Crucial for bounding box transforms

    dataset = objectdata(imgdir,box,class_name,transform_pipeline)
    dataloader = DataLoader(
                dataset, 
                batch_size=2, # Use a small batch size for testing
                shuffle=True, 
                num_workers=0, # Set to 0 for initial debugging to avoid multiprocessing issues
                collate_fn=lambda x: tuple(zip(*x)) # Custom collate_fn for object detection
            ) 

