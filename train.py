import os
import glob
import torch
import pprint
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection

'''
    OCR DATASET CLASS
    Dataset Used = BanglaWriting
    Dataset Manual = https://arxiv.org/pdf/2011.07499.pdf
    Dataset Download Link - https://data.mendeley.com/datasets/r43wkvdk4w/1
'''

class OCRDataset(Dataset):
    
    def __init__(self, img_dir, targets):
        self.img_dir = img_dir
        self.targets = targets

    def __len__(self):
        return len(self.img_dir)
    
    def __getitem__(self, item):
        image = Image.open(self.img_dir[item])
        image = image.resize((128, 64), resample=Image.BILINEAR)

        targets = self.targets[item]

        image = np.array(image)
        image = np.stack((image,)*1, axis=-1)

        # Reshape to tensor format supported by Pytorch (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


# defining the model
class OCRModel(nn.Module):
    def __init__(self, num_chars):
        super(OCRModel, self).__init__()
        self.conv_1 = nn.Conv2d(1, 128, kernel_size=(3, 6), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.linear_1 = nn.Linear(1024, 64)  # 1024 = 64*16
        self.drop_1 = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        x = F.relu(self.conv_1(images))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)  # [bs, 64, 16, 29] (bs, c, h, w)
        
        x = x.permute(0, 3, 1, 2)  # bs, w, c, h
        x = x.view(bs, x.size(1), -1)
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        
        x, _ = self.gru(x)
        x = self.output(x)
        
        x = x.permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(x, 2).to(torch.float64)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    word_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("°")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp)
        word_preds.append(remove_duplicates(tp))
    return word_preds


# define train and test functions
def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for data in tk0:
            for key, value in data.items():
                data[key] = value.to("cuda" if torch.cuda.is_available() else "cpu")
            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
        return fin_preds, fin_loss / len(data_loader)


def visualize_sample(dataset, index):
    import matplotlib.pyplot as plt
    
    sample = dataset[index]
    npimg = sample['images'].numpy()
    print(f"Original shape: {npimg.shape}")
    
    # Change the orientation of the image to display
    npimg = np.transpose(npimg, (1, 2, 0)).astype(np.float32)
    print(f"Display shape: {npimg.shape}")
    
    plt.figure(figsize=(8, 4))
    plt.imshow(npimg.squeeze(), cmap='gray')
    plt.title(f"Sample image {index}")
    plt.axis('off')
    plt.show()


def main():
    filepath = './img'
    print('Training process started')
    
    # Load image files and prepare targets
    image_files = glob.glob(os.path.join(filepath, '*jpg'))
    if not image_files:
        print(f"No image files found in {filepath}. Check path or file extensions.")
        return
    
    # Extract target names from file paths (fixed to handle different path formats)
    targets_orig = []
    for img_file in image_files:
        parts = img_file.replace('\\', '/').split('/')
        # Get filename without extension
        filename = os.path.splitext(parts[-1])[0]
        # Get first part of filename (assuming format like "word123" or "word 123")
        target = filename.split(" ")[0]
        targets_orig.append(target)
    
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]
    
    # Encode targets
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    
    # Process targets character by character and encode them
    targets_enc = []
    for t in targets:
        # Transform each character and add 1 (to reserve 0 for blank in CTC)
        encoded = [lbl_enc.transform([c])[0] + 1 for c in t]
        targets_enc.append(encoded)
    
    # Add padding to make all sequences the same length
    maxlen = max(len(t) for t in targets_enc)
    print(f"Maximum target length: {maxlen}")
    
    # Create a padded array with proper shape from the beginning
    padded_targets = np.zeros((len(targets_enc), maxlen), dtype=np.int64)
    
    # Fill the array with actual values
    for i, t in enumerate(targets_enc):
        padded_targets[i, :len(t)] = t
    
    # Now padded_targets has a consistent shape and can be used for train/test split
    print(f"Total unique classes/characters: {len(lbl_enc.classes_)}")
    print(f"Shape of padded targets: {padded_targets.shape}")
    
    # Divide into train test 
    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        train_orig_targets,
        test_orig_targets,
    ) = model_selection.train_test_split(
        image_files, padded_targets, targets_orig, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_imgs)}, Testing samples: {len(test_imgs)}")
    
    # Loading images and their corresponding labels to train and test dataset
    train_dataset = OCRDataset(img_dir=train_imgs, targets=train_targets)
    test_dataset = OCRDataset(img_dir=test_imgs, targets=test_targets)
    
    # Defining the data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = OCRModel(len(lbl_enc.classes_))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    # Define number of epoch and start training
    num_epoch = 10
    for epoch in range(num_epoch):
        print(f"\nStarting Epoch {epoch+1}/{num_epoch}")
        train_loss = train_fn(model, train_loader, optimizer)
        valid_preds, test_loss = eval_fn(model, test_loader)
        valid_word_preds = []
        
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_enc)
            valid_word_preds.extend(current_preds)
        
        # Remove any excess predictions if they don't match the target count
        valid_word_preds = valid_word_preds[:len(test_orig_targets)]
        
        combined = list(zip(test_orig_targets, valid_word_preds))
        print("Sample predictions:")
        print(combined[:5])
        
        test_dup_rem = [remove_duplicates(c) for c in test_orig_targets]
        accuracy = metrics.accuracy_score(test_dup_rem, valid_word_preds)
        
        print(f"Epoch={epoch+1}, Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Accuracy={accuracy:.4f}")
        scheduler.step(test_loss)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'label_encoder': lbl_enc,
        'num_classes': len(lbl_enc.classes_)
    }, "ocr_model.pth")
    print("Model saved successfully as ocr_model.pth")


if __name__ == "__main__":
    main()