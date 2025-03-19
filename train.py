
'''
OCR DATASET CLASS
Dataset Used = BanglaWriting
Dataset Manual = https://arxiv.org/pdf/2011.07499.pdf
Dataset Download Link - https://data.mendeley.com/datasets/r43wkvdk4w/1



import Levenshtein
edit_distances = [Levenshtein.distance(t, p) for t, p in zip(test_orig_targets, valid_word_preds)]



import difflib
edit_distances = [sum(1 for _ in difflib.ndiff(t, p) if _.startswith('+ ') or _.startswith('- ')) for t, p in zip(test_orig_targets, valid_word_preds)]

    
'''


import os
import glob
import torch
import pprint
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Levenshtein

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection



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
        # Convert to grayscale if image is in color
        if len(image.shape) == 3 and image.shape[2] > 1:
            image = np.mean(image, axis=2).astype(np.uint8)
        
        # Ensure it's a single channel image
        image = np.expand_dims(image, axis=2)

        # Normalize image to [0,1] range
        image = image / 255.0

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
        x = self.pool_2(x)  # [bs, 64, 16, ?] (bs, c, h, w)
        
        x = x.permute(0, 3, 1, 2)  # bs, w, c, h
        x = x.view(bs, x.size(1), -1)
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        
        x, _ = self.gru(x)
        x = self.output(x)
        
        x = x.permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            
            # Calculate actual lengths of targets (excluding padding zeros)
            target_lengths = []
            for i in range(bs):
                count = 0
                for j in range(targets.size(1)):
                    if targets[i, j] != 0:
                        count += 1
                target_lengths.append(max(1, count))  # Ensure at least length 1
            
            target_lengths = torch.tensor(target_lengths, dtype=torch.int32)
            
            # Use targets directly - CTC loss expects a flattened tensor
            loss = nn.CTCLoss(blank=0, reduction='mean')(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None


def remove_duplicates(x):
    result = []
    prev_char = None
    
    for char in x:
        if char != prev_char:  # Only add if not the same as previous
            result.append(char)
            prev_char = char
    
    # Remove blank characters (represented by '°') and join
    return ''.join([c for c in result if c != '°'])


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    word_preds = []
    
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            if k == 0:  # This is the blank token in CTC
                temp.append('°')
            else:
                # Adjust index to match encoder (subtract 1)
                try:
                    p = encoder.classes_[k-1]
                    temp.append(p)
                except IndexError:
                    # Handle index errors gracefully
                    temp.append('°')
        
        # Join characters and remove duplicates
        tp = "".join(temp)
        cleaned = remove_duplicates(tp)
        word_preds.append(cleaned)
    
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
            if loss is not None:
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


def clean_target(text):
    # Remove non-letter characters and standardize for comparison
    # This step depends on your specific dataset
    return ''.join(c for c in text if c.isalnum()).lower()


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
    
    # Debug: Show original targets
    print("Sample original targets:", targets_orig[:5])
    
    # Apply preprocessing for consistency
    targets_clean = [clean_target(t) for t in targets_orig]
    
    # Split targets into characters
    targets = [[c for c in x] for x in targets_clean]
    targets_flat = [c for clist in targets for c in clist]
    
    # Debug: Show unique characters
    unique_chars = sorted(set(targets_flat))
    print(f"Unique characters in dataset: {unique_chars}")
    print(f"Total unique characters: {len(unique_chars)}")
    
    # Encode targets
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    
    # Debug: Show encoding mapping
    print("Character encoding mapping:")
    for i, c in enumerate(lbl_enc.classes_):
        print(f"  {c} -> {i+1}")  # +1 because 0 is reserved for blank
    
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
        image_files, padded_targets, targets_clean, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_imgs)}, Testing samples: {len(test_imgs)}")
    
    # Loading images and their corresponding labels to train and test dataset
    train_dataset = OCRDataset(img_dir=train_imgs, targets=train_targets)
    test_dataset = OCRDataset(img_dir=test_imgs, targets=test_targets)
    
    # Visualize a sample for debugging
    if len(train_dataset) > 0:
        print("\nVisualizing a training sample:")
        sample = train_dataset[0]
        print(f"Sample image shape: {sample['images'].shape}")
        print(f"Sample target: {sample['targets']}")
    
    # Defining the data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # Don't shuffle test data
    
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
    num_epoch = 10000
    best_accuracy = 0
    
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
        
        # Debug: Show some predictions vs targets
        print("\nSample predictions:")
        for i in range(min(5, len(test_orig_targets))):
            print(f"Target: '{test_orig_targets[i]}' → Pred: '{valid_word_preds[i]}'")
        
        # Preprocessing targets and predictions for fair comparison
        # For CTC, we need to compare without duplicates
        accuracy = metrics.accuracy_score(test_orig_targets, valid_word_preds)
        
        # Calculate character error rate
        total_chars = sum(len(t) for t in test_orig_targets)
        # edit_distances = [metrics.edit_distance(t, p) for t, p in zip(test_orig_targets, valid_word_preds)]
        edit_distances = [Levenshtein.distance(t, p) for t, p in zip(test_orig_targets, valid_word_preds)]

        cer = sum(edit_distances) / total_chars if total_chars > 0 else 1.0
        
        print(f"Epoch={epoch+1}, Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
        print(f"Accuracy={accuracy:.4f}, Character Error Rate={cer:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'accuracy': accuracy,
                'label_encoder': lbl_enc,
                'num_classes': len(lbl_enc.classes_)
            }, "ocr_model_best.pth")
            print(f"New best accuracy: {accuracy:.4f}. Model saved as ocr_model_best.pth")
        
        scheduler.step(test_loss)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'label_encoder': lbl_enc,
        'num_classes': len(lbl_enc.classes_)
    }, "ocr_model_final.pth")
    print("Final model saved successfully as ocr_model_final.pth")


if __name__ == "__main__":
    main()