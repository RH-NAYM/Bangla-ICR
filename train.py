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








import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_hidden_size = hidden_size // 2
        
        self.query = nn.Linear(hidden_size, self.attention_hidden_size)
        self.key = nn.Linear(hidden_size, self.attention_hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.attention_hidden_size]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale.to(query.device)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        attention = torch.softmax(energy, dim=-1)
        
        output = torch.matmul(attention, V)
        
        return output, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels):
        super(SpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 16))
        
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 8 * 16, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)
        )
        
        # Initialize transformation parameters
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x):
        xs = self.localization(x)
        xs = self.adaptive_pool(xs)
        xs = xs.view(-1, 128 * 8 * 16)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            
    def forward(self, features):
        results = []
        last_inner = None
        
        # Process features in reverse order (from high to low resolution)
        for i, (feature, inner_block, layer_block) in enumerate(zip(
            reversed(features), reversed(self.inner_blocks), reversed(self.layer_blocks)
        )):
            if last_inner is None:
                inner_result = inner_block(feature)
            else:
                # Upsample and add
                inner_result = inner_block(feature)
                inner_result = inner_result + F.interpolate(
                    last_inner, size=inner_result.shape[-2:], mode='nearest'
                )
                
            last_inner = inner_result
            results.insert(0, layer_block(inner_result))
            
        return results

class ComplexOCRModel(nn.Module):
    def __init__(self, num_chars, input_channels=1, hidden_size=1024):
        super(ComplexOCRModel, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Spatial Transformer for input preprocessing
        self.stn = SpatialTransformer(input_channels)
        
        # Convolutional Backbone
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.res_layers = nn.ModuleList([
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 1024, stride=2),
            ResidualBlock(1024, 1024)
        ])
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([256, 512, 1024], 256)
        
        # Sequence encoder - now outputs hidden_size channels to match GRU input
        self.sequence_encoder = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, hidden_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Adaptive pooling to handle varying sequence lengths
        self.adaptive_height_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size)
        
        # Bidirectional GRU layers
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size//2, bidirectional=True, batch_first=True),
            nn.GRU(hidden_size, hidden_size//2, bidirectional=True, batch_first=True),
            nn.GRU(hidden_size, hidden_size//2, bidirectional=True, batch_first=True)
        ])
        
        # Self-attention mechanism
        self.self_attention = nn.ModuleList([
            Attention(hidden_size) for _ in range(4)
        ])
        
        # Fully connected layers for feature transformation
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        ])
        
        # Dropout layers
        self.dropout = nn.Dropout(0.5)
        
        # Character prediction head
        self.char_pred = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_chars + 1)  # +1 for blank in CTC
        )
        
        # Additional hidden layers to increase model size
        self.additional_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(30)
        ])
        
        # Large parameter tensors to increase model size (approximately 192MB)
        self.large_param1 = nn.Parameter(torch.randn(2000, 2000))
        self.large_param2 = nn.Parameter(torch.randn(2000, 2000))
        self.large_param3 = nn.Parameter(torch.randn(2000, 2000))
        self.large_param4 = nn.Parameter(torch.randn(2000, 2000))
        self.large_param5 = nn.Parameter(torch.randn(2000, 2000))
        self.large_param6 = nn.Parameter(torch.randn(2000, 2000))
        
    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        
        # Apply spatial transformer
        x = self.stn(images)
        
        # Extract features through convolutional backbone
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Apply residual blocks and collect features for FPN
        features = []
        for i, layer in enumerate(self.res_layers):
            x = layer(x)
            # Collect features for FPN at specific layers
            if i in [2, 4, 7]:  # After 256, 512, and 1024 channels
                features.append(x)
        
        # Apply FPN to get multi-scale features
        fpn_features = self.fpn(features)
        
        # Use the highest resolution feature map for sequence modeling
        x = fpn_features[0]
        
        # Apply sequence encoder
        x = self.sequence_encoder(x)
        
        # Collapse height dimension for sequence modeling
        x = self.adaptive_height_pool(x)
        x = x.squeeze(2)  # Remove height dimension
        x = x.permute(0, 2, 1)  # [bs, seq_len, channels]
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Apply GRU layers with residual connections
        gru_out = x
        for i, gru in enumerate(self.gru_layers):
            residual = gru_out
            gru_out, _ = gru(gru_out)
            gru_out = gru_out + residual
            
            # Apply self-attention after each GRU layer
            if i < len(self.self_attention):
                attn_out, _ = self.self_attention[i](gru_out, gru_out, gru_out)
                gru_out = gru_out + attn_out
                
            # Apply fully connected transformation
            if i < len(self.fc_layers):
                fc_out = self.fc_layers[i](gru_out)
                gru_out = gru_out + fc_out
                
            gru_out = self.dropout(gru_out)
        
        # Apply character prediction head
        x = self.char_pred(gru_out)
        
        # Prepare for CTC loss
        x = x.permute(1, 0, 2)  # [seq_len, bs, num_classes]
        
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
            
            # Use CTC loss
            loss = nn.CTCLoss(blank=0, reduction='mean')(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss
        
        return x, None
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
    
    
    
    
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
    model = ComplexOCRModel(len(lbl_enc.classes_))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    # Define number of epoch and start training
    num_epoch = 2
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
            }, "models/ocr_model_final.pth")
            print(f"New best accuracy: {accuracy:.4f}. Model saved as models/ocr_model_final.pth")
        
        scheduler.step(test_loss)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'label_encoder': lbl_enc,
        'num_classes': len(lbl_enc.classes_)
    }, "models/ocr_model_final.pth")
    print("Final model saved successfully as models/ocr_model_final.pth")


if __name__ == "__main__":
    main()