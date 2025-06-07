import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np

class EmotionDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=128, neutral_undersample_ratio=None, seed=42):
        self.data = pd.read_csv(data_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # 라벨 매핑
        self.label2id = {
            'happy': 0,
            'depressed': 1,
            'surprised': 2,
            'angry': 3,
            'neutral': 4
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        # 소프트라벨 → 하드라벨 변환
        if 'label' not in self.data.columns:
            label_cols = ['happy', 'depressed', 'surprised', 'angry', 'neutral']
            self.data['label'] = self.data[label_cols].values.argmax(axis=1)

        # neutral 언더샘플링
        if neutral_undersample_ratio is not None:
            np.random.seed(seed)
            neutral_idx = self.data[self.data['label'] == 4].index
            keep_n = int(len(neutral_idx) * neutral_undersample_ratio)
            keep_idx = np.random.choice(neutral_idx, keep_n, replace=False)
            non_neutral_idx = self.data[self.data['label'] != 4].index
            self.data = self.data.loc[keep_idx].append(self.data.loc[non_neutral_idx])
            self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        # 5차원 소프트라벨 벡터 반환
        label_vec = self.data.loc[idx, ['happy', 'depressed', 'surprised', 'angry', 'neutral']].values.astype(np.float32)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_vec, dtype=torch.float)
        }

def get_class_weights(dataset):
    """클래스 가중치 계산"""
    labels = dataset.data['label'].values
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights) 