import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import numpy as np
import wandb

from data.dataset import EmotionDataset, get_class_weights
from models.kobert_classifier import KoBERTEmotionClassifier
from utils.logger import setup_logger, setup_wandb
from utils.metrics import compute_metrics, get_gpu_memory_usage

def set_seed(seed):
    """시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(config):
    # 설정 로드
    with open(config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 로깅 설정
    logger = setup_logger(config['logging']['log_dir'])
    setup_wandb(config)
    
    # 시드 고정
    set_seed(config['training']['seed'])
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 데이터셋 로드
    train_dataset = EmotionDataset(
        config['data']['train_path'],
        config['model']['name'],
        config['model']['max_length'],
        neutral_undersample_ratio=config['data'].get('neutral_undersample_ratio', None),
        seed=config['training']['seed']
    )
    val_dataset = EmotionDataset(
        config['data']['val_path'],
        config['model']['name'],
        config['model']['max_length']
    )
    
    # 클래스 가중치 계산 (KLDivLoss에서는 사용하지 않음)
    # class_weights = get_class_weights(train_dataset)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # 데이터로더 설정
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size']
    )
    
    # 모델 초기화
    model = KoBERTEmotionClassifier(
        config['model']['name'],
        config['model']['num_labels'],
        config['model']['dropout']
    ).to(device)
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # 학습 루프
    best_val_acc = 0
    for epoch in range(config['training']['num_epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["training"]["num_epochs"]}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            log_probs = torch.log(outputs + 1e-8)  # log_softmax 대신 log(softmax)
            loss = criterion(log_probs, labels)
            
            loss = loss / config['training']['gradient_accumulation_steps']
            loss.backward()
            
            if (progress_bar.n + 1) % config['training']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # GPU 메모리 사용량 로깅
            if (progress_bar.n + 1) % 100 == 0:
                gpu_memory = get_gpu_memory_usage()
                logger.info(f'GPU Memory Usage: {gpu_memory:.2f} MB')
        
        # 검증
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = model(input_ids, attention_mask)
                val_preds.append(outputs.cpu().numpy())
                val_labels.append(labels)
        
        val_preds = np.concatenate(val_preds)
        val_labels = torch.cat(val_labels)
        
        # 평가: 소프트라벨 → argmax 하드라벨로 변환
        metrics = compute_metrics(val_preds, val_labels.argmax(dim=1))
        val_acc = metrics['accuracy']
        
        logger.info(f'Epoch {epoch + 1} - Validation Accuracy: {val_acc:.4f}')
        logger.info(f'Classification Report:\n{metrics["classification_report"]}')
        
        # 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(config['logging']['save_dir'], 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            logger.info(f'Best model saved to {save_path}')
        
        # wandb 로깅
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': total_loss / len(train_loader),
            'val_accuracy': val_acc,
            'gpu_memory': get_gpu_memory_usage()
        })

if __name__ == '__main__':
    train('config/config.yaml') 