import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch

def compute_metrics(preds, labels):
    """평가 메트릭 계산"""
    preds = np.argmax(preds, axis=1)
    labels = labels.cpu().numpy()
    
    accuracy = accuracy_score(labels, preds)
    report = classification_report(
        labels, 
        preds, 
        target_names=['happy', 'depressed', 'surprised', 'angry', 'neutral'],
        digits=4
    )
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    }

def get_gpu_memory_usage():
    """GPU 메모리 사용량 확인"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB 단위
    return 0 