from sklearn.model_selection import KFold
import os
import yaml
from pathlib import Path


def create_cv_folds(data_yaml = 'data/yolo-test.yaml', n_splits=5, random_state=42):
    folder_path = os.path.dirname(data_yaml)
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    images_dir = Path(f'{folder_path}/images')
    labels_dir = Path(f'{folder_path}/labels')
    all_data = []
    for img_path in images_dir.glob('*.*'):
        label_path = labels_dir / f"{img_path.stem}.txt"
        all_data.append({
            'image': str(img_path),
            'label': str(label_path)
        })
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds_data = []
    cwd = Path.cwd()
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        train_images = [all_data[i]['image'] for i in train_idx]
        val_images = [all_data[i]['image'] for i in val_idx]
        with open(f'train_{fold}.txt', 'w+') as file:
            file.writelines(str(Path(cwd, s)) + '\n' for s in train_images)
        with open(f'val_{fold}.txt', 'w+') as file:
            file.writelines(str(Path(cwd, s)) + '\n' for s in val_images)
        fold_config = data.copy()
        fold_config['train'] = f'train_{fold}.txt'
        fold_config['val'] = f'val_{fold}.txt'
        fold_yaml = f'data_{fold}.yaml'
        with open(fold_yaml, 'w+') as f:
            yaml.dump(fold_config, f)
        folds_data.append(fold_yaml)
    return folds_data