import optuna
from optuna.trial import Trial
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import logging
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import yaml
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import optuna
from sklearn.metrics import precision_score

class DualPathwayObjectDetection(nn.Module):
    def __init__(self, num_classes, num_anchors=7):
        super(DualPathwayObjectDetection, self).__init__()
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.output_channels_yolo = num_anchors * (num_classes + 5)
        self.output_channels_ssd = num_anchors * (num_classes + 4)
        
        self.yolo_pathway = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels_yolo, kernel_size=1)
        )
        
        self.ssd_pathway = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels_ssd, kernel_size=1)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2)
      
        features = self.feature_extractor(x)
        yolo_output = self.yolo_pathway(features)

        batch_size = x.size(0)
        H, W = yolo_output.size(2), yolo_output.size(3)

        yolo_output = yolo_output.permute(0, 2, 3, 1).contiguous()
        yolo_output = yolo_output.view(batch_size, H, W, self.num_anchors, self.num_classes + 5)
        yolo_output = yolo_output.permute(0, 3, 4, 1, 2).contiguous()  
        
        box_coords = yolo_output[:, :, :4]  
        objectness = yolo_output[:, :, 4:5]  
        class_pred = yolo_output[:, :, 5:]  

        output = torch.cat([box_coords, objectness, class_pred], dim=2)
        
        return output
    
def compute_loss(self, predictions, targets, boxes, valid_boxes_mask):
    """
    Compute the combined loss for object detection
    Args:
        predictions: shape [batch_size, num_anchors, (4 + 1 + num_classes), H, W]
        targets: shape [batch_size, max_boxes]
        boxes: shape [batch_size, max_boxes, 4]
        valid_boxes_mask: shape [batch_size, max_boxes]
    """
    batch_size = predictions.size(0)
  
    pred_boxes = predictions[:, :, :4] 
    pred_obj = predictions[:, :, 4]   
    pred_cls = predictions[:, :, 5:]
    box_loss = torch.tensor(0.0, device=self.device)
    obj_loss = torch.tensor(0.0, device=self.device)
    cls_loss = torch.tensor(0.0, device=self.device)
    total_valid_boxes = 0
    
    valid_boxes_mask = valid_boxes_mask.bool()
    
    for b in range(batch_size):
        valid_mask = valid_boxes_mask[b]
        valid_boxes = boxes[b][valid_mask]
        valid_targets = targets[b][valid_mask]
        
        if valid_boxes.numel() == 0:
            continue
        
        batch_pred_boxes = pred_boxes[b].permute(0, 2, 3, 1).contiguous().view(-1, 4)
        batch_pred_obj = pred_obj[b].contiguous().view(-1)
        batch_pred_cls = pred_cls[b].permute(0, 2, 3, 1).contiguous().view(-1, pred_cls.size(2))
        
        try:
            ious = box_iou(batch_pred_boxes, valid_boxes)           
            best_ious, best_n = ious.max(dim=0)  
            
            box_loss += F.mse_loss(
                batch_pred_boxes[best_n],
                valid_boxes,
                reduction='sum'
            )
            
            obj_targets = torch.zeros_like(batch_pred_obj)
            obj_targets[best_n] = 1.0
            obj_loss += F.binary_cross_entropy_with_logits(
                batch_pred_obj,
                obj_targets,
                reduction='sum'
            )
            
            cls_loss += F.cross_entropy(
                batch_pred_cls[best_n],
                valid_targets,
                reduction='sum'
            )
            
            total_valid_boxes += len(valid_boxes)
            
        except RuntimeError as e:
            print(f"Error in batch {b}:")
            print(f"Pred boxes shape: {batch_pred_boxes.shape}")
            print(f"Valid boxes shape: {valid_boxes.shape}")
            print(f"Valid targets shape: {valid_targets.shape}")
            raise e
    
    eps = 1e-6
    if total_valid_boxes > 0:
        box_loss = box_loss / total_valid_boxes
        obj_loss = obj_loss / total_valid_boxes
        cls_loss = cls_loss / total_valid_boxes
    else:
        box_loss = box_loss * 0
        obj_loss = obj_loss * 0
        cls_loss = cls_loss * 0
    
    total_loss = box_loss * 5.0 + obj_loss * 1.0 + cls_loss * 1.0
    
    return total_loss, {
        'box_loss': box_loss.item(),
        'obj_loss': obj_loss.item(),
        'cls_loss': cls_loss.item(),
        'total_loss': total_loss.item()
    }

def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes
    Args:
        box1: [N, 4] in xywh format
        box2: [M, 4] in xywh format
    Returns:
        IoU matrix of shape [N, M]
    """
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
 
    x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area.unsqueeze(1) + b2_area - intersection

    iou = intersection / (union + 1e-16)
    return iou

class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.study = optuna.create_study(direction="minimize")
        self.optimize_hyperparameters()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.study.best_params["lr"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1, verbose=True
        )
    
    def compute_loss(self, predictions, targets, boxes, valid_boxes_mask):
        """
        Compute the combined loss for object detection
        Args:
            predictions: shape [batch_size, num_anchors, (4 + 1 + num_classes), H, W]
            targets: shape [batch_size, max_boxes]
            boxes: shape [batch_size, max_boxes, 4]
            valid_boxes_mask: shape [batch_size, max_boxes]
        """
        batch_size = predictions.size(0)
        pred_boxes = predictions[:, :, :4]  
        pred_obj = predictions[:, :, 4]     
        pred_cls = predictions[:, :, 5:]    
        
        box_loss = torch.tensor(0.0, device=self.device)
        obj_loss = torch.tensor(0.0, device=self.device)
        cls_loss = torch.tensor(0.0, device=self.device)
        total_valid_boxes = 0
    
        valid_boxes_mask = valid_boxes_mask.bool()
   
        for b in range(batch_size):
            valid_mask = valid_boxes_mask[b]
            valid_boxes = boxes[b][valid_mask]
            valid_targets = targets[b][valid_mask]
            
            if valid_boxes.numel() == 0:
                continue
            batch_pred_boxes = pred_boxes[b].permute(0, 2, 3, 1).contiguous().view(-1, 4)
            batch_pred_obj = pred_obj[b].contiguous().view(-1)
            batch_pred_cls = pred_cls[b].permute(0, 2, 3, 1).contiguous().view(-1, pred_cls.size(2))
            
            try:
                ious = box_iou(batch_pred_boxes, valid_boxes) 
                best_ious, best_n = ious.max(dim=0)
                box_loss += F.mse_loss(
                    batch_pred_boxes[best_n],
                    valid_boxes,
                    reduction='sum'
                )
                
                obj_targets = torch.zeros_like(batch_pred_obj)
                obj_targets[best_n] = 1.0
                obj_loss += F.binary_cross_entropy_with_logits(
                    batch_pred_obj,
                    obj_targets,
                    reduction='sum'
                )

                cls_loss += F.cross_entropy(
                    batch_pred_cls[best_n],
                    valid_targets,
                    reduction='sum'
                )
                
                total_valid_boxes += len(valid_boxes)
                
            except RuntimeError as e:
                print(f"Error in batch {b}:")
                print(f"Pred boxes shape: {batch_pred_boxes.shape}")
                print(f"Valid boxes shape: {valid_boxes.shape}")
                print(f"Valid targets shape: {valid_targets.shape}")
                raise e
        
        eps = 1e-6
        if total_valid_boxes > 0:
            box_loss = box_loss / total_valid_boxes
            obj_loss = obj_loss / total_valid_boxes
            cls_loss = cls_loss / total_valid_boxes
        else:
            box_loss = box_loss * 0
            obj_loss = obj_loss * 0
            cls_loss = cls_loss * 0
        total_loss = box_loss * 5.0 + obj_loss * 1.0 + cls_loss * 1.0
        
        return total_loss, {
            'box_loss': box_loss.item(),
            'obj_loss': obj_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }

    def optimize_hyperparameters(self):
        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            return self.validate_with_hyperparameters(lr)
        
        self.study.optimize(objective, n_trials=5)
        print(f"Best hyperparameters: {self.study.best_params}")
    
    def validate_with_hyperparameters(self, lr):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
    
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch_idx, (images, boxes, targets, valid_boxes_mask) in enumerate(pbar):
                    images = images.to(self.device)
                    boxes = boxes.to(self.device)
                    targets = targets.to(self.device)
                    valid_boxes_mask = valid_boxes_mask.to(self.device)
                
                    predictions = self.model(images).requires_grad_()
                
                    loss, loss_dict = self.compute_loss(predictions, targets, boxes, valid_boxes_mask)
                
                    total_loss += loss.item()
                    pbar.set_postfix({
                        'val_loss': f'{loss.item():.4f}',
                        'avg_val_loss': f'{total_loss / (batch_idx + 1):.4f}'
                    })
    
        return total_loss / num_batches
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        all_preds = []
        all_targets = []
        all_valid_masks = []

        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, (images, boxes, targets, valid_boxes_mask) in enumerate(pbar):
                images = images.to(self.device)
                boxes = boxes.to(self.device)
                targets = targets.to(self.device)
                valid_boxes_mask = valid_boxes_mask.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(images).requires_grad_()

                loss, loss_dict = self.compute_loss(predictions, targets, boxes, valid_boxes_mask)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()

                # Collect predictions, targets, and valid masks
                all_preds.append(predictions.detach().cpu())
                all_targets.append(targets.detach().cpu())
                all_valid_masks.append(valid_boxes_mask.detach().cpu())

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
        precision = 0.0
        try:
            flat_preds_classes = []
            flat_target_classes = []
            debug_pred_sizes = []
            debug_target_sizes = []
            
            for preds, targets, valid_mask in zip(all_preds, all_targets, all_valid_masks):
                for i in range(preds.size(0)):
                    valid_indices = valid_mask[i].nonzero(as_tuple=False).squeeze()
                    if valid_indices.numel() == 0:
                        continue
                    batch_preds = preds[i]
                    batch_targets = targets[i]
 
                    if batch_preds.numel() > 0 and batch_targets.numel() > 0:
                        batch_pred_classes = batch_preds.argmax(dim=-1)
                        debug_pred_sizes.append(len(batch_pred_classes))
                        debug_target_sizes.append(len(batch_targets))

                        flat_preds_classes.append(batch_pred_classes)
                        flat_target_classes.append(batch_targets)
            
            if flat_preds_classes and flat_target_classes:
                flat_preds_classes = torch.cat(flat_preds_classes).numpy()
                flat_target_classes = torch.cat(flat_target_classes).numpy()
     
                print("\nDebug Information:")
                print("Prediction class sizes:", debug_pred_sizes)
                print("Target class sizes:", debug_target_sizes)
                print("Final flat_preds_classes shape:", flat_preds_classes.shape)
                print("Final flat_target_classes shape:", flat_target_classes.shape)
               
                if len(flat_preds_classes) > 0 and len(flat_target_classes) > 0:
                    min_length = min(len(flat_preds_classes), len(flat_target_classes))
                    flat_preds_classes = flat_preds_classes[:min_length]
                    flat_target_classes = flat_target_classes[:min_length]
                    
                    precision = precision_score(flat_target_classes, flat_preds_classes, average='macro')
                else:
                    precision = 0.0
            else:
                precision = 0.0

        except Exception as e:
            print(f"Precision calculation error: {e}")
            precision = 0.0

        return total_loss / num_batches, precision
        
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch_idx, (images, boxes, targets, valid_boxes_mask) in enumerate(pbar):
                    images = images.to(self.device)
                    boxes = boxes.to(self.device)
                    targets = targets.to(self.device)
                    
                    predictions = self.model(images).requires_grad_()
                    
                    loss, loss_dict = self.compute_loss(predictions, targets, boxes, valid_boxes_mask)
                    
                    total_loss += loss.item()
                    pbar.set_postfix({
                        'val_loss': f'{loss.item():.4f}',
                        'avg_val_loss': f'{total_loss / (batch_idx + 1):.4f}'
                    })
        
        return total_loss / num_batches
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        best_precision = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            train_loss, train_precision = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model.pth')
            
            if train_precision > best_precision:
                best_precision = train_precision
            
            print(f'Training Loss: {train_loss:.4f}, Training Precision: {train_precision:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Best Precision: {best_precision:.4f}')

class HyperparameterTuner:
    def __init__(self, train_dataset, val_dataset, device, num_classes):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.num_classes = num_classes
        self.best_trial = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("HyperparameterTuning")
        
    def objective(self, trial: Trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_int('batch_size', 4, 32, step=4),
            'num_anchors': trial.suggest_int('num_anchors', 3, 9, step=2),
            'box_loss_weight': trial.suggest_float('box_loss_weight', 1.0, 10.0),
            'obj_loss_weight': trial.suggest_float('obj_loss_weight', 0.1, 2.0),
            'cls_loss_weight': trial.suggest_float('cls_loss_weight', 0.1, 2.0),
            'optimizer_name': trial.suggest_categorical('optimizer_name', ['Adam', 'SGD', 'AdamW']),
            'scheduler_patience': trial.suggest_int('scheduler_patience', 2, 5),
            'scheduler_factor': trial.suggest_float('scheduler_factor', 0.1, 0.5)
        }
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        model = DualPathwayObjectDetection(
            num_classes=self.num_classes,
            num_anchors=params['num_anchors']
        ).to(self.device)
        
        if params['optimizer_name'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        elif params['optimizer_name'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
        else:  # AdamW
            optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=params['scheduler_patience'],
            factor=params['scheduler_factor'],
            verbose=True
        )
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            loss_weights={
                'box': params['box_loss_weight'],
                'obj': params['obj_loss_weight'],
                'cls': params['cls_loss_weight']
            },
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        num_evaluation_epochs = 5
        best_val_loss = float('inf')
        
        for epoch in range(num_evaluation_epochs):
            train_loss = trainer.train_epoch()
            val_loss = trainer.validate()
            
            trial.report(val_loss, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            scheduler.step(val_loss)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_loss
    
    def tune(self, n_trials=100, study_name="object_detection_optimization"):
        """
        Run hyperparameter tuning
        
        Args:
            n_trials: Number of trials to run
            study_name: Name of the study for saving results
        
        Returns:
            dict: Best hyperparameters found
        """
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        try:
            study.optimize(self.objective, n_trials=n_trials, timeout=None)
            
            self.best_trial = study.best_trial
            self.logger.info(f"\nBest trial:")
            self.logger.info(f"  Value: {study.best_trial.value:.4f}")
            self.logger.info(f"  Params: ")
            for key, value in study.best_trial.params.items():
                self.logger.info(f"    {key}: {value}")
            
            self.save_study_results(study, f"{study_name}_results.pkl")
            
            return study.best_trial.params
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise
    
    @staticmethod
    def save_study_results(study, filename):
        """Save study results to a file"""
        import joblib
        joblib.dump(study, filename)
    
    @staticmethod
    def load_study_results(filename):
        """Load study results from a file"""
        import joblib
        return joblib.load(filename)

def create_model_with_best_params(best_params, num_classes, device):
    """Create a model instance with the best hyperparameters"""
    model = DualPathwayObjectDetection(
        num_classes=num_classes,
        num_anchors=best_params['num_anchors']
    ).to(device)
    
    if best_params['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    elif best_params['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'], momentum=0.9)
    else:  # AdamW
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=best_params['scheduler_patience'],
        factor=best_params['scheduler_factor'],
        verbose=True
    )
    
    return model, optimizer, scheduler

def main():
    config = load_yaml()
    num_classes = len(config['names'])
    IMAGE_SIZE = 640
    
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
 
    train_dataset = CustomDataset('train', transform=train_transform)
    val_dataset = CustomDataset('valid', transform=val_transform)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tuner = HyperparameterTuner(train_dataset, val_dataset, device, num_classes)
    
    best_params = tuner.tune(n_trials=50)
    
    model, optimizer, scheduler = create_model_with_best_params(best_params, num_classes, device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
  
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_weights={
            'box': best_params['box_loss_weight'],
            'obj': best_params['obj_loss_weight'],
            'cls': best_params['cls_loss_weight']
        },
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    trainer.train(num_epochs=50)  
    return model, trainer.history 

def load_yaml(file_path='data.yaml'):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class CustomDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'images'
        self.label_dir = self.root_dir / 'labels'
        self.transform = transform
        self.image_files = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png'))
        
        self.config = load_yaml()
        self.class_names = self.config['names']
        self.num_classes = len(self.class_names)
    
    def __len__(self):
        return len(self.image_files)
    
    def parse_label_line(self, line):
        try:
            data = line.strip().split()
            if len(data) != 5:
                return None, None
            
            class_id = int(data[0])
            coords = list(map(float, data[1:]))
            
            if not all(0 <= x <= 1 for x in coords):
                return None, None
                
            return class_id, coords
        except (ValueError, IndexError):
            return None, None
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, coords = self.parse_label_line(line)
                    if class_id is not None and coords is not None:
                        if 0 <= class_id < self.num_classes:
                            boxes.append(coords)
                            labels.append(class_id)
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)
        
        
        
        if self.transform:
            try:
                transformed = self.transform(image=image, bboxes=boxes.numpy(), class_labels=labels.numpy())
                image = transformed['image']  # Now a torch tensor [C, H, W]
                
                if len(transformed['bboxes']) > 0:
                    boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                    labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros(0, dtype=torch.long)
            except Exception as e:
                print(f"Transform failed for image {img_path}: {str(e)}")
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.long)
                image = self.transform(image=image)['image']
        
        return image, boxes, labels

def custom_collate_fn(batch):
    images = []
    boxes = []
    labels = []
    valid_boxes_mask = []

    for image, box, label in batch:
        images.append(image)
        boxes.append(box)
        labels.append(label)
        valid_boxes_mask.append(torch.ones(box.shape[0], dtype=torch.bool))

    images = torch.stack(images, dim=0)
    boxes = torch.nn.utils.rnn.pad_sequence(boxes, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    valid_boxes_mask = torch.nn.utils.rnn.pad_sequence(valid_boxes_mask, batch_first=True, padding_value=False)

    return images, boxes, labels, valid_boxes_mask

def compute_output_shape(model):
    """
    Compute the output shape of the model using a dummy input
    Args:
        model: The DualPathwayObjectDetection model
    Returns:
        tuple: The shape of the model's output
    """
    model = model.cpu()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    model.eval()
    
    with torch.no_grad():
        try:
            output = model(dummy_input)
            return output.shape
        except Exception as e:
            print(f"Error computing output shape: {str(e)}")
            return None
        finally:
            model.train()

def main():
    config = load_yaml()
    num_classes = len(config['names'])
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    IMAGE_SIZE = 640
    
    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    val_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    train_dataset = CustomDataset('train', transform=train_transform)
    val_dataset = CustomDataset('valid', transform=val_transform)
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    for i in range(min(5, len(train_dataset))):
        image, boxes, labels = train_dataset[i]
        print(f"Sample {i}:")
        print("  Image shape:", image.shape)
        print("  Boxes shape:", boxes.shape)
        print("  Labels shape:", labels.shape)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualPathwayObjectDetection(num_classes=num_classes, num_anchors=7)
    
    output_shape = compute_output_shape(model)
    print(f"Model output shape: {output_shape}")
    
    model = model.to(device)
    
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train(NUM_EPOCHS)

if __name__ == '__main__':
    
    main()
