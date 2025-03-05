import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import defaultdict
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.BAM import BAMBlock
import fightingcv_attention.attention




# ==================== 配置参数 ====================
class Config:
    # 数据参数
    csv_path = "./output.csv"  # CSV格式：image_path, attr1, attr2..., attr8
    img_dir = "./Training_Dataset"
    attributes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']  # 8个属性
    num_classes = {'N': 2, 'D': 2, 'G': 2, 'C': 2,
                   'A': 2, 'H': 2, 'M': 2, 'O': 2}  # 每个属性的类别数
    # 训练参数
    batch_size = 32
    epochs = 30
    lr = 1e-4
    weight_decay = 1e-4  # L2正则化
    dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 图像处理
    img_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 输出设置
    plot_path = "training_metrics.png"
    model_save_path = "best_model.pth"


# ==================== 数据加载 ====================
class MultiAttrDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_weights = self._calculate_class_weights()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(Config.img_dir, self.df.iloc[idx]['all-Fundus'])
        image = Image.open(img_path).convert('RGB')
        # image = preprocess(image)  ##对图像进行预处理

        labels = {}
        for attr in Config.attributes:
            labels[attr] = self.df.iloc[idx][attr]

        if self.transform:
            image = self.transform(image)


        return image, labels

    def _calculate_class_weights(self):
        all_labels = self.df.iloc[:, 1:].values
        class_weights = []
        for i in range(all_labels.shape[1]):
            pos = np.sum(all_labels[:, i])  ##正样本
            neg = len(all_labels) - pos ##负样本
            weight = neg / (pos + 1e-7)  # 当正样本越少，即负样本越高，权重越高，被选中的概率越大
            class_weights.append(weight)
        return torch.tensor(class_weights, dtype=torch.float32)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(Config.mean, Config.std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(Config.mean, Config.std)
    ])
    return train_transform, val_transform


# 创建加权采样器解决样本失衡
def create_sampler(dataset):
    sample_weights = []
    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        labels_ = []
        for key in labels.keys():
            labels_.append(float(labels[key]))
        weight = sum(np.array(labels_) * dataset.class_weights.numpy()) + 1  # 基础权重
        sample_weights.append(weight)
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def prepare_loaders():
    all_df = pd.read_csv(Config.csv_path)
    train_df, val_df = train_test_split(all_df, test_size=0.1, random_state=42)

    train_transform, val_transform = get_transforms()

    train_ds = MultiAttrDataset(train_df, train_transform)
    val_ds = MultiAttrDataset(val_df, val_transform)
    train_sampler = create_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size,
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ==================== 模型定义 ====================

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        base_model = models.efficientnet_b4(pretrained=True)
        self.features=nn.Sequential(*list(base_model.children()))[:-2]

        # 注意力机制
        # self.se=CBAMBlock(channel=2048, reduction=16, kernel_size=7)
        # self.se=BAMBlock(channel=1792, reduction=16, dia_val=2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1792, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1792, 1),
            nn.Sigmoid()
        )


        self.header=nn.ModuleDict()
        for attr in Config.attributes:
            self.header[attr] = nn.Sequential(
                nn.Linear(1792, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(Config.dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Linear(256, 2)
            )
    def forward(self,x):
        x=self.features(x) # [B, 2048, 7, 7]
        x=self.se(x)
        x=torch.nn.functional.adaptive_avg_pool2d(x,(1,1)).flatten(1)

        outputs={}
        for attr in Config.attributes:
            outputs[attr]=self.header[attr](x)

        return outputs

# ==================== 训练工具 ====================
class MetricTracker:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.metrics = {
            'train': {attr: {'acc': [], 'prec': [], 'rec': [], 'f1': []}
                      for attr in Config.attributes},
            'val': {attr: {'acc': [], 'prec': [], 'rec': [], 'f1': []}
                    for attr in Config.attributes}
        }
        self.avg_metrics = {
            'train': {'acc': [], 'prec': [], 'rec': [], 'f1': []},
            'val': {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        }

    def update(self, phase, loss, metrics):
        getattr(self, f'{phase}_loss').append(loss)#动态获取train/val阶段的属性值
        for attr in Config.attributes:
            for m in ['acc', 'prec', 'rec', 'f1']:
                self.metrics[phase][attr][m].append(metrics[attr][m])

        # 计算平均指标
        for m in ['acc', 'prec', 'rec', 'f1']:
            avg_val = np.mean([metrics[attr][m] for attr in Config.attributes])
            self.avg_metrics[phase][m].append(avg_val)


# 损失函数
def get_criterion():

    class_weights = {
        'N': torch.tensor([1.0, 1.0]),
        'D': torch.tensor([0.2, 1.0]),
        'G': torch.tensor([0.025, 1.0]),
        'C': torch.tensor([0.1, 1.0]),
        'A': torch.tensor([0.1, 1.0]),
        'H': torch.tensor([0.1, 1.0]),
        'M': torch.tensor([0.05, 1.0]),
        'O': torch.tensor([0.25, 1.0])
    }

    criterion_dict = {}
    for attr in Config.attributes:
        # 将权重张量移动到指定设备
        class_weights[attr] = class_weights[attr].to(Config.device)
        criterion_dict[attr] = nn.CrossEntropyLoss(weight=class_weights[attr])
    return criterion_dict



def calculate_metrics(true_labels, pred_labels):
    metrics = {}
    for attr in Config.attributes:
        y_true = true_labels[attr]
        y_pred = pred_labels[attr]
        metrics[attr] = {
            'acc': accuracy_score(y_true, y_pred),
            'prec': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'rec': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
    return metrics


def plot_metrics(tracker):
    plt.figure(figsize=(18, 12))

    # Loss曲线
    plt.subplot(2, 2, 1)
    plt.plot(tracker.train_loss, label='Train')
    plt.plot(tracker.val_loss, label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy曲线
    plt.subplot(2, 2, 2)
    plt.plot(tracker.avg_metrics['train']['acc'], label='Train')
    plt.plot(tracker.avg_metrics['val']['acc'], label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Precision-Recall曲线
    plt.subplot(2, 2, 3)
    plt.plot(tracker.avg_metrics['train']['prec'], label='Train Precision')
    plt.plot(tracker.avg_metrics['train']['rec'], label='Train Recall')
    plt.plot(tracker.avg_metrics['val']['prec'], label='Val Precision')
    plt.plot(tracker.avg_metrics['val']['rec'], label='Val Recall')
    plt.title('Precision & Recall Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    # F1曲线
    plt.subplot(2, 2, 4)
    plt.plot(tracker.avg_metrics['train']['f1'], label='Train')
    plt.plot(tracker.avg_metrics['val']['f1'], label='Validation')
    plt.title('F1 Score Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(Config.plot_path)
    plt.close()


# ==================== 训练流程 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据
    train_loader, val_loader = prepare_loaders()

    # 初始化模型
    model = Model().to(device)

    loss_func = get_criterion()


    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    tracker = MetricTracker()
    best_f1 = 0

    for epoch in range(Config.epochs):
        # 训练阶段
        model.train()
        train_preds = defaultdict(list)
        train_labels = defaultdict(list)
        epoch_train_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)

            batch_labels = {key: value.to(device) for key, value in labels.items()}

            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算损失
            total_loss = 0
            for attr in Config.attributes:
                loss = loss_func[attr](outputs[attr], batch_labels[attr])
                total_loss += loss #该批次内所有样本的平均损失

            total_loss.backward()
            optimizer.step()

            # 收集预测结果a
            with torch.no_grad():
                for attr in Config.attributes:
                    _, preds = torch.max(outputs[attr], 1)
                    train_preds[attr].extend(preds.cpu().numpy())
                    train_labels[attr].extend(batch_labels[attr].cpu().numpy())

            epoch_train_loss += total_loss.item() * inputs.size(0)

        # 计算训练指标
        train_loss = epoch_train_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(train_labels, train_preds)
        tracker.update('train', train_loss, train_metrics)

        # 验证阶段
        model.eval()
        val_preds = defaultdict(list)
        val_labels = defaultdict(list)
        epoch_val_loss = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                batch_labels = {k: v.to(device) for k, v in labels.items()}

                outputs = model(inputs)

                # 计算损失
                total_loss = 0
                for attr in Config.attributes:
                    loss = loss_func[attr](outputs[attr], batch_labels[attr])
                    total_loss += loss
                epoch_val_loss += total_loss.item() * inputs.size(0)

                # 收集预测结果
                for attr in Config.attributes:
                    _, preds = torch.max(outputs[attr], 1)
                    val_preds[attr].extend(preds.cpu().numpy())
                    val_labels[attr].extend(batch_labels[attr].cpu().numpy())

        # 计算验证指标
        val_loss = epoch_val_loss / len(val_loader.dataset)
        val_metrics = calculate_metrics(val_labels, val_preds)
        tracker.update('val', val_loss, val_metrics)

        # 打印日志
        print(f"\nEpoch {epoch + 1}/{Config.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("=" * 60)
        # print(f"{'Attribute':<15} {'Acc(T/V)':<12} {'Prec(T/V)':<12} {'Rec(T/V)':<12} {'F1(T/V)':<12}")
        print(f"{'Attribute':<15} {'Acc(T/V)':<12} {'Prec(T/V)':<12} {'Rec(T/V)':<12}")
        acc_all_train = []
        prec_all_train = []
        rec_all_train = []
        # f1_all_train = []
        acc_all_test = []
        prec_all_test = []
        rec_all_test = []
        # f1_all_test = []
        for attr in Config.attributes:
            t = train_metrics[attr]
            v = val_metrics[attr]
            acc_all_train.append(t['acc'])
            prec_all_train.append(t['prec'])
            rec_all_train.append(t['rec'])
            # f1_all_train.append(t['f1'])
            acc_all_test.append(v['acc'])
            prec_all_test.append(v['prec'])
            rec_all_test.append(v['rec'])
            # f1_all_test.append(v['f1'])
            print(f"{attr:<15} {t['acc']:.3f}/{v['acc']:.3f}  {t['prec']:.3f}/{v['prec']:.3f}  "
                  f"{t['rec']:.3f}/{v['rec']:.3f} ")
        print(
            f"{'average':<15} {sum(acc_all_train) / len(acc_all_train):.3f}/{sum(acc_all_test) / len(acc_all_test):.3f}  {sum(prec_all_train) / len(prec_all_train):.3f}/{sum(prec_all_test) / len(prec_all_test):.3f}  "
            f"{sum(rec_all_train) / len(rec_all_train):.3f}/{sum(rec_all_test) / len(rec_all_test) :.3f} ")

        # 保存最佳模型
        current_f1 = np.mean([v['f1'] for v in val_metrics.values()])
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"\nSaved new best model with Val F1: {best_f1:.4f}")

        # 调整学习率
        scheduler.step(val_loss)

    # 绘制曲线
    plot_metrics(tracker)


if __name__ == "__main__":
    main()