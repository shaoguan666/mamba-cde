# Sepsis预测模型AUC提升方案

## 当前问题诊断

### 训练日志分析
- **测试集指标**:
  - 准确率: 0.8836 (高)
  - AUROC: 0.7321 (低)
  - Average Precision: 0.1450 (非常低)

### 核心问题
1. **类别不平衡严重**: 高准确率+低AUROC说明模型倾向预测负类
2. **训练不稳定**: Epoch 20准确率骤降至0.304
3. **模型选择策略错误**: 基于accuracy而非AUROC选择最佳模型(第163行)
4. **过早停止**: 基于training accuracy早停(第182-185行)

---

## 改进方案（按优先级）

### 🔥 Priority 1: 修改模型选择和早停策略

**影响**: 最大 | **实施难度**: 简单 | **预期AUROC提升**: +0.05~0.10

#### 修改点1: 基于AUROC选择最佳模型
**位置**: `common.py` 第163-166行

```python
# 当前代码(错误):
if val_metrics.accuracy > best_val_accuracy:
    best_val_accuracy = val_metrics.accuracy
    del best_model
    best_model = copy.deepcopy(model)

# 修改为:
if num_classes == 2:  # 二分类任务
    if val_metrics.auroc > best_val_auroc:
        best_val_auroc = val_metrics.auroc
        del best_model
        best_model = copy.deepcopy(model)
else:  # 多分类任务
    if val_metrics.accuracy > best_val_accuracy:
        best_val_accuracy = val_metrics.accuracy
        del best_model
        best_model = copy.deepcopy(model)
```

#### 修改点2: 基于AUROC早停
**位置**: `common.py` 第116-117行和第182-185行

```python
# 在第116-117行添加:
best_val_auroc = 0
best_val_auroc_epoch = 0

# 在第163行后添加:
if num_classes == 2 and val_metrics.auroc > best_val_auroc:
    best_val_auroc = val_metrics.auroc
    best_val_auroc_epoch = epoch

# 修改第182-185行的早停条件:
# 对于二分类任务，使用AUROC而非accuracy
if num_classes == 2:
    if epoch > best_val_auroc_epoch + plateau_terminate:
        tqdm_range.write('Breaking because of no improvement in val AUROC for {} epochs.'
                         ''.format(plateau_terminate))
        breaking = True
else:
    if epoch > best_train_accuracy_epoch + plateau_terminate:
        tqdm_range.write('Breaking because of no improvement in training accuracy for {} epochs.'
                         ''.format(plateau_terminate))
        breaking = True
```

#### 修改点3: 打印AUROC信息
**位置**: `common.py` 第168-171行

```python
# 对于二分类任务，打印AUROC
if num_classes == 2:
    tqdm_range.write('Epoch: {}  Train loss: {:.3}  Train accuracy: {:.3}  Train AUROC: {:.3}  '
                     'Val loss: {:.3}  Val accuracy: {:.3}  Val AUROC: {:.3}'
                     ''.format(epoch, train_metrics.loss, train_metrics.accuracy,
                              train_metrics.auroc, val_metrics.loss,
                              val_metrics.accuracy, val_metrics.auroc))
else:
    # 原有代码
    tqdm_range.write('Epoch: {}  Train loss: {:.3}  Train accuracy: {:.3}  Val loss: {:.3}  '
                     'Val accuracy: {:.3}'
                     ''.format(epoch, train_metrics.loss, train_metrics.accuracy,
                              val_metrics.loss, val_metrics.accuracy))
```

---

### 🔥 Priority 2: 优化类别不平衡处理

**影响**: 高 | **实施难度**: 中等 | **预期AUROC提升**: +0.03~0.08

#### 方案2.1: 动态调整pos_weight (推荐)

当前`pos_weight=10`可能不够。应该基于实际数据比例设置：

**位置**: `sepsis.py` 第24行

```python
# 计算实际的类别比例
def calculate_pos_weight(train_dataloader):
    """计算正负样本比例"""
    neg_count = 0
    pos_count = 0
    for batch in train_dataloader:
        *_, true_y, _ = batch
        neg_count += (true_y == 0).sum().item()
        pos_count += (true_y == 1).sum().item()
    ratio = neg_count / pos_count
    print(f"负样本数: {neg_count}, 正样本数: {pos_count}, 比例: {ratio:.2f}")
    return ratio

# 在main函数中:
# pos_weight = calculate_pos_weight(train_dataloader)
# pos_weight = torch.tensor(pos_weight)
```

然后在运行时传入：
```python
# sepsis.py 第62行
return common.main(name, times, train_dataloader, val_dataloader, test_dataloader, device,
                   new_make_model, num_classes, max_epochs, lr, kwargs,
                   pos_weight=pos_weight,  # 使用计算的pos_weight
                   step_mode=True)
```

#### 方案2.2: 实现Focal Loss (更激进)

**位置**: 创建新文件`experiments/losses.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Arguments:
        alpha: 正样本权重 (default: 0.25)
        gamma: focusing参数 (default: 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Arguments:
            logits: [N] 未经sigmoid的预测值
            targets: [N] 真实标签(0或1)
        """
        # 计算BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')

        # 计算pt
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # 计算alpha_t
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()
```

修改`common.py` 第233行：
```python
from losses import FocalLoss

# 在main函数中
if num_classes == 2:
    model = _SqueezeEnd(model)
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 旧代码
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)  # 新代码
```

---

### 🟡 Priority 3: 降低学习率

**影响**: 中 | **实施难度**: 简单 | **预期AUROC提升**: +0.02~0.05

**问题**: Epoch 20准确率骤降说明学习率过大

**位置**: `sepsis.py` 第30行

```python
# 当前:
lr = 0.0001 * (batch_size / 32)  # = 0.0032 (太大)

# 修改为:
lr = 0.0001 * (batch_size / 32) * 0.3  # = 0.001 (更稳定)
# 或者直接设置固定值:
lr = 0.0005  # 推荐
```

---

### 🟡 Priority 4: 增加训练轮数

**影响**: 中 | **实施难度**: 简单 | **预期AUROC提升**: +0.02~0.04

**问题**: 111个epoch就停止了，可能还没收敛

**位置**: `common.py` 第123行

```python
# 对于sepsis任务，增加plateau_terminate
if step_mode:
    epoch_per_metric = 10
    plateau_terminate = 150  # 从100改为150
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
```

或者在运行时指定更多epochs:
```bash
python run_sepsis.py --model ncde-spectral --epochs 300
```

---

### 🟢 Priority 5: 模型架构优化

**影响**: 中-低 | **实施难度**: 中等 | **预期AUROC提升**: +0.01~0.03

#### 方案5.1: 增加hidden_channels

**位置**: `run_sepsis.py` 第61-65行

```python
'ncde-spectral': {
    'hidden_channels': 128,  # 从64改为128
    'hidden_hidden_channels': None,
    'num_hidden_layers': None
},
```

#### 方案5.2: 调整SpectralModulatedVectorField参数

**位置**: `common.py` 第294-297行

```python
vector_field = models.SpectralModulatedVectorField(
    input_channels=input_channels,
    hidden_channels=hidden_channels,
    time_dim=64,  # 从32改为64，增加时间编码容量
    spectral_sigma=1.0  # 从2.0改为1.0，更激进的频域滤波
)
```

---

### 🟢 Priority 6: 数据增强/采样

**影响**: 中-低 | **实施难度**: 高 | **预期AUROC提升**: +0.01~0.05

#### 方案6.1: 过采样正样本
在`datasets/sepsis.py`中实现SMOTE或简单的重复采样

#### 方案6.2: 欠采样负样本
随机丢弃一部分负样本，使正负比例更平衡

---

## 实施建议

### Phase 1: 快速提升 (预期总提升: +0.10~0.18)
1. ✅ 修改模型选择策略为基于AUROC (Priority 1)
2. ✅ 降低学习率至0.0005 (Priority 3)
3. ✅ 计算并使用实际的pos_weight (Priority 2.1)

### Phase 2: 进一步优化 (预期总提升: +0.05~0.12)
4. 增加训练轮数至300 (Priority 4)
5. 尝试Focal Loss (Priority 2.2)
6. 增加hidden_channels至128 (Priority 5.1)

### Phase 3: 精细调优 (预期总提升: +0.02~0.08)
7. 调整模型超参数 (Priority 5.2)
8. 实施数据采样策略 (Priority 6)

---

## 实验记录模板

| 实验 | 修改内容 | AUROC | AP | 准确率 | 备注 |
|------|---------|-------|----|---------|----|
| Baseline | 原始配置 | 0.732 | 0.145 | 0.884 | |
| Exp-1 | 基于AUROC选择模型 | ? | ? | ? | |
| Exp-2 | Exp-1 + lr=0.0005 | ? | ? | ? | |
| Exp-3 | Exp-2 + 计算pos_weight | ? | ? | ? | |

---

## 预期最终效果

保守估计: **AUROC 0.80 ~ 0.85**
乐观估计: **AUROC 0.85 ~ 0.90**

关键是Priority 1的修改，这个修改能确保我们选择的模型真正在AUROC上表现最好，而不是在多数类预测上表现好。
