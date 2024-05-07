import torch
from torch import nn, Tensor
from torch.nn import functional as F

class FeatureAlignerLossCOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        # 尺度调整参数，初始化为可学习的参数
        # self.scale_factor = nn.Parameter(torch.tensor([scale_factor]))

    def calculate_cosine_loss(self, tensor1, tensor2, scale_factor):
        """计算两个张量之间的余弦相似度损失，包括尺度调整和特征归一化。"""
        # 将[B, C, H, W]的张量重塑为[B, C*H*W]以便计算余弦相似度
        # print(tensor1.shape)
        tensor1_flat = tensor1.reshape(tensor1.size(0), -1)
        tensor2_flat = tensor2.reshape(tensor2.size(0), -1)
        
        # 特征归一化（L2归一化）
        tensor1_norm = F.normalize(tensor1_flat, p=2, dim=1)
        tensor2_norm = F.normalize(tensor2_flat, p=2, dim=1)
        
        # 计算余弦相似度
        cosine_sim = self.cosine_similarity(tensor1_norm, tensor2_norm)

        # 将相似度转换为损失（1 - 相似度），以便在相似度较低时损失较高
        cosine_loss = 1 - cosine_sim.mean()
        # print(cosine_loss.device)
        # 应用尺度调整
        scaled_cosine_loss = scale_factor * cosine_loss

        return scaled_cosine_loss

    def forward(self, list1, list2, scale_factor):
        """对两个列表中相对应位置的张量进行特征对齐，并计算余弦相似度损失的平均值，包括尺度调整和特征归一化。"""
        cosine_losses = []
        for tensor1, tensor2 in zip(list1, list2):
            cosine_loss = self.calculate_cosine_loss(tensor1, tensor2, scale_factor)
            cosine_losses.append(cosine_loss)
        # 计算余弦相似度损失的平均值
        average_cosine_loss = torch.mean(torch.stack(cosine_losses))
        return average_cosine_loss

class FeatureAlignerLossMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def calculate_mse_loss(self, tensor1, tensor2, scale_factor):
        """计算两个张量之间的均方误差损失，包括尺度调整。"""
        # 对于MSE，不需要将[B, C, H, W]的张量重塑为[B, C*H*W]，直接计算即可
        mse_loss = self.mse_loss(tensor1, tensor2)
        # 应用尺度调整
        scaled_mse_loss = scale_factor * mse_loss
        return scaled_mse_loss

    def forward(self, list1, list2, scale_factor=1.0):
        """对两个列表中相对应位置的张量进行特征对齐，并计算MSE损失的平均值，包括尺度调整。"""
        mse_losses = []
        for tensor1, tensor2 in zip(list1, list2):
            mse_loss = self.calculate_mse_loss(tensor1, tensor2, scale_factor)
            mse_losses.append(mse_loss)
        # 计算MSE损失的平均值
        average_mse_loss = torch.mean(torch.stack(mse_losses))
        return average_mse_loss
              

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]

        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min: # 找出前几个之后求平均值
            loss_hard, _ = loss.topk(n_min)

        loss = torch.mean(loss_hard)
        return loss

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)