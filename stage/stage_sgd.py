import torch
from torch import Tensor
from typing import (
    List,
    Dict,
    Iterable,
    Union,
    Any,
    Optional
)

from .my_optimizers import MyTorchOptimizer

# STAGE-SGD
class STAGE_SGD(MyTorchOptimizer):
    """
        args:
            params: 最適化するパラメータ
            lr: 学習率
            tau: sech()の調整用パラメータ
            eps: 最低保証学習率
    """
    def __init__(
            self,
            params: Iterable[Tensor],
            lr: float=1e-3,
            tau: float=1.0,
            eps: float=1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            tau=tau,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        """ 1step update """

        for group in self.param_groups:
            lr: float = group['lr']
            tau: float = group['tau']
            eps: float = group['eps']
            params: List[Tensor] = group['params']

            """ 各パラメータ事の処理 """
            for p in params:
                if p.grad is None:
                    continue

                # 勾配を計算
                g_t: Optional[Tensor] = p.grad

                # --- stage-sgd update ---

                # sechでスケーリング
                scale_g: Tensor = g_t * tau # 元の書き換えを防ぐため新しいTensorを作る
                scale: Tensor = 1.0 / torch.cosh(scale_g)

                # 勾配の最低保証
                # Tensorとスカラの比較に注意
                scale = torch.clamp(scale, min=eps)

                # 補正後の勾配
                tamed_g: Tensor = g_t * scale

                # パラメータ更新
                p.add_(tamed_g, alpha=-lr)