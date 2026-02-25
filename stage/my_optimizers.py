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

class MyTorchOptimizer:

    defaults: Dict[str, Any]
    state: Dict[Tensor, Any]
    param_groups: List[Dict[str, Any]]

    def __init__(
            self,
            params: Iterable[Tensor],
            defaults: List[Dict[str, Any]],
    ) -> None:
        """
            params: 最適化するパラメータ model.parameters()
            defaults: 学習率などの初期設定 dict
        """
        # クラス変数の定義
        # デフォルト設定
        self.defaults = defaults
        # state（記憶領域）を初期化
        self.state = {}
        # ここにパラメータや設定を入れる
        self.param_groups = []


        # model.parameters() はイテレータなのでリストに変換
        param_list: List[Tensor] = list(params)

        # defaultとparamsを合体した辞書を作成
        single_group: Dict[str, Any] = {
            'params': param_list
        }
        single_group.update(defaults)

        self.param_groups.append(single_group)

    def zero_grad(self) -> None:
        """
            勾配をリセットする用のメソッド
            学習用関数でoptimizer.zero_grad()で使う
        """
        groups = self.param_groups
        for group in groups:

            params: List[Tensor] = group['params']
            for param in params:

                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

    def step(self) -> None:
        """
            パラメータの更新を行うメソッド
            子クラスのstepを呼び出す
        """
        raise NotImplementedError
