import dataclasses
from enum import Enum
from typing import Dict, Union

from src.data.augmentations import Compose, RandomNoise, RandomShift


class Mode(Enum):
    train = "train"
    eval = "eval"


# @dataclasses.datasclass()
# 이거는 일반 클래스를 데이터클래스로 변환해줌.


@dataclasses.dataclass()
class DatasetConfig:
    batch_size: int = 64
    num_workers: int = 8
    path: Dict = dataclasses.field(default_factory=lambda: {
        Mode.train: "/data/common/MIT-BIH/mitbih_train.csv",
        Mode.eval: "/data/common/MIT-BIH/mitbih_test.csv"
    })
    # dict타입으로 되어있는 transforms변수
    # Mode.train과 Mode.eval이라는 dict key를가짐. 이것은 각각 train과 val 모드에 대응하는 데이터 변환을 정의
    # Compose를 통해 여러 데이터 변환 함수를 하나의 변환 파이프라인으로 결합 / Compose에 전달된 리스트 내의 각 변환은 순서대로 데이터에 적용.
    # 리스트 내의 RandomNoise : 데이터에 noise를 줌, Random Shift : 데이터를 shift시킴
    # Mode.eval은 그대로 사용. 즉, validation과정에서는 데이터 augmentation과정을 진행 안함. 원본 데이터를 그대로 사용.
    transforms: Dict = dataclasses.field(default_factory=lambda: {
        Mode.train: Compose([RandomNoise(0.05, 0.5), RandomShift(10, 0.5)]), Mode.eval: lambda x: x})


@dataclasses.dataclass()
class ModelConfig:
    num_layers: int = 6
    signal_length: int = 187
    num_classes: int = 5
    input_channels: int = 1
    embed_size: int = 192
    num_heads: int = 8
    expansion: int = 4
    early_stoppint_epochs : int = 25


@dataclasses.dataclass()
class EcgConfig:
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    device: Union[int, str] = "cuda"
    lr: float = 2e-4
    num_epochs: int = 150

