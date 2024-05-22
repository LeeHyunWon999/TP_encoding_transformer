import torch


#
class Compose:

    def __init__(self, transforms) -> None:
        self.transforms = transforms

    # 객체가 함수처럼 호출될 때 동작하는 메소드 
    # transform에 저장된 변환을 순차적으로 적용.
    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal


class RandomShift:

    # max_num_sample을 통해 신호가 이동할 수 있는 최대 샘플 수 정의
    # probability를 통해 실제로 shift가 적용될 확률 정의. (0~1의 값으로 default는 0.5)
    # 이 두개를 파라미터로 받아와 _m과 _p에 적용
    def __init__(self, max_num_samples: int, probability=0.5) -> None:
        self._m = max_num_samples
        self._p = probability

    # 매개변수로 ECG신호인 signal을 받음
    def __call__(self, signal):
        
        # torch.rand(1)을 통해 0~1 사이의 수를 하나 생성하고, 이것이 probability보다 큰지 비교
        # 즉, 만약 probability가 0.5면 즉, 0.5보다 크면 원본신호 그대로 return 즉,0.5 확률로 shift할건지 결정.
        if torch.rand(1) > self._p:
            return signal

        # shift하기로 됬으면 진행
        # -m과 m사이의 random 값을 생성하여 이동시킬 샘플 수를 정의. 그런다음 torch.roll을 통해 signal을 이동 시키고 신호를 return시킴.
        # roll()의 세번째 인자 0은 신호를 어떤 차원에서 이동시킬 지 정의 / 여기서는 0번 차원 즉 첫번쨰 차원을 이동시킴.
        # 신호는 dataloader에서 보면 1개의 index로 가져와서 1차원이기 때문에 왼쪽이나 오른쪽으로 이동하게됨.
        return torch.roll(signal, torch.randint(-self._m, self._m, (1, )).item(), 0)


class RandomNoise:

    # 이거도 마찬가지로 max_amplitude와 probability를 매개변수로 받음.
    # probability는 위에와 동일하고, max_amplitude또한 위의 max_num_samples와 비슷하게 얼마나 noise를 줄건지 noise의 최대 진폭을 정의함.
    def __init__(self, max_amplitude, probability) -> None:
        self._a = max_amplitude
        self._p = probability

    # p에 따라 확률적으로 Random noise진행.
    def __call__(self, signal):
        if torch.rand(1) > self._p:
            return signal

        # noise를 주기로 했을 때 return
        # torch.rand(len(signal))을 통해 signal의 길이만큼의 random 값(0~1)을 생성 하여 -0.5 한 값에 2를 곱하고 _a를 곱합.
        # 이 값이 원본 signal값과 더해져서 noise가 추가됨.
        return signal + (torch.rand(len(signal)) - 0.5) * 2. * self._a
