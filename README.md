# ReSpeaker V3 + EfficientAT 실시간 추론

ReSpeaker Array V3 USB 마이크의 채널 0 오디오를 16 kHz로 실시간 수집하고, 2초 단위로 EfficientAT `mn10_as` 모델을 실행해 10개 커스텀 클래스로 집계합니다.

마이크 입력은 16 kHz로 받지만, EfficientAT AudioSet pretrained 모델의 공식 전처리 조건에 맞추기 위해 모델 입력 직전에 32 kHz로 리샘플링합니다. `mn10_as`는 10초 입력 기준으로 학습되어서, 실시간 2초 청크는 모델 입력 직전에 10초 길이로 zero padding됩니다.

## 설치

```bash
bash install.sh
```

`install.sh`는 다음 작업을 수행합니다.

- `git clone https://github.com/fschmid56/EfficientAT.git`
- BLE GATT 실행에 필요한 `python3-dbus`, `python3-gi`, `gir1.2-glib-2.0`, `bluez` 설치
- `pip install torch torchaudio sounddevice "numpy<2"`
- EfficientAT 모델 import에 필요한 `torchvision` 설치

첫 실행 시 `mn10_as` pretrained weight가 EfficientAT GitHub Release에서 자동 다운로드됩니다.

### Jetson 가상환경 설치

Jetson에서 시스템 Python이 아니라 프로젝트 가상환경 안에 설치하려면 먼저 가상환경을 활성화한 뒤 설치 스크립트를 실행하세요.

```bash
cd ~/jetson_ef

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip
bash install.sh
bash install_eval.sh
```

가상환경이 제대로 활성화됐는지 확인하려면:

```bash
which python
which pip
```

출력이 프로젝트의 `.venv/bin/python`, `.venv/bin/pip`를 가리키면 됩니다.

BLE 버전(`realtime_inference_ble.py`)은 Ubuntu/Jetson의 `python3-dbus`, `python3-gi` 시스템 패키지를 사용합니다. 이미 `python3 -m venv .venv`로 가상환경을 만들었다면 다음 값으로 바꾼 뒤 다시 실행하세요.

```bash
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg
```

Jetson에서는 `torch`, `torchaudio`, `torchvision`을 일반 `pip install`로 설치하면 JetPack/CUDA 버전과 맞지 않을 수 있습니다. 이 경우 NVIDIA가 제공하는 JetPack 버전별 PyTorch wheel을 먼저 설치한 뒤, 나머지 패키지만 설치하세요.

```bash
pip install sounddevice "numpy<2" pandas scikit-learn seaborn matplotlib librosa tqdm
```

## 실행

```bash
python realtime_inference.py
```

기본값은 2초 청크마다 항상 추론 결과를 출력합니다. `--min-db 30`은 `-30 dBFS`보다 작은 입력을 빨간색 `status=낮음(소리작음 ...)`으로 표시하는 기준입니다.

각 청크는 모델 입력 전에 expander 방식으로 왜곡/강조합니다. 작은 진폭의 노이즈는 `-18 dB` 줄이고, 큰 진폭의 메인 소리는 `+8 dB` 키워서 노이즈와 이벤트의 차이를 벌립니다.

출력 점수는 EfficientAT의 527개 AudioSet logit에 sigmoid를 적용한 값입니다. 커스텀 클래스가 여러 AudioSet 라벨을 묶는 경우, 해당 라벨들의 sigmoid 점수 중 가장 큰 값을 표시합니다. 이 값들은 multi-label confidence라서 전체 합이 100%가 되지 않습니다.

기본 `--min-score 0.05`가 적용되어 최고 커스텀 점수가 5% 미만이면 빨간색 `status=낮음(점수낮음 ...)`으로 표시합니다. 입력 레벨과 점수가 모두 기준을 넘으면 초록색 `status=감지`로 표시됩니다.

```bash
python realtime_inference.py --min-db 30 --min-score 0.05 --enhance-threshold-db 35 --noise-reduction-db 18 --main-gain-db 8
```

매 2초마다 다음 형식으로 출력됩니다.

```text
[HH:MM:SS] 예측: construction (72.3%) | status=감지 | level=-22.4 dBFS | enhanced=-15.1 dBFS | quiet_gain=0.13x loud_gain=2.51x | 전체: construction=72.3%, gunshot=3.1%, ...
```

디바이스명이 자동 탐색되지 않으면 입력 디바이스 목록을 확인한 뒤 index를 직접 지정할 수 있습니다.

```bash
python realtime_inference.py --list-devices
python realtime_inference.py --device-index 3
```

ReSpeaker 4 Mic Array 6채널 펌웨어는 보통 `ch0=처리된 오디오`, `ch1-4=raw mic`, `ch5=playback`입니다. `ch0`이 클리핑되거나 너무 크게 나오면 raw mic 채널을 지정해서 실행하세요.

```bash
python realtime_inference.py --device-index 15 --channel-index 1
```

## ReSpeaker V3 연결 확인

```bash
python -m sounddevice
```

출력 목록에서 `ReSpeaker`, `Seeed`, `Array V3`와 유사한 이름의 입력 디바이스가 보이고, 입력 채널 수가 6개인지 확인하세요.

## 문제 해결

- 디바이스를 못 찾을 때: `python -m sounddevice` 또는 `python realtime_inference.py --list-devices`로 실제 이름과 index를 확인한 뒤 `--device-index`를 사용하세요.
- CUDA가 없을 때: 스크립트가 자동으로 CPU를 사용합니다. CPU에서는 추론이 느릴 수 있지만 별도 설정은 필요 없습니다.
- EfficientAT 저장소가 없을 때: `bash install.sh`를 먼저 실행하거나 `--efficientat-dir`로 clone된 경로를 지정하세요.
- 모델 로딩 중 `torchvision` 오류가 날 때: `pip install torchvision`을 실행하세요.
- `No module named 'dbus'`가 뜰 때: `sudo apt-get install -y python3-dbus python3-gi gir1.2-glib-2.0 bluez`를 실행하고, 가상환경을 쓴다면 `.venv/pyvenv.cfg`의 `include-system-site-packages`를 `true`로 바꾸세요.
- Jetson에서 PyTorch 설치가 실패할 때: JetPack/CUDA 버전에 맞는 NVIDIA 제공 PyTorch, torchaudio wheel을 먼저 설치한 뒤 `pip install sounddevice "numpy<2" torchvision`을 실행하세요.
- `Numpy is not available` 또는 `compiled using NumPy 1.x cannot be run in NumPy 2.x`가 뜰 때: 현재 PyTorch/확장 모듈이 NumPy 2.x와 맞지 않는 상태입니다. 가상환경에서 `python -m pip install --force-reinstall "numpy<2"`를 실행한 뒤 다시 시작하세요.

## ESC-50 평가

ESC-50에서 10개 커스텀 클래스에 매핑 가능한 샘플만 사용해 EfficientAT `mn10_as` 정확도를 평가할 수 있습니다.

```bash
git clone https://github.com/karolpiczak/ESC-50.git
bash install_eval.sh
python evaluate_esc50.py --esc50_dir ./ESC-50
```

옵션:

```bash
python evaluate_esc50.py --esc50_dir ./ESC-50 --model mn10_as --device auto
```

결과는 실행마다 `results/ver1`, `results/ver2`, ... 형태의 새 폴더에 저장됩니다.

저장 파일:

- `results/verN/confusion_matrix.png`
- `results/verN/eval_results.csv`
- `results/verN/classification_report.txt`
