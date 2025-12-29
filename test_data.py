import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from speechbrain.lobes.features import Fbank

# 1. 设置路径 (指向你刚才解压的位置)
data_path = r"D:\Datasets\LibriSpeech\dev-clean"

# 2. 自动找一个音频文件测试
audio_files = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".flac"):
            audio_files.append(os.path.join(root, file))

if not audio_files:
    print(f"错误：在 {data_path} 中没找到 .flac 文件，请检查路径！")
else:
    test_wav = audio_files[0]
    print(f"读取到音频文件: {test_wav}")

    # 3. 加载音频
    signal, fs = torchaudio.load(test_wav)

    # 4. 使用 SpeechBrain 提取梅尔频谱特征 (Fbank)
    fbank_maker = Fbank(n_mels=80)
    features = fbank_maker(signal)

    # 5. 绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制波形图
    plt.subplot(2, 1, 1)
    plt.plot(signal.t().numpy())
    plt.title("Waveform (Original)")
    plt.grid(True)

    # 绘制梅尔频谱图
    plt.subplot(2, 1, 2)
    plt.imshow(features.detach().cpu().squeeze().T, origin='lower', aspect='auto')
    plt.title("Mel Spectrogram (AI Features)")
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    print("generating......")
    plt.show()