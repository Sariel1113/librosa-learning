import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载音频 (使用 librosa 自带的示例语音)
y, sr = librosa.load(librosa.ex('libri1'))

# 2. 生成梅尔语谱图 (n_mels=128)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, win_length=2048)
S_db = librosa.power_to_db(S, ref=np.max)

# ---------------------------------------------------------
# 3. 模拟补偿逻辑 (模拟 NAL 逻辑)
# ---------------------------------------------------------
# 我们先克隆一份语谱图，用于做补偿处理.原始特征（Raw Feature）通常是神圣不可侵犯的，所有修改必须在副本上进行，这是一种良好的编程习惯。
S_db_compensated = S_db.copy()

# 获取梅尔频带对应的中心频率 (Hz)
mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)

# 练习逻辑：假设我们要给 2000Hz 以上的信号补偿 20dB
# 找到频率大于 2000Hz 的那些行的索引
high_freq_mask = mel_freqs > 2000

# 将这些行对应的数值增加 20dB
S_db_compensated[high_freq_mask, :] += 20

# ---------------------------------------------------------
# 4. 可视化对比
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))

# 子图 1: 原始语谱图
plt.subplot(2, 1, 1)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Original Mel Spectrogram (Before NAL Compensation)')

# 子图 2: 补偿后的语谱图
plt.subplot(2, 1, 2)
librosa.display.specshow(S_db_compensated, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('NAL Compensated Mel Spectrogram (+20dB above 2000Hz)')

plt.tight_layout()
plt.show()

print(f"实验完成！补偿了 {np.sum(high_freq_mask)} 个频带。")