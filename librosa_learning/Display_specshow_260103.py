import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

y, sr = librosa.load(librosa.example('choice'), duration=15)
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
D = librosa.stft(y)
D_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# 3. 确定我们要增强的频率范围
# 假设我们要增强 3000Hz 以上的声音
# librosa 有个工具可以把 Hz 转成 D 矩阵的行索引 (Index)
freq_threshold = 3000
# 1. 算出每一个 FFT bin 代表多少 Hz (频率分辨率)
# 公式：每个格子的宽度 = 采样率 / n_fft
bin_resolution = sr / 2048

# 2. 目标频率 / 每个格子的宽度 = 对应的行索引
# 使用 int() 取整，因为矩阵索引必须是整数
idx_threshold = int(freq_threshold / bin_resolution)

print(f"频率 {freq_threshold}Hz 对应的矩阵行索引是: {idx_threshold}")
# 4. 创建一个 D 的副本进行修改
D_enhanced = D.copy()
# 【核心修改步骤】
# 将 idx_threshold 行以后的所有数据（高频部分）乘以 5 倍
D_enhanced[idx_threshold:, :] = D_enhanced[idx_threshold:, :] * 0.05
# 5. 还原回声音
# 因为我们是在原始复数矩阵 D 上修改的，保留了原配相位，所以直接用 istft
y_enhanced = librosa.istft(D_enhanced)


# 绘制第一张图
img = librosa.display.specshow(D_dB, y_axis='linear', x_axis='time', sr=sr, ax=ax[0])
ax[0].set(title='Linear-frequency power spectrogram')
ax[0].label_outer()

# 绘制第二张图
hop_length = 1024
D_dB = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D_dB, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[1])
ax[1].set(title='Log-frequency power spectogram')
ax[1].label_outer()

# 绘制第三张图
hop_length = 1024
D_enhanced_dB = librosa.amplitude_to_db(np.abs(librosa.stft(y_enhanced, hop_length=hop_length)), ref=np.max)
librosa.display.specshow(D_enhanced_dB, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[2])
ax[2].set(title='Enhance-Log-frequency power spectogram')
ax[2].label_outer()

fig.colorbar(img, ax=ax, format="%+2.f dB")


plt.show()
def play(data, title):
    print(f"正在播放: {title}...")
    sd.play(data, sr)
    sd.wait() # 等待当前音频播完

# 6. 播放对比
print("播放原始声音...")
sd.play(y, sr)
sd.wait()
print("播放高频增强后的声音（助听模式）...")
sd.play(y_enhanced, sr)
sd.wait()
