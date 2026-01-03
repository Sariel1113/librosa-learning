import librosa
import numpy as np
import sounddevice as sd

# 1. 加载音频
y, sr = librosa.load(librosa.ex('choice'))
D = librosa.stft(y)
S = np.abs(D)  # 提取振幅（能量）

# 2. 自动计算阈值：取所有格点能量的中位数
# 这样无论录音是大声还是小声，阈值都能自动适应
threshold = np.median(S)

# 计算能量的比例（0到1之间）
S_norm = (S - S.min()) / (S.max() - S.min())

# 这种写法会让 mask 包含 1.0 到 8.0 之间所有的数值
# 能量越小，值越接近 8.0；能量越大，值越接近 1.0
mask_smooth = 1.0 + 7.0 * (1.0 - S_norm)
D_mask_smooth = D * mask_smooth
y_smart = librosa.istft(D_mask_smooth)

D_dB_orig = librosa.amplitude_to_db(np.abs(D), ref=np.max)
D_dB_smart = librosa.amplitude_to_db(np.abs(D_mask_smooth), ref=np.max)

# 再次绘图

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

img1 = librosa.display.specshow(D_dB_orig, sr=sr, x_axis='time', y_axis='log', ax=ax[0])
ax[0].set(title='Original Spectrogram (dB)')
fig.colorbar(img1, ax=ax[0], format="%+2.f dB")

img2 = librosa.display.specshow(mask_smooth, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
ax[1].set(title='Smart Mask (Multipliers)')
fig.colorbar(img2, ax=ax[1])

img3 = librosa.display.specshow(D_dB_smart, sr=sr, x_axis='time', y_axis='log', ax=ax[2])
ax[2].set(title='Enhanced Spectrogram (dB)')
fig.colorbar(img3, ax=ax[2], format="%+2.f dB")

plt.show()

# 5. 播放对比
print(f"当前自动阈值: {threshold:.4f}")
print("播放原始声音...")
sd.play(y, sr)
sd.wait()

print("播放智能动态增强后的声音...")
sd.play(y_smart, sr)
sd.wait()