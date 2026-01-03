import librosa
import numpy as np
import matplotlib.pyplot as plt
import time
import sounddevice as sd

y, sr = librosa.load(librosa.example('trumpet'), duration=3.5)
D = librosa.stft(y)
S = np.abs(librosa.stft(y))
y_inv = librosa.griffinlim(S, n_iter=32)      # Griffin-Lim 恢复
y_istft = librosa.istft(S)           # 错误示范：直接用幅度谱（相位全0）

fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
librosa.display.waveshow(y, sr=sr, color='b', ax=ax[0])
ax[0].set(title='Original', xlabel=None)
ax[0].label_outer()
librosa.display.waveshow(y_inv, sr=sr, color='g', ax=ax[1])
ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
ax[1].label_outer()
librosa.display.waveshow(y_istft, sr=sr, color='r', ax=ax[2])
ax[2].set_title('Magnitude-only istft reconstruction')

plt.show()
# 播放函数
def play(data, title):
    print(f"正在播放: {title}...")
    sd.play(data, sr)
    sd.wait() # 等待当前音频播完
    time.sleep(0.5) # 停顿一下

# 3. 开始对比播放
play(y, "原始音频 (Original)")
play(y_inv, "Griffin-Lim 还原 (32次迭代)")
play(y_istft, "无相位直接还原 (ISTFT with zero phase)")

print("播放完毕！")
