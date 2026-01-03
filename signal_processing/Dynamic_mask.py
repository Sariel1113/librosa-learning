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

# 3. 创建“智能掩蔽矩阵” (Mask)
# np.where(条件, 符合条件的值, 不符合条件的值)
# 如果能量小于中位数，我们认为它是“弱细节”，放大 8 倍
# 如果能量大于中位数，我们认为它是“强信号”，保持 1 倍
mask = np.where(S < threshold, 8.0, 1.0)

# 4. 应用掩蔽并还原
# 这就是 AI 语音增强最核心的公式：D_modified = D_original * Mask
D_smart = D * mask
y_smart = librosa.istft(D_smart)

# 5. 播放对比
print(f"当前自动阈值: {threshold:.4f}")
print("播放原始声音...")
sd.play(y, sr)
sd.wait()

print("播放智能动态增强后的声音...")
sd.play(y_smart, sr)
sd.wait()