import numpy as np
import librosa
import sounddevice as sd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# 1. 加载音频
y, sr = librosa.load(librosa.ex('choice'))

# 2. STFT 变换
D = librosa.stft(y, n_fft=2048)
# --- 修正点在此 ---
magnitude, phase = librosa.magphase(D)
stft_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

# 3. 听力学增益 (NAL-R)
my_hl = {250: 10, 500: 20, 1000: 40, 2000: 60, 4000: 70, 8000: 85}
my_gain = {f: (loss * 0.6 if f >= 2000 else loss * 0.3) for f, loss in my_hl.items()}

# 4. 插值并转换
freq_keys = np.array(list(my_gain.keys()))
gain_vals = np.array(list(my_gain.values()))
# 4. 优化后的增益处理
f_interp = interp1d(freq_keys, gain_vals, kind='linear', fill_value="extrapolate")
gain_db = f_interp(stft_freqs)

# A. 限制增益：不要让它飞到天上去（Capping）
gain_db = np.clip(gain_db, a_min=None, a_max=30)

# B. 曲线平滑：消除金属感伪影（Smoothing）
gain_db = savgol_filter(gain_db, window_length=151, polyorder=2)

# C. 线性转换
gain_linear = 10**(gain_db / 20)

# 5. 应用增益并用 iSTFT 还原
# 我们保留了 phase，所以声音不会“变调”
magnitude_aided = magnitude * gain_linear[:, np.newaxis]
D_aided = magnitude_aided * phase  # 幅度乘回原始相位
y_aided = librosa.istft(D_aided)

# 6. 归一化 (防爆音)
if np.max(np.abs(y_aided)) > 0:
    y_aided = y_aided / np.max(np.abs(y_aided))

# 7. 实时播放对比
print("🔊 原始声音...")
sd.play(y, sr)
sd.wait()

print("🔊 助听器补偿后的声音 (高保真相位保留)...")
sd.play(y_aided, sr)
sd.wait()