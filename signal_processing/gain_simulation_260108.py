import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sounddevice as sd

# --- 1. 环境与参数设置 ---
N_MELS = 128
F_MAX = 8000
SR = 22050  # 标准采样率
# 加载示例音频（librosa 自带的录音）
y, sr = librosa.load(librosa.ex('choice'), duration=15)

# --- 2. 听力学逻辑：计算增益 ---
# 建立你的听力损失字典 (HL: Hearing Loss)
my_hl = {250: 10, 500: 20, 1000: 40, 2000: 60, 4000: 70, 8000: 85}
# 计算 NAL-R 风格增益
my_gain = {}
for freq, loss in my_hl.items():
    if freq >= 2000:
        my_gain[freq] = loss * 0.6
    else:
        my_gain[freq] = loss * 0.3

# --- 3. 数学映射：插值对齐 ---
freq_keys = np.array(list(my_gain.keys()))
gain_values = np.array(list(my_gain.values()))
# 创建插值函数
f_interp = interp1d(freq_keys, gain_values, kind='linear', fill_value="extrapolate")
# 生成 Mel 频率坐标轴
mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=F_MAX)
# 将 6 个点的增益映射到 128 个 Mel 频带上
full_gain_curve = f_interp(mel_freqs)

# --- 4. DSP 处理：应用增益 ---
# 生成原始 Mel 语谱图 (并转为 dB)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=F_MAX)
S_db = librosa.power_to_db(S, ref=np.max)
# 应用增益 (矩阵广播机制：每一帧都加上这条增益曲线)
# S_db 是 (128, T), full_gain_curve 是 (128,)
S_db_aided = S_db + full_gain_curve[:, np.newaxis]

# --- 5. 结果可视化 ---
plt.figure(figsize=(12, 10))
# 图 1：原始语谱图
plt.subplot(3, 1, 1)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', fmax=F_MAX, cmap='magma')
plt.title('Original Spectrogram (Before Hearing Aid)')
plt.colorbar(format='%+2.0f dB')
# 图 2：计算出的补偿增益曲线
plt.subplot(3, 1, 2)
plt.plot(mel_freqs, full_gain_curve, color='cyan', linewidth=2, label='Interpolated Gain Curve')
plt.scatter(freq_keys, gain_values, color='red', zorder=5, label='Original Prescription Points')
plt.fill_between(mel_freqs, 0, full_gain_curve, color='cyan', alpha=0.2)
plt.title('Target Gain Curve (NAL-R Style)')
plt.ylabel('Gain (dB)')
plt.xlabel('Frequency (Hz)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
# 图 3：补偿后的语谱图
plt.subplot(3, 1, 3)
librosa.display.specshow(S_db_aided, sr=sr, x_axis='time', y_axis='mel', fmax=F_MAX, cmap='magma')
plt.title('Aided Spectrogram (After NAL-R Compensation)')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

print("✅ 实验完成！请查看 PyCharm 弹出的对比图。")
print(f"提示：4000Hz 处的增益已应用：{f_interp(4000):.1f} dB")

# --- 6. 还原声音 (Vocoder 环节) ---
# 1. 把 dB 转回 能量 (Power)
S_aided_power = librosa.db_to_power(S_db_aided)
# 2. 从 Mel 语谱图还原为普通的线性频谱 (Invert Mel)
S_aided_stft = librosa.feature.inverse.mel_to_stft(S_aided_power, sr=sr, n_fft=2048)
# 3. 使用 Griffin-Lim 算法重建相位并生成音频
print("正在合成声音，请稍候...")
y_aided = librosa.griffinlim(S_aided_stft)

def play(data, title):
    print(f"正在播放: {title}...")
    sd.play(data, sr)
    sd.wait() # 等待当前音频播完

# 6. 播放对比
print("播放原始声音...")
sd.play(y, sr)
sd.wait()
print("播放高频增强后的声音（助听模式）...")
sd.play(y_aided, sr)
sd.wait()

print("播放结束")
