import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载 MP3
music = r'D:\1a-ysh-947444664\本科阶段\课题相关\2024大创\听觉对比度阈值测试的首发应用研究\创新创业大赛\展示\STM.mp3'
y, sr = librosa.load(music, sr=None, offset=4.9, duration=1)

# 2. 【核心优化】减小窗口以应对扫频信号，提高时间分辨率，防止能量散开产生蓝点
# n_fft 设为 1024 或 2048，对于 2.5kHz 绰绰有余
n_fft = 8096
hop_length = 128 # 保持步长小，让斜线平滑
win_length = 1024

D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hamming')
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# 3. 仿照老师的范围：-5 到 -45
S_db_pure = np.clip(S_db, -45, -5)

plt.figure(figsize=(10, 7))

# 4. 绘图：去掉 shading='gouraud' 可能会让斜线更清晰（不那么糊）
img = librosa.display.specshow(S_db_pure,
                               sr=sr,
                               hop_length=hop_length,
                               x_axis='time',
                               y_axis='linear', # 先用 linear
                               cmap='jet',
                               vmin=-45,
                               vmax=-5)

# 5. 【纵轴优化】模仿老师的 kHz 显示
plt.ylim(0, 2500)
plt.yticks([0, 500, 1000, 1500, 2000, 2500], ['0', '0.5', '1', '1.5', '2', '2.5'])
plt.ylabel('频率 (kHz)', fontproperties="SimSun") # 如果环境支持中文
plt.xlabel('时间 (秒)', fontproperties="SimSun")

# 6. 颜色条设置
cbar = plt.colorbar(img)
cbar.set_ticks([-45, -40, -35, -30, -25, -20, -15, -10, -5])

plt.tight_layout()
plt.show()