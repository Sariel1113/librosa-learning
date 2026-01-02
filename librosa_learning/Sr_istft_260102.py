import librosa
import numpy as np
import sounddevice as sd

# 加载音频
y, sr = librosa.load(librosa.ex('trumpet'))
D = librosa.stft(y)

# 1. 实验 A：原样转回（带相位，完美的）
y_perfect = librosa.istft(D)

# 2. 实验 B：只保留幅度，扔掉相位
D_abs = np.abs(D)
y_robotic = librosa.istft(D_abs)

# --- 播放逻辑 ---
print("正在播放：原始还原（完美无损）...")
sd.play(y_perfect, sr)
sd.wait()  # <--- 关键！这行会让程序停住，直到音频播放完毕

print("正在播放：无相位版本（机器人感/电音感）...")
sd.play(y_robotic, sr)
sd.wait()  # <--- 关键！

print("播放结束")