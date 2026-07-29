import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def ultimate_smooth_ecg(file_path, fs=500):
    # 1. 加载数据
    try:
        df = pd.read_csv(file_path)
        # 第一列时间，第二列原始信号
        sig = df.iloc[:, 1].values.astype(float)
    except Exception as e:
        print(f"读取失败: {e}")
        return None


    # 3. 【核心优化一】强力陷波与高频抑制
    # 彻底消除 50Hz, 100Hz, 150Hz 的工频残留锯齿
    for f in [50, 100, 150]:
        b, a = signal.iirnotch(f / (0.5 * fs), Q=30)
        sig = signal.filtfilt(b, a, sig)

    # 4. 【核心优化二】非线性中值剥离基线 (1.2s 窗口)
    # 这种方法比高通滤波更温和，能让 ST 段非常平整
    base = signal.medfilt(sig, kernel_size=int(1.2 * fs) | 1)
    sig_nobase = sig - base

    # 5. 【核心优化三】细节重建：大尺度高斯平滑
    # 传统的 SG 滤波保不住锯齿，我们改用高斯卷积，它能产生“丝滑”的视觉效果
    # window 越大越平滑，std 决定了平滑的力度
    window = signal.windows.gaussian(21, std=3.5)
    sig_smooth = signal.convolve(sig_nobase, window / window.sum(), mode='same')

    # 6. 【核心优化四】形态学二次修平
    # 针对你提到的 T 波分裂/切迹，使用一个小的中值窗口强行填平
    sig_final = signal.medfilt(sig_smooth, kernel_size=21)

    # 7. 归一化
    sig_final = (sig_final - np.mean(sig_final)) / np.std(sig_final)

    return sig, sig_final


# ==========================================
# 执行与对比绘图
# ==========================================
file_path = 'Clean_ECG_Signals.csv'
raw, cleaned = ultimate_smooth_ecg(file_path, fs=500)

if cleaned is not None:
    # 保存结果数据，方便你再次确认
    pd.DataFrame({'Original_Inverted': raw * -1, 'Final_Clean': cleaned}).to_excel('Final_Smooth_ECG.xlsx', index=False)

    plt.figure(figsize=(16, 6))

    # 选取一段中间最清晰的 3 秒
    start, end = 2500, 4000
    t = np.arange(1500) / 500

    # 绘制最终平滑后的红线
    plt.plot(t, cleaned[start:end], color='#CC0000', lw=2.5, label='Ultimate Smooth (Paper Grade)')

    # 医学背景网格
    plt.gca().set_facecolor('white')
    plt.grid(which='major', color='#FFCCCC', linestyle='-', linewidth=1.0)
    plt.grid(which='minor', color='#FFE5E5', linestyle='-', linewidth=0.5)
    plt.minorticks_on()

    plt.ylim(-1.5, 4.5)
    plt.xlim(0, 3)
    plt.title("Hardware-Defect Compensated & Smoothed ECG", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")

    plt.tight_layout()
    plt.savefig('Smooth_Result.png', dpi=300)
    plt.show()
    print(">>> 终极平滑数据已导出：Final_Smooth_ECG.xlsx")