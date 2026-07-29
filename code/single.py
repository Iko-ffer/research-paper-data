import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.decomposition import FastICA
from collections import Counter
import os
import warnings

warnings.filterwarnings('ignore')


class FetalSignalProcessorV3:
    def __init__(self, fs: int = 500, hr_window_sec: float = 1.0, hr_step_sec: float = 0.5):
        self.fs = fs
        self.hr_win_samples = int(hr_window_sec * fs)  # 1s窗口 = 500点
        self.hr_step_samples = int(hr_step_sec * fs)  # 0.5s步长 = 250点
        # 通道识别投票参数 (沿用 8s/4s 逻辑确保稳定性)
        self.vote_win = 8 * fs
        self.vote_step = 4 * fs

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        nyq = self.fs / 2
        b_band, a_band = butter(4, [5 / nyq, 45 / nyq], btype='band')
        b_notch, a_notch = butter(4, [49 / nyq, 51 / nyq], btype='bandstop')
        processed = np.zeros_like(data)
        for ch in range(data.shape[0]):
            sig = filtfilt(b_notch, a_notch, data[ch])
            processed[ch] = filtfilt(b_band, a_band, sig)
        return processed

    def get_window_avg_hr(self, signal_win, is_fetal=False):
        """计算1s窗口内所有Peak间距的平均瞬时值"""
        if np.std(signal_win) < 1e-6: return np.nan
        sig_norm = (signal_win - np.mean(signal_win)) / (np.std(signal_win) + 1e-8)

        # 动态阈值：使用当前窗口峰值的60%作为显著度，精准避开T波
        dyn_prom = np.max(np.abs(sig_norm)) * 0.6
        # 物理不应期约束
        dist = int(self.fs * 0.28) if is_fetal else int(self.fs * 0.45)

        peaks, _ = find_peaks(np.abs(sig_norm), distance=dist, prominence=dyn_prom)

        if len(peaks) < 2: return np.nan

        # 计算该窗口内所有跳动的平均HR
        rr_intervals = np.diff(peaks) / self.fs
        return np.mean(60.0 / rr_intervals)

    def process_file(self, input_file, output_root):
        # 1. 建立独立存储结构
        ecg_path = os.path.join(output_root, "separated_ecg")
        hr_path = os.path.join(output_root, "calculated_hr")
        os.makedirs(ecg_path, exist_ok=True)
        os.makedirs(hr_path, exist_ok=True)

        # 2. 读取与ICA分离
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        df.columns = ["".join(filter(str.isprintable, c)).strip() for c in df.columns]
        time_col = df.columns[0]
        data_cols = [c for c in df.columns if '通道' in c]

        raw_data = df[data_cols].values.T
        clean_data = self.preprocess(raw_data)

        ica = FastICA(n_components=len(data_cols), random_state=42)
        sources = ica.fit_transform(clean_data.T).T
        total_len = sources.shape[1]

        # 3. 通道识别 (基于8s滑动窗口投票)
        m_votes, f_votes = [], []
        for start in range(0, total_len - self.vote_win, self.vote_step):
            end = start + self.vote_win
            for i in range(sources.shape[0]):
                sig = sources[i, start:end]
                sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                p, _ = find_peaks(np.abs(sig_norm), distance=int(self.fs * 0.25), prominence=1.1)
                if len(p) >= 4:
                    bpm = (60 * self.fs) / np.mean(np.diff(p))
                    if 55 <= bpm <= 110:
                        m_votes.append(i)
                    elif 120 <= bpm <= 185:
                        f_votes.append(i)

        m_idx = Counter(m_votes).most_common(1)[0][0] if m_votes else 0
        f_idx = Counter(f_votes).most_common(1)[0][0] if f_votes else 1

        # 4. 【保存ECG】 500Hz 全采样
        ecg_df = pd.DataFrame({
            'RelTime_ms': df[time_col].values,
            'Maternal_ECG': sources[m_idx],
            'Fetal_ECG': sources[f_idx]
        })
        ecg_df.to_csv(os.path.join(ecg_path, os.path.basename(input_file).replace('.csv', '_ecg.csv')), index=False)

        # 5. 【保存HR】 0.5s步长计算 (计算前1s的均值)
        hr_list = []
        for i in range(0, total_len - self.hr_win_samples, self.hr_step_samples):
            # 视窗范围 [i : i+500]
            end = i + self.hr_win_samples
            m_hr = self.get_window_avg_hr(sources[m_idx, i:end], is_fetal=False)
            f_hr = self.get_window_avg_hr(sources[f_idx, i:end], is_fetal=True)

            hr_list.append({
                'RelTime_ms': df[time_col].iloc[end - 1],  # 时间戳定在窗口末端
                'Maternal_HR': m_hr,
                'Fetal_HR': f_hr
            })

        hr_df = pd.DataFrame(hr_list).interpolate().fillna(method='bfill')
        hr_df.to_csv(os.path.join(hr_path, os.path.basename(input_file).replace('.csv', '_hr.csv')), index=False)

        print(f"处理完成: {input_file} | 母体通道: {m_idx}, 胎儿通道: {f_idx}")
        self._plot_verification(sources, m_idx, f_idx, df[time_col].values[:2500])

    def _plot_verification(self, sources, m_idx, f_idx, t_axis):
        """可视化前5秒的Peak定位情况"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, idx, title, is_f in zip([ax1, ax2], [m_idx, f_idx], ["Maternal", "Fetal"], [False, True]):
            sig = sources[idx, :len(t_axis)]
            sig_n = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
            prom = np.max(np.abs(sig_n)) * 0.6
            dist = int(self.fs * (0.45 if not is_f else 0.28))
            p, _ = find_peaks(np.abs(sig_n), distance=dist, prominence=prom)
            ax.plot(t_axis, sig_n, color='black', alpha=0.7, lw=0.8)
            ax.scatter(t_axis[p], sig_n[p], color='red', s=35, label='Detected Peaks')
            ax.set_title(f"{title} (Source {idx}) - 1.0s Window Analysis")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    processor = FetalSignalProcessorV3()
    processor.process_file("1_slm.csv", "final_output")