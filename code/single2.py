import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt, find_peaks
from scipy.interpolate import interp1d
import os


class FHRDataArchiver:
    def __init__(self, fs=500, gold_fs=5):
        self.fs = fs
        self.gold_fs = gold_fs

    def adaptive_separation(self, v1, v6):
        """自适应对冲分离逻辑"""
        # 基线漂移去除
        bl_win = int(self.fs * 0.8)
        if bl_win % 2 == 0: bl_win += 1
        v1_c = v1 - medfilt(v1, bl_win)
        v6_c = v6 - medfilt(v6, bl_win)

        # 寻找母体 R 波计算系数 K
        m_peaks, _ = find_peaks(v1_c, distance=int(self.fs * 0.5), prominence=np.std(v1_c) * 1.2)
        v1_amps, v6_amps = [], []
        win = int(self.fs * 0.02)
        for p in m_peaks:
            start, end = max(0, p - win), min(len(v1_c), p + win)
            v1_amps.append(np.max(v1_c[start:end]))
            v6_amps.append(np.abs(np.min(v6_c[start:end])))

        K = np.median(v1_amps) / (np.median(v6_amps) + 1e-9) if v1_amps else 1.0
        f_raw = v1_c + (v6_c * K)

        # 5-40Hz 带通，保留 R 波特征
        nyq = 0.5 * self.fs
        b, a = butter(2, [5 / nyq, 40 / nyq], btype='band')
        return filtfilt(b, a, f_raw)

    def calculate_fhr(self, f_sig):
        """提取瞬时心率"""
        peaks, _ = find_peaks(f_sig, distance=int(self.fs * 0.3), prominence=np.std(f_sig) * 0.8)
        if len(peaks) < 5: return None

        intervals = np.diff(peaks) / self.fs
        raw_hr = 60.0 / intervals
        hr_times = (peaks[:-1] + peaks[1:]) / (2 * self.fs)  # 时间戳中心化

        # 生理约束平滑
        valid = (raw_hr >= 100) & (raw_hr <= 170)
        if not np.any(valid): return None

        f_hr = raw_hr.copy()
        f_hr[~valid] = np.nan
        ok = ~np.isnan(f_hr)
        f_hr[~ok] = np.interp(hr_times[~ok], hr_times[ok], f_hr[ok])
        return hr_times, medfilt(f_hr, 9)

    def find_best_sync_offset(self, my_hr_1hz, gold_hr_1hz, max_scan_sec=120):
        """
        自动对齐核心：
        通过滑动金标准序列，寻找与算法结果 MAE 最小的偏移位置
        """
        best_mae = float('inf')
        best_off = 0

        # 确保有足够的重叠区域进行比对
        search_range = range(-max_scan_sec, max_scan_sec + 1)

        for off in search_range:
            if off >= 0:
                # 金标准领先（需要截掉金标准开头）
                g_seg = gold_hr_1hz[off:]
                m_seg = my_hr_1hz[:len(g_seg)]
            else:
                # 算法领先（需要截掉算法开头）
                m_seg = my_hr_1hz[-off:]
                g_seg = gold_hr_1hz[:len(m_seg)]

            length = min(len(m_seg), len(g_seg))
            if length < 30: continue  # 至少比对 30 秒数据

            mae = np.nanmean(np.abs(m_seg[:length] - g_seg[:length]))
            if mae < best_mae:
                best_mae = mae
                best_off = off
        return best_off

    def run(self, root_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        subjects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for sub in subjects:
            sub_path = os.path.join(root_dir, sub)
            for f in [f for f in os.listdir(sub_path) if f.endswith('_slm.csv')]:
                tag = f"{sub}_{f.replace('_slm.csv', '')}"
                gold_f = os.path.join(sub_path, f.replace('_slm.csv', '_g.csv'))
                if not os.path.exists(gold_f): continue

                print(f"--- 正在处理: {tag} ---")
                raw_df = pd.read_csv(os.path.join(sub_path, f))
                v1, v6 = raw_df.iloc[:, 1].values, raw_df.iloc[:, 6].values

                # 1. 物理分离并保存 fECG 波形
                f_ecg = self.adaptive_separation(v1, v6)
                pd.DataFrame({
                    'time_s': np.arange(len(f_ecg)) / self.fs,
                    'fetal_ecg_uv': f_ecg
                }).to_csv(os.path.join(output_dir, f"{tag}_fECG_waveform.csv"), index=False)

                # 2. 计算心率趋势
                hr_data = self.calculate_fhr(f_ecg)
                if hr_data is None: continue
                my_t, my_hr = hr_data

                # 3. 准备对齐：统一到 1Hz
                gold_df = pd.read_csv(gold_f)
                g_hr_raw = pd.to_numeric(gold_df.iloc[:, 1], errors='coerce').ffill().values
                g_t_raw = np.arange(len(g_hr_raw)) / self.gold_fs

                t_common = np.arange(0, int(min(my_t[-1], g_t_raw[-1])))
                f_m = interp1d(my_t, my_hr, bounds_error=False, fill_value=np.nanmean(my_hr))
                f_g = interp1d(g_t_raw, g_hr_raw, bounds_error=False, fill_value=np.nanmean(g_hr_raw))
                m_1hz, g_1hz = f_m(t_common), f_g(t_common)

                # 4. 执行自动同步对齐
                offset = self.find_best_sync_offset(m_1hz, g_1hz)
                print(f"    检测到物理时间偏置: {offset} 秒")

                if offset >= 0:
                    final_g, final_m = g_1hz[offset:], m_1hz[:len(g_1hz) - offset]
                else:
                    final_m, final_g = m_1hz[-offset:], g_1hz[:len(m_1hz) + offset]

                # 5. 保存对齐后的心率结果表
                min_len = min(len(final_m), len(final_g))
                pd.DataFrame({
                    'time_aligned_s': np.arange(min_len),
                    'my_fhr_bpm': final_m[:min_len],
                    'gold_fhr_bpm': final_g[:min_len],
                    'error_bpm': final_m[:min_len] - final_g[:min_len]
                }).to_csv(os.path.join(output_dir, f"{tag}_fHR_comparison.csv"), index=False)

                # 6. 生成报告图
                self._quick_plot(f_ecg, final_m[:min_len], final_g[:min_len], offset, tag, output_dir)

    def _quick_plot(self, ecg, m, g, off, tag, path):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(5000) / self.fs, ecg[:5000], lw=0.7, color='teal')
        plt.title(f"Separated Waveform (10s) | {tag}")
        plt.subplot(2, 1, 2)
        plt.plot(g, 'k--', label='Gold Standard', alpha=0.5)
        plt.plot(m, 'r-', label='Algorithm', lw=1.2)
        plt.title(f"Aligned Heart Rate | Detected Offset: {off}s | MAE: {np.mean(np.abs(m - g)):.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{tag}_summary.png"))
        plt.close()


if __name__ == "__main__":
    FHRDataArchiver().run(".", "Fetal_Output_Data")