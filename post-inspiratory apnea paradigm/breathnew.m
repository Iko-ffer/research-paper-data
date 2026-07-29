clear; close all; clc;

%% 1. 数据读取与参数配置
filename = 'try.xlsx';  % 请确保多通道数据放在该文件内
if ~exist(filename, 'file')
    error('未找到数据文件：%s', filename);
end

data = xlsread(filename);  
[n_samples, n_channels] = size(data);

% 确保只取前 6 通道
n_channels_actual = min(6, n_channels);
ecg_signals = data(:, 1:n_channels_actual);

fs = 500;  % 采样率 500Hz
t = (0:n_samples-1)' / fs; % 列向量时间轴
fprintf('成功加载数据: %d 样本, %d 通道\n', n_samples, n_channels_actual);

%% 2. 多通道独立预处理与信号质量评估 (SQA)
ecg_clean_all = zeros(n_samples, n_channels_actual);
channel_quality = zeros(1, n_channels_actual);  

% 预剪切滤波器组系数 —— 已修正：'band' 改为 'bandpass'
[b_high, a_high] = butter(4, 0.5/(fs/2), 'high');        % 去基线漂移
[b_band, a_band] = butter(4, [0.5, 45]/(fs/2), 'bandpass');  % 带通滤波
if exist('iirnotch', 'file')
    [b_notch, a_notch] = iirnotch(50/(fs/2), (50/(fs/2))/35); % 50Hz工频陷波
end

fprintf('\n--- 正在进行多通道信号预处理与质量评估 ---\n');
for ch = 1:n_channels_actual
    ecg_raw = ecg_signals(:, ch);
    
    % 级联滤波
    ecg_filtered = filtfilt(b_high, a_high, ecg_raw);
    ecg_filtered = filtfilt(b_band, a_band, ecg_filtered);
    if exist('iirnotch', 'file')
        ecg_filtered = filtfilt(b_notch, a_notch, ecg_filtered);
    end
    
    % 平滑滤波
    ecg_clean_all(:, ch) = sgolayfilt(ecg_filtered, 3, 11);
    
    % 计算信噪比 (SNR) 与质量得分
    signal_power = mean(ecg_clean_all(:, ch).^2);
    noise_power = mean((ecg_raw - ecg_clean_all(:, ch)).^2);
    snr = 10 * log10(signal_power / max(noise_power, 1e-6));
    
    % 归一化质量得分到 [0, 1]
    channel_quality(ch) = min(1, max(0, (snr + 15) / 35)); 
    fprintf('通道 %d -> 信噪比: %.2f dB, 质量得分: %.2f\n', ch, snr, channel_quality(ch));
end

%% 3. 多通道 R 波检测与通道检测得分
channel_r_peaks = cell(1, n_channels_actual);
channel_detection_scores = zeros(1, n_channels_actual);

diff_filter = [1, 2, 0, -2, -1] * (1/8); % 多尺度微分
window_size = round(0.12 * fs);          % 积分窗口
window = ones(1, window_size) / window_size;

for ch = 1:n_channels_actual
    ecg_ch = ecg_clean_all(:, ch);
    
    % 能量积分器
    ecg_diff = conv(ecg_ch, diff_filter, 'same');
    ecg_enhanced = ecg_diff.^2 + 0.5 * abs(ecg_diff);
    ecg_integral = conv(ecg_enhanced, window, 'same');
    
    % 自适应阈值检测
    baseline = median(ecg_integral);
    peak_threshold = max(ecg_integral) * 0.25;
    min_peak_distance = round(0.3 * fs); 
    
    [~, locs] = findpeaks(ecg_integral, 'MinPeakDistance', min_peak_distance, ...
                            'MinPeakHeight', baseline + peak_threshold);
    
    % 局部精确定位 R 波极大值
    r_peaks_ch = zeros(size(locs));
    search_win = round(0.06 * fs);
    for i = 1:length(locs)
        start_idx = max(1, locs(i) - search_win);
        end_idx = min(n_samples, locs(i) + search_win);
        [~, max_idx] = max(ecg_ch(start_idx:end_idx));
        r_peaks_ch(i) = start_idx + max_idx - 1;
    end
    r_peaks_ch = unique(r_peaks_ch);
    channel_r_peaks{ch} = r_peaks_ch(r_peaks_ch > 0 & r_peaks_ch <= n_samples);
    
    % 计算心律一致性得分
    if length(channel_r_peaks{ch}) > 2
        rr_intervals_ch = diff(channel_r_peaks{ch}) / fs;
        cv_rr = std(rr_intervals_ch) / mean(rr_intervals_ch);
        channel_detection_scores(ch) = min(1, length(channel_r_peaks{ch}) / (n_samples/fs * 1.2)) * (1 - min(0.5, cv_rr));
    else
        channel_detection_scores(ch) = 0;
    end
end

%% 4. 智能融合：构建黄金参考 ECG 信号
reference_signal = zeros(n_samples, 1);
total_weight = 0;

for ch = 1:n_channels_actual
    ch_weight = channel_quality(ch) * channel_detection_scores(ch);
    if channel_detection_scores(ch) > 0.3 && ch_weight > 0
        avg_amp = median(abs(ecg_clean_all(channel_r_peaks{ch}, ch)));
        norm_factor = 1 / max(avg_amp, 1e-3);
        
        reference_signal = reference_signal + ch_weight * norm_factor * ecg_clean_all(:, ch);
        total_weight = total_weight + ch_weight;
    end
end

if total_weight > 0
    reference_signal = reference_signal / total_weight;
else
    [~, best_ch] = max(channel_quality);
    reference_signal = ecg_clean_all(:, best_ch);
end
reference_signal = reference_signal - median(reference_signal); 

%% 5. 在融合信号上提取最终 R 波与生理参数
ecg_diff_ref = conv(reference_signal, diff_filter, 'same');
ecg_integ_ref = conv(ecg_diff_ref.^2, window, 'same');
[~, locs_ref] = findpeaks(ecg_integ_ref, 'MinPeakDistance', round(0.35 * fs), ...
                          'MinPeakHeight', median(ecg_integ_ref) + max(ecg_integ_ref)*0.2);

final_r_peaks = zeros(size(locs_ref));
for i = 1:length(locs_ref)
    start_idx = max(1, locs_ref(i) - round(0.05*fs));
    end_idx = min(n_samples, locs_ref(i) + round(0.05*fs));
    [~, max_idx] = max(reference_signal(start_idx:end_idx));
    final_r_peaks(i) = start_idx + max_idx - 1;
end
final_r_peaks = unique(final_r_peaks);
final_r_peaks(final_r_peaks <= 0 | final_r_peaks > n_samples) = [];

n_peaks = length(final_r_peaks);
r_times = t(final_r_peaks);
r_amps = reference_signal(final_r_peaks);

%% 6. 计算瞬时心率曲线
rr_intervals_raw = diff(final_r_peaks) / fs * 1000; % ms

rr_intervals_cleaned = rr_intervals_raw;
abnormal_rr = (rr_intervals_raw < 333) | (rr_intervals_raw > 2000);
if any(abnormal_rr)
    rr_intervals_cleaned(abnormal_rr) = median(rr_intervals_raw(~abnormal_rr));
end

instant_hr_all_r = 60000 ./ rr_intervals_cleaned; 
hr_times = r_times(2:end); 

instant_hr_curve = interp1(hr_times, instant_hr_all_r, t, 'pchip', 'extrap');

%% 7. 呼吸曲线提取 (EDR)
upper_env = interp1(r_times, r_amps, t, 'pchip', 'extrap');
lower_env = movmean(movmin(reference_signal, round(fs * 0.8)), round(fs * 0.8));
resp_envelope_raw = upper_env - lower_env;

% 呼吸带通滤波
[b_res, a_res] = butter(2, [0.12, 0.45]/(fs/2), 'bandpass');
resp_signal = filtfilt(b_res, a_res, resp_envelope_raw);
resp_signal = (resp_signal - mean(resp_signal)) / std(resp_signal); 

% 频域计算整体呼吸频率 (Resp Rate)
L = length(resp_signal);
Y = fft(resp_signal - mean(resp_signal));
P1 = abs(Y(1:floor(L/2)+1)/L);
f_ax = fs*(0:(L/2))/L;

valid_f_idx = (f_ax >= 0.1) & (f_ax <= 0.5);
[~, max_f_idx] = max(P1(valid_f_idx));
f_selected = f_ax(valid_f_idx);
brpm = f_selected(max_f_idx) * 60;

%% 8. 统计指标计算
mean_hr = mean(instant_hr_all_r);
sdnn = std(rr_intervals_cleaned);

fprintf('\n=== 分析统计结果 ===\n');
fprintf('平均心率 (Mean HR): %.2f bpm\n', mean_hr);
fprintf('呼吸频率 (Resp Rate): %.2f BrPM\n', brpm);
fprintf('心率变异性指标 (SDNN): %.2f ms\n', sdnn);

%% 9. 创建与保存结果表格
T_Synchronized = table(t, reference_signal, instant_hr_curve, resp_signal, ...
    'VariableNames', {'Time_s', 'Fused_Reference_ECG', 'Instant_HR_BPM', 'EDR_Respiration_Waveform'});
writetable(T_Synchronized, 'Table_Synchronized_Outputs.csv');

T_Heartbeats = table((1:length(rr_intervals_raw))', hr_times, rr_intervals_raw, rr_intervals_cleaned, instant_hr_all_r, ...
    'VariableNames', {'Beat_Index', 'Time_s', 'Original_RR_ms', 'Cleaned_RR_ms', 'Instant_HR_BPM'});
writetable(T_Heartbeats, 'Table_Heartbeat_Events.csv');

T_Summary = table({'Mean_HR'; 'Resp_Rate'; 'SDNN'; 'Total_Beats'}, ...
                  [mean_hr; brpm; sdnn; n_peaks], ...
                  {'bpm'; 'BrPM'; 'ms'; 'counts'}, ...
                  'VariableNames', {'Metric', 'Value', 'Unit'});
writetable(T_Summary, 'Table_Summary_Statistics.xlsx');
fprintf('\n[系统消息] 所有表格数据已成功导出至当前工作目录。\n');

%% 10. 绘图可视化展现
figure('Color', 'w', 'Position', [100, 100, 1200, 800], 'Name', '6-Channel Fusion & Multi-modal Physiological Analysis');

subplot(3, 1, 1);
plot(t, reference_signal, 'b', 'LineWidth', 1); hold on;
plot(t(final_r_peaks), reference_signal(final_r_peaks), 'r^', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
ylabel('ECG Amplitude');
title('Fused Multi-channel Reference ECG & Detected R-peaks');
grid on; hold off;

subplot(3, 1, 2);
plot(t, instant_hr_curve, 'k', 'LineWidth', 1.5);
ylabel('Heart Rate (BPM)');
title(sprintf('Continuous Instantaneous Heart Rate Curve (Mean HR: %.1f bpm)', mean_hr));
grid on;

subplot(3, 1, 3);
plot(t, resp_signal, 'Color', [0 0.5 0], 'LineWidth', 1.5);
ylabel('EDR Signal');
xlabel('Time (s)');
title(sprintf('ECG-Derived Respiration (EDR) Waveform (Breathing Rate: %.1f BrPM)', brpm));
grid on;

% 统一设置时间轴刻度
tick_interval = 20; 
xtick_values = 0:tick_interval:max(t);
for sp = 1:3
    subplot(3, 1, sp);
    set(gca, 'XTick', xtick_values);
    xlim([0 max(t)]);
end

sgtitle('6-Channel ECG Fusion & Multi-parameter Extraction Output');
saveas(gcf, 'ECG_Analysis_Dashboard.png');