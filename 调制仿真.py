import os
import time

import matplotlib.pyplot as plt
import librosa
import librosa.display
from mutagen.mp3 import MP3
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert  # 使用scipy的正确希尔伯特变换实现


def load_mp3(file_path):
    """加载MP3音频文件"""
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在")
        return None
    # 检查文件是否为MP3
    if not file_path.lower().endswith('.mp3'):
        print(f"错误：'{file_path}' 不是MP3文件")
        return None

    try:
        audio = MP3(file_path)
        print("MP3文件基本属性：")
        print(f"文件名: {os.path.basename(file_path)}")
        print(f"文件大小: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
        print(f"时长: {audio.info.length:.2f} 秒 ({audio.info.length / 60:.2f} 分钟)")
        print(f"比特率: {audio.info.bitrate / 1000:.2f} kbps")
        print(f"采样率: {audio.info.sample_rate} Hz")
        print(f"声道数: {'单声道' if audio.info.channels == 1 else '立体声'}")

        y, sr = librosa.load(file_path, sr=None)  # 加载音频，保持原始采样率
        print(f"音频序列长度{y.shape,type(y)},已归一化到单声道，幅度1")

        return y, sr
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return None


def am_modulate(signal, carrier_freq, sample_rate, modulation_index=0.5):
    """
    调幅（AM）调制
    :param signal: 输入音频信号（numpy数组）
    :param carrier_freq: 载波频率 (Hz)
    :param sample_rate: 采样率 (Hz)
    :param modulation_index: 调制指数（0-1之间，默认0.5）
    :return: 调制后的信号
    """
    # 确保信号归一化到[-1, 1]范围
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else signal

    # 生成时间序列
    t = np.arange(len(signal)) / sample_rate
    # print(type(t),t.shape.)
    # 生成载波
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    # plt.plot(range(len(carrier)), carrier)
    # plt.show()
    # AM调制：(1 + 调制指数*信号) * 载波
    modulated = (1 + modulation_index * signal) * carrier

    return modulated



def fm_modulate(signal, carrier_freq, sample_rate, modulation_index=5.0,max_baseband_freq=20000):
    """
    调频（FM）调制
    :param signal: 输入音频信号（numpy数组）,与载波采样率相同的音频
    :param carrier_freq: 载波频率 (Hz)
    :param sample_rate: 采样率 (Hz)
    :param modulation_index: 调制指数（决定频率偏移程度，默认5.0）
    :return: 调制后的信号
    """
    # 确保信号归一化到[-1, 1]范围
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else signal

    # 生成时间序列
    t = np.arange(len(signal)) / sample_rate

    #计算最大频偏Δf = β×fₘ，并用于相位调制
    max_freq_deviation = modulation_index * max_baseband_freq  # Δf = β×fₘ

    # 计算信号的积分（用于频率调制）
    integral = np.cumsum(signal) / sample_rate

    # FM调制：cos(2πfc*t + 2π*调制指数*积分)
    modulated = np.cos(2 * np.pi * carrier_freq * t
                       + 2 * np.pi * max_freq_deviation * integral)

    return modulated


def ssb_modulate(signal, carrier_freq, sample_rate, sideband='upper'):
    """
    单边带（SSB）调制
    :param signal: 输入音频信号（numpy数组）
    :param carrier_freq: 载波频率 (Hz)
    :param sample_rate: 采样率 (Hz)
    :param sideband: 边带选择，'upper'（上边带）或'lower'（下边带）
    :return: 调制后的信号
    """
    # 确保信号归一化到[-1, 1]范围
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else signal

    # 生成时间序列
    t = np.arange(len(signal)) / sample_rate

    # 希尔伯特变换
    analytic_signal = hilbert(signal)
    hilbert_component = np.imag(analytic_signal)  # 正交分量（希尔伯特变换结果）


    # 生成同相和正交载波
    carrier_i = np.cos(2 * np.pi * carrier_freq * t)  # 同相载波
    carrier_q = np.sin(2 * np.pi * carrier_freq * t)  # 正交载波

    # 单边带调制
    if sideband == 'upper':
        # 上边带：信号*cos(ωct) - 希尔伯特变换*sin(ωct)
        modulated = signal * carrier_i - hilbert_component * carrier_q
    else:
        # 下边带：信号*cos(ωct) + 希尔伯特变换*sin(ωct)
        modulated = signal * carrier_i + hilbert_component * carrier_q

    return modulated


def signal_spectrum(signal, sample_rate):
    """绘制信号的频谱
    """
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / sample_rate)[:n // 2]
    yf_abs = 2.0 / n * np.abs(yf[:n // 2])  # 幅度归一化

    return xf, yf_abs


def ask_modulate(bitstream:bytes, carrier_freq:int, sample_rate:int, symbol_duration:float):
    """
    幅移键控(ASK)调制
    :param bitstream: 二进制比特流
    :param carrier_freq: 载波频率
    :param sample_rate: 采样率
    :param symbol_duration: 每个符号的持续时间(秒)
    :return: 调制后的信号、时间轴
    """
    # 每个符号的采样点数
    samples_per_symbol = int(sample_rate * symbol_duration)
    # 总采样点数
    total_samples = len(bitstream) * samples_per_symbol

    # 生成时间轴
    t = np.linspace(0, len(bitstream) * symbol_duration, total_samples, endpoint=False)

    # 生成基带信号(矩形脉冲)
    baseband = np.repeat(bitstream, samples_per_symbol)

    # 生成载波
    carrier = np.cos(2 * np.pi * carrier_freq * t)

    # ASK调制: 用基带信号控制载波幅度
    modulated = baseband * carrier

    return modulated, t, baseband


def fsk_modulate(bitstream, f0, f1, sample_rate, symbol_duration):
    """
    频移键控(FSK)调制
    :param bitstream: 二进制比特流
    :param f0: 对应比特0的频率
    :param f1: 对应比特1的频率
    :param sample_rate: 采样率
    :param symbol_duration: 每个符号的持续时间(秒)
    :return: 调制后的信号、时间轴
    """
    samples_per_symbol = int(sample_rate * symbol_duration)
    total_samples = len(bitstream) * samples_per_symbol
    t = np.linspace(0, len(bitstream) * symbol_duration, total_samples, endpoint=False)

    # 初始化调制信号
    modulated = np.zeros(total_samples)

    # 为每个比特分配相应的频率
    for i, bit in enumerate(bitstream):
        start_idx = i * samples_per_symbol
        end_idx = start_idx + samples_per_symbol
        t_symbol = t[start_idx:end_idx]

        if bit == 1:
            modulated[start_idx:end_idx] = np.cos(2 * np.pi * f1 * t_symbol)
        else:
            modulated[start_idx:end_idx] = np.cos(2 * np.pi * f0 * t_symbol)

    # 生成基带信号
    baseband = np.repeat(bitstream, samples_per_symbol)

    return modulated, t, baseband


def psk_modulate(bitstream, carrier_freq, sample_rate, symbol_duration):
    """
    相移键控(PSK)调制
    BPSK常用差分编码，QAM/QPSK常用导频的方式，MSK常用训练序列，OFDM常用导频子载波
    :param bitstream: 二进制比特流
    :param carrier_freq: 载波频率
    :param sample_rate: 采样率
    :param symbol_duration: 每个符号的持续时间(秒)
    :return: 调制后的信号、时间轴
    """
    samples_per_symbol = int(sample_rate * symbol_duration)
    total_samples = len(bitstream) * samples_per_symbol
    t = np.linspace(0, len(bitstream) * symbol_duration, total_samples, endpoint=False)

    # 生成载波
    carrier = np.cos(2 * np.pi * carrier_freq * t)

    # 生成相位偏移: 0对应0°，1对应180°
    phase_shift = np.repeat(bitstream * np.pi, samples_per_symbol)  # 180°相位差

    # PSK调制: 用相位偏移控制载波相位
    modulated = np.cos(2 * np.pi * carrier_freq * t + phase_shift)

    # 生成基带信号
    baseband = np.repeat(bitstream, samples_per_symbol)

    return modulated, t, baseband


def map_bits_to_4fsk_symbols(bitstream):
    """
    将二进制比特流按2位分组，映射为4FSK符号（0/1/2/3）
    映射规则：00→0，01→1，10→2，11→3（可自定义）
    """
    # 确保比特流长度为2的倍数，不足则补0（避免分组残留）
    if len(bitstream) % 2 != 0:
        bitstream = np.append(bitstream, 0)  # 补0处理

    # 按2位分组，转换为十进制符号（0-3）
    symbol_count = len(bitstream) // 2
    symbols = np.zeros(symbol_count, dtype=int)

    for i in range(symbol_count):
        # 取当前组的2个比特（高位在前，低位在后）
        bit1 = bitstream[2 * i]
        bit2 = bitstream[2 * i + 1]
        # 二进制转十进制（如 01 → 0*2 + 1 = 1）
        symbols[i] = bit1 * 2 + bit2

    return symbols


def fourfsk_modulate(bitstream, f0, f1, f2, f3, sample_rate, symbol_duration):
    """
    四进制频移键控（4FSK）调制
    :param bitstream: 二进制比特流（输入）
    :param f0/f1/f2/f3: 对应4个符号（0/1/2/3）的载波频率（Hz）
    :param sample_rate: 采样率（Hz）
    :param symbol_duration: 每个符号的持续时间（秒，每个符号对应2个比特）
    :return: 调制后的4FSK信号、时间轴、基带符号序列、基带比特流（补0后）
    """
    # 1. 比特流→4FSK符号（0-3）
    symbols = map_bits_to_4fsk_symbols(bitstream)
    symbol_count = len(symbols)

    # 2. 计算关键参数
    samples_per_symbol = int(sample_rate * symbol_duration)  # 每个符号的采样点数
    total_samples = symbol_count * samples_per_symbol  # 总采样点数
    total_duration = symbol_count * symbol_duration  # 总持续时间

    # 3. 生成时间轴（覆盖所有符号）
    t = np.linspace(0, total_duration, total_samples, endpoint=False)

    # 4. 初始化调制信号
    modulated = np.zeros(total_samples)

    # 5. 符号→频率映射，填充调制信号
    # 频率映射表：符号0→f0，符号1→f1，符号2→f2，符号3→f3（需确保频率间隔合理）
    freq_map = {0: f0, 1: f1, 2: f2, 3: f3}

    for i, symbol in enumerate(symbols):
        # 计算当前符号的采样点范围
        start_idx = i * samples_per_symbol
        end_idx = start_idx + samples_per_symbol
        t_symbol = t[start_idx:end_idx]  # 当前符号的局部时间轴

        # 根据符号选择频率，生成载波
        carrier_freq = freq_map[symbol]
        modulated[start_idx:end_idx] = np.cos(2 * np.pi * carrier_freq * t_symbol)

    # 6. 生成基带信号（用于波形对比：符号序列重复为采样点长度）
    baseband_symbol = np.repeat(symbols, samples_per_symbol)
    # 补0后的比特流（用于显示输入数据）
    padded_bitstream = np.repeat(bitstream, samples_per_symbol // 2)  # 每个比特对应 samples_per_symbol/2 个采样点

    return modulated, t, symbols, padded_bitstream, baseband_symbol


def QPSK():
    """正交相移键控，扩展OQPSK"""
    pass
def QAM():
    """正交振幅调制"""
    pass
def OFDM():
    """正交频分复用"""
    pass
def MSK():
    """最小频移键控，扩展GMSK"""
    pass


if __name__ == "__main__":
    #
    # x = np.linspace(0, 1, 4000, endpoint=False)
    # signal = 0.7 * np.cos(2 * np.pi * 100 * x) + 0.3 * np.cos(2 * np.pi * 800 * x)
    # xs, ffts = signal_spectrum(signal, 4000)
    #
    #
    # # # 载波与目标采样率参数（60MHz载波）
    # carrier_freq = 1e6  # 1MHz载波频率
    # target_sr = 3e6  # 目标采样率
    # # # 音频重采样
    # resampled_signal = librosa.resample(
    #     y=signal.astype(np.float32),  # 音频信号（需float32类型）
    #     orig_sr=4000,  # 原始采样率
    #     target_sr=target_sr  # 目标采样率
    # )
    # xres, fftres = signal_spectrum(resampled_signal, target_sr)
    # #
    # # # am调制
    # amsignal = am_modulate(resampled_signal, carrier_freq, target_sr)
    # xam, fftam = signal_spectrum(amsignal, target_sr)
    # #
    # plt.subplot(6, 1, 1)
    # plt.plot(x, signal, "-")
    # plt.subplot(6, 1, 2)
    # plt.plot(xs, ffts)
    #
    # plt.subplot(6, 1, 4)
    # plt.plot(xres, fftres)
    # plt.subplot(6, 1, 5)
    # plt.plot(range(len(amsignal)),amsignal)
    # plt.subplot(6, 1, 6)
    # plt.plot(xam,fftam)
    # plt.show()

    # 可以修改为你要分析的MP3文件路径
    mp3_file = "Assets/6B052518A42ACF20.mp3"  # 替换为实际的MP3文件路径

    signal,sr = load_mp3(mp3_file)

    signal = signal[3400000:3408820]        # 截取0.2s：
    xs,ffts = signal_spectrum(signal,sr)

    # 载波与目标采样率参数（60MHz载波）
    carrier_freq = 200000  # 1MHz载波频率
    target_sr = 500000  # 目标采样率130MHz（>2×60MHz，满足奈奎斯特准则）

    # 音频重采样
    start = time.time()
    resampled_signal = librosa.resample(
        y=signal.astype(np.float32),  # 音频信号（需float32类型）
        orig_sr=sr,  # 原始采样率
        target_sr=target_sr  # 目标采样率
    )
    # xres, fftres = signal_spectrum(resampled_signal, target_sr)

    # am调制
    amsignal = ssb_modulate(resampled_signal, carrier_freq, target_sr ) #,sideband="lower"
    xam,fftam = signal_spectrum(amsignal, target_sr)
    print(xam[:100],fftam.shape)
    end = time.time()
    # print(f"am调制后的信号长度{len(amsignal)}频谱长度,插值和调制用时{end-start}")
    print(f"fm调制后的信号长度{len(amsignal)}频谱长度{len(xam), len(fftam)},插值和调制用时{end - start}")
    # amsignal = fm_modulate(signal, 430000000, sr)
    # xff,fftfm = signal_spectrum(amsignal,sr)
    #
    # ssbsignal = ssb_modulate(signal, 430000000, sr)
    # xfss,fftssb = signal_spectrum(ssbsignal,sr)

    plt.subplot(411)
    plt.plot(range(len(signal)),signal,"r-")
    plt.subplot(412)
    plt.plot(xs,ffts,"r-")
    # plt.subplot(413)
    # plt.plot(range(len(resampled_signal)),resampled_signal,"g-")
    # plt.subplot(414)
    # plt.plot(xres[:14740], fftres[:14740],"g-")

    plt.subplot(413)
    plt.plot(range(len(amsignal)), amsignal, "b-")
    plt.plot(range(len(resampled_signal)),resampled_signal,"r-")
    plt.subplot(414)
    # plt.plot( xam[300000//5:305000//5],fftam[300000//5:305000//5],"b-")
    plt.plot( xam,fftam,"b-")
    # plt.subplot(815)
    # plt.plot(range(10000),amsignal,"k-")
    # plt.subplot(816)
    # plt.plot(xff,fftfm,"k-")
    # plt.subplot(817)
    # plt.plot(range(10000),ssbsignal,"b-")
    # plt.subplot(818)
    # plt.plot(xfss,fftssb,"b-")

    plt.show()

    # plot_wavelet_time_freq(
    #     resampled_signal,
    #     target_sr,
    #     wavelet='morl',
    #     # title='测试信号的小波时频谱'
    # )
    # plt.show()
    #
    # plot_wavelet_time_freq(
    #     amsignal,
    #     target_sr,
    #     wavelet='morl',
    #     # title='测试信号的小波时频谱'
    # )
    # plt.show()
    from scipy.signal import stft

    f, t, Zxx = stft(amsignal, fs=target_sr, nperseg=2048)  # 2048点窗口
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')  # STFT时频谱（快速）
    plt.show()