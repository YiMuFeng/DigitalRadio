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

def am_modulate_iq(signal, sample_rate, modulation_index=0.5, max_baseband_freq=3400):
    """
    生成AM调制的IQ信号（0中频基带信号）
    :param signal: 输入音频信号（numpy数组），采样率需与sample_rate一致
    :param sample_rate: 基带采样率 (Hz)，建议16000（匹配你的链路）
    :param modulation_index: 调制指数（0-1之间，避免过调制，默认0.5）
    :param max_baseband_freq: 基带最高频率 (Hz)，用于计算带宽，默认3400（语音）
    :return: I路信号、Q路信号（numpy数组），占用带宽 (Hz)
    """
    # 1. 信号归一化到[-1, 1]，避免过调制
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else signal

    # 2. AM调制的基带幅度成分（1 + m×signal，m为调制指数）
    # 0中频下，AM的幅度随音频变化，相位保持0（正交分量Q=0）
    amplitude = 1 + modulation_index * signal  # 幅度包络（含直流分量）

    # 3. 生成0中频IQ信号
    # AM调制的0中频IQ信号中，Q路恒为0（仅I路携带幅度信息，相位不变）
    i = amplitude * np.ones_like(signal)  # I路：幅度包络（同相分量）
    q = np.zeros_like(signal)  # Q路：0（正交分量，无信息）

    # 4. 计算AM信号带宽（2×基带最高频率，因双边带）
    bandwidth = 2 * max_baseband_freq

    return i, q, bandwidth

def fm_modulate(signal, carrier_freq, sample_rate, modulation_index=5.0,max_baseband_freq=20000):
    """
    调频（FM）调制
    :param signal: 输入音频信号（numpy数组）,与载波采样率相同的音频
    :param carrier_freq: 载波频率 (Hz)
    :param sample_rate: 采样率 (Hz)
    :param modulation_index: 调制指数（决定频率偏移程度，默认5.0）
    :return: 调制后的信号,占用的带宽
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

    return modulated,2*(max_freq_deviation+max_baseband_freq)


"""
            基带最高频率    频偏    调制指数    带宽
    窄带FM    3.4k        5.5k    1.617     11k
    FM广播    15k	     75k	 5	        180K
    WBFM语音  8k          25k     3.125       66k
	数据FM    20k	     100k	 5	        240k
    音频FM	 20k	     150k	 7.5	    340k
"""
def fm_modulate_iq(signal, sample_rate, modulation_index=5.0, max_baseband_freq=3400):
    """
    生成FM调制的IQ信号（0中频基带信号）
    :param signal: 输入音频信号（numpy数组），采样率需与sample_rate一致
    :param sample_rate: 采样率 (Hz)，建议16000（与你的基带采样率匹配）
    :param modulation_index: 调制指数（β）， 默认5.0
    :param max_baseband_freq: 基带信号最高频率 (Hz)，建议3400（语音带宽）
    :return: I路信号、Q路信号（numpy数组），占用带宽 (Hz)
    """
    # 1. 信号归一化到[-1, 1]
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else signal

    # 2. 计算最大频偏 Δf = β × fₘ（调制指数×基带最高频率）
    max_freq_deviation = modulation_index * max_baseband_freq  # 最大频偏

    # 3. 计算音频信号的积分（频率调制本质是相位的积分）
    # 积分公式：∫signal(t)dt ≈ 累加和 / 采样率
    integral = np.cumsum(signal) / sample_rate  # 得到相位的变化量

    # 4. 生成0中频IQ信号（无载波频率，后续通过数字上变频搬移）
    # 相位 θ = 2π × Δf × ∫signal(t)dt （0中频，不含载频项）
    theta = 2 * np.pi * max_freq_deviation * integral
    i = np.cos(theta)  # I路（同相分量）
    q = np.sin(theta)  # Q路（正交分量）

    # 5. 计算FM信号带宽（ Carson公式：B = 2×(Δf + fₘ)）
    bandwidth = 2 * (max_freq_deviation + max_baseband_freq)

    return i, q, bandwidth

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


def ssb_modulate_iq(signal, sample_rate, sideband='upper', max_baseband_freq=3400):
    """
    生成SSB调制的IQ信号（0中频基带信号）
    :param signal: 输入音频信号（numpy数组），采样率需与sample_rate一致
    :param sample_rate: 基带采样率 (Hz)，建议16000（匹配链路）
    :param sideband: 边带选择，'upper'（上边带）或'lower'（下边带）
    :param max_baseband_freq: 基带最高频率 (Hz)，用于计算带宽，默认3400（语音）
    :return: I路信号、Q路信号（numpy数组），占用带宽 (Hz)
    """
    # 1. 信号归一化到[-1, 1]
    signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else signal

    # 2. 希尔伯特变换：生成信号的正交分量（用于抑制一个边带）
    analytic_signal = hilbert(signal)  # 解析信号（实部为原信号，虚部为希尔伯特变换）
    hilbert_component = np.imag(analytic_signal)  # 正交分量（90°相移后的信号）

    # 3. 生成0中频SSB的IQ信号（无载波，后续通过数字上变频搬移）
    # SSB的核心是通过I/Q正交分量抵消一个边带，0中频下公式简化：
    if sideband == 'upper':
        # 上边带：I = 原信号，Q = -希尔伯特分量（抑制下边带）
        i = signal
        q = -hilbert_component
    else:
        # 下边带：I = 原信号，Q = 希尔伯特分量（抑制上边带）
        i = signal
        q = hilbert_component

    # 4. 计算SSB信号带宽（等于基带最高频率，因仅保留一个边带）
    bandwidth = max_baseband_freq  # 单边带带宽=基带带宽（如3400Hz）

    return i, q, bandwidth


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

def ask_modulate_iq(bitstream: bytes, sample_rate: int, symbol_duration: float, max_symbol_freq: float = None):
    """
    生成ASK调制的IQ信号（0中频基带信号）
    :param bitstream: 二进制比特流（0/1序列）
    :param sample_rate: 基带采样率 (Hz)，建议与链路匹配（如16000）
    :param symbol_duration: 每个符号的持续时间 (秒)，如0.01s（100波特率）
    :param max_symbol_freq: 符号最高频率 (Hz)，用于计算带宽，默认=1/(2×symbol_duration)
    :return: I路信号、Q路信号（numpy数组），占用带宽 (Hz)，时间轴
    """
    # 1. 计算符号速率与采样参数
    samples_per_symbol = int(sample_rate * symbol_duration)  # 每个符号的采样点数
    total_samples = len(bitstream) * samples_per_symbol      # 总采样点数
    symbol_rate = 1 / symbol_duration                        # 符号速率 (波特率)
    if max_symbol_freq is None:
        max_symbol_freq = symbol_rate / 2  # 矩形脉冲的最高频率约为符号速率的1/2

    # 2. 生成基带信号（矩形脉冲，0/1幅度键控）
    baseband = np.repeat(bitstream, samples_per_symbol)  # 重复每个比特到采样点

    # 3. 生成0中频ASK的IQ信号
    # ASK调制仅通过幅度变化传递信息，相位不变，因此Q路恒为0
    i = baseband  # I路：基带信号直接控制幅度（同相分量）
    q = np.zeros_like(baseband)  # Q路：0（正交分量无信息）

    # 4. 计算ASK信号带宽（2×符号最高频率，双边带）
    bandwidth = 2 * max_symbol_freq

    # 5. 生成时间轴
    t = np.linspace(
        0, len(bitstream) * symbol_duration,
        total_samples, endpoint=False
    )

    return i, q, bandwidth, t

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

def fsk_modulate_iq(bitstream, f0, f1, sample_rate, symbol_duration):
    """
    生成2FSK调制的IQ信号（0中频基带信号）
    :param bitstream: 二进制比特流（0/1序列，bytes类型）
    :param f0: 对应比特0的频率 (Hz，相对于0中频的偏移)  f0=1200Hz、
    :param f1: 对应比特1的频率 (Hz，相对于0中频的偏移)  f1=2400Hz
    :param sample_rate: 基带采样率 (Hz)，建议与链路匹配（如16000）
    :param symbol_duration: 每个符号的持续时间 (秒)
    :return: I路信号、Q路信号（numpy数组），时间轴
    """
    # 1. 计算采样参数
    samples_per_symbol = int(sample_rate * symbol_duration)  # 每个符号的采样点数
    total_samples = len(bitstream) * samples_per_symbol  # 总采样点数
    bitstream = np.frombuffer(bitstream, dtype=np.uint8)  # 将bytes转换为numpy数组（0/1）

    # 2. 初始化I/Q信号和时间轴
    i = np.zeros(total_samples)
    q = np.zeros(total_samples)
    t = np.linspace(
        0, len(bitstream) * symbol_duration,
        total_samples, endpoint=False
    )

    # 3. 逐符号生成0中频2FSK的IQ信号
    # 0中频下，频率f0/f1表现为相对于0的偏移，用正交载波cos(2πft)和sin(2πft)表示
    for idx, bit in enumerate(bitstream):
        start = idx * samples_per_symbol
        end = start + samples_per_symbol
        t_symbol = t[start:end]  # 当前符号的时间序列

        if bit == 1:
            # 比特1：用频率f1的正交信号，I=cos(2πf1*t)，Q=sin(2πf1*t)
            i[start:end] = np.cos(2 * np.pi * f1 * t_symbol)
            q[start:end] = np.sin(2 * np.pi * f1 * t_symbol)
        else:
            # 比特0：用频率f0的正交信号，I=cos(2πf0*t)，Q=sin(2πf0*t)
            i[start:end] = np.cos(2 * np.pi * f0 * t_symbol)
            q[start:end] = np.sin(2 * np.pi * f0 * t_symbol)

    return i, q, t

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


def psk_modulate_iq(bitstream, sample_rate, symbol_duration):
    """
    生成PSK调制的IQ信号（0中频基带信号），默认BPSK（2PSK）
    :param bitstream: 二进制比特流（bytes类型）
    :param sample_rate: 基带采样率 (Hz)，建议16000（匹配链路）
    :param symbol_duration: 每个符号的持续时间 (秒)
    :return: I路信号、Q路信号（numpy数组），时间轴
    """

    # 1. 计算采样参数
    samples_per_symbol = int(sample_rate * symbol_duration)  # 每个符号的采样点数
    bitstream = np.frombuffer(bitstream, dtype=np.uint8)  # bytes转0/1数组
    total_samples = len(bitstream) * samples_per_symbol  # 总采样点数

    # 2. 初始化I/Q信号和时间轴
    i = np.zeros(total_samples)
    q = np.zeros(total_samples)
    t = np.linspace(
        0, len(bitstream) * symbol_duration,
        total_samples, endpoint=False
    )

    # 3. BPSK调制逻辑：0→0°相位，1→180°相位（0中频下无载波，直接用相位表示）
    # 0°相位：I=1, Q=0；180°相位：I=-1, Q=0（幅度恒定，仅相位翻转）
    for idx, bit in enumerate(bitstream):
        start = idx * samples_per_symbol
        end = start + samples_per_symbol

        if bit == 0:
            # 0°相位：I路=1，Q路=0
            i[start:end] = 1.0
            q[start:end] = 0.0
        else:
            # 180°相位：I路=-1，Q路=0（与0°反相）
            i[start:end] = -1.0
            q[start:end] = 0.0

    return i, q, t

def qpsk_modulate_iq(bitstream: bytes, sample_rate: int, symbol_duration: float):
    """
    4PSK（QPSK）调制，生成0中频IQ信号
    :param bitstream: 二进制比特流（bytes类型）
    :param sample_rate: 基带采样率 (Hz)，建议16000
    :param symbol_duration: 每个符号的持续时间 (秒)
    :return: I路信号、Q路信号（numpy数组），时间轴，
    """
    # 1. 基本参数计算
    bits_per_symbol = 2  # 4PSK每符号2比特
    samples_per_symbol = int(sample_rate * symbol_duration)  # 每个符号的采样点数
    symbol_rate = 1 / symbol_duration  # 符号速率（波特率）

    # 2. 比特流转换为比特数组（bytes→0/1数组）
    bit_array = np.unpackbits(np.frombuffer(bitstream, dtype=np.uint8)).astype(int)
    total_bits = len(bit_array)

    # 3. 补零使总比特数为bits_per_symbol的整数倍
    padding = (bits_per_symbol - (total_bits % bits_per_symbol)) % bits_per_symbol
    bit_array = np.pad(bit_array, (0, padding), mode='constant')
    total_symbols = len(bit_array) // bits_per_symbol

    # 4. 格雷码映射表（4PSK相位：45°、135°、225°、315°，对应弧度）
    # 格雷码分组 → 相位（度）→ 弧度
    gray_map = {
        (0, 0): np.pi * 45 / 180,    # 00 → 45°
        (0, 1): np.pi * 135 / 180,   # 01 → 135°
        (1, 1): np.pi * 225 / 180,   # 11 → 225°
        (1, 0): np.pi * 315 / 180    # 10 → 315°
    }

    # 5. 初始化I/Q信号和时间轴
    total_samples = total_symbols * samples_per_symbol
    i = np.zeros(total_samples)
    q = np.zeros(total_samples)
    t = np.linspace(0, total_symbols * symbol_duration, total_samples, endpoint=False)

    # 6. 逐符号生成IQ信号（I=cos(θ), Q=sin(θ)，θ为映射相位）
    for sym_idx in range(total_symbols):
        # 提取当前符号的2比特
        bit_start = sym_idx * bits_per_symbol
        bit_end = bit_start + bits_per_symbol
        bits = tuple(bit_array[bit_start:bit_end])

        # 查找相位并计算I/Q
        theta = gray_map[bits]
        i[sym_idx*samples_per_symbol : (sym_idx+1)*samples_per_symbol] = np.cos(theta)
        q[sym_idx*samples_per_symbol : (sym_idx+1)*samples_per_symbol] = np.sin(theta)

    return i, q, t

def eightpsk_modulate_iq(bitstream: bytes, sample_rate: int, symbol_duration: float):
    """
    8PSK调制，生成0中频IQ信号
    :param bitstream: 二进制比特流（bytes类型）
    :param sample_rate: 基带采样率 (Hz)，建议16000
    :param symbol_duration: 每个符号的持续时间 (秒)
    :return: I路信号、Q路信号（numpy数组），时间轴，
    """
    # 1. 基本参数计算
    bits_per_symbol = 3  # 8PSK每符号3比特
    samples_per_symbol = int(sample_rate * symbol_duration)  # 每个符号的采样点数
    symbol_rate = 1 / symbol_duration  # 符号速率（波特率）

    # 2. 比特流转换为比特数组（bytes→0/1数组）
    bit_array = np.unpackbits(np.frombuffer(bitstream, dtype=np.uint8)).astype(int)
    total_bits = len(bit_array)

    # 3. 补零使总比特数为bits_per_symbol的整数倍
    padding = (bits_per_symbol - (total_bits % bits_per_symbol)) % bits_per_symbol
    bit_array = np.pad(bit_array, (0, padding), mode='constant')
    total_symbols = len(bit_array) // bits_per_symbol

    # 4. 格雷码映射表（8PSK相位：22.5°、67.5°、...、337.5°，对应弧度）
    # 格雷码分组 → 相位（度）→ 弧度
    gray_map = {
        (0, 0, 0): np.pi * 22.5 / 180,   # 000 → 22.5°
        (0, 0, 1): np.pi * 67.5 / 180,   # 001 → 67.5°
        (0, 1, 1): np.pi * 112.5 / 180,  # 011 → 112.5°
        (0, 1, 0): np.pi * 157.5 / 180,  # 010 → 157.5°
        (1, 1, 0): np.pi * 202.5 / 180,  # 110 → 202.5°
        (1, 1, 1): np.pi * 247.5 / 180,  # 111 → 247.5°
        (1, 0, 1): np.pi * 292.5 / 180,  # 101 → 292.5°
        (1, 0, 0): np.pi * 337.5 / 180   # 100 → 337.5°
    }

    # 5. 初始化I/Q信号和时间轴
    total_samples = total_symbols * samples_per_symbol
    i = np.zeros(total_samples)
    q = np.zeros(total_samples)
    t = np.linspace(0, total_symbols * symbol_duration, total_samples, endpoint=False)

    # 6. 逐符号生成IQ信号（I=cos(θ), Q=sin(θ)，θ为映射相位）
    for sym_idx in range(total_symbols):
        # 提取当前符号的3比特
        bit_start = sym_idx * bits_per_symbol
        bit_end = bit_start + bits_per_symbol
        bits = tuple(bit_array[bit_start:bit_end])

        # 查找相位并计算I/Q
        theta = gray_map[bits]
        i[sym_idx*samples_per_symbol : (sym_idx+1)*samples_per_symbol] = np.cos(theta)
        q[sym_idx*samples_per_symbol : (sym_idx+1)*samples_per_symbol] = np.sin(theta)

    return i, q, t


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


def fourfsk_modulate_iq(bitstream: bytes, f0: int, f1: int, f2: int, f3: int,
                        sample_rate: int, symbol_duration: float):
    """
    四进制频移键控（4FSK）的IQ调制，生成0中频基带信号
    :param bitstream: 二进制比特流（输入，bytes类型）
    :param f0/f1/f2/f3: 4个符号对应的频率偏移（Hz，相对于0中频）f0=-2400Hz, f1=-1200Hz, f2=1200Hz, f3=2400Hz
    :param sample_rate: 基带采样率（Hz），建议16000
    :param symbol_duration: 每个符号的持续时间（秒，1符号=2比特）
    :return: I路信号、Q路信号（numpy数组），时间轴，符号序列，补0后比特流
    """
    # --------------------------
    # 1. 辅助函数：比特流→4FSK符号（0-3，2比特/符号）
    # --------------------------
    def map_bits_to_4fsk_symbols(bitstream):
        bit_array = np.unpackbits(np.frombuffer(bitstream, dtype=np.uint8)).astype(int)
        total_bits = len(bit_array)
        # 补0使总比特数为2的整数倍（每个符号2比特）
        padding = (2 - (total_bits % 2)) % 2
        bit_array = np.pad(bit_array, (0, padding), mode='constant')
        # 2比特一组转换为符号（0-3）
        symbols = []
        for i in range(0, len(bit_array), 2):
            bit1, bit2 = bit_array[i], bit_array[i+1]
            symbol = bit1 * 2 + bit2  # 00→0, 01→1, 10→2, 11→3
            symbols.append(symbol)
        return np.array(symbols), bit_array  # 返回符号序列和补0后的比特数组

    # --------------------------
    # 2. 比特流转符号，计算核心参数
    # --------------------------
    symbols, padded_bit_array = map_bits_to_4fsk_symbols(bitstream)
    symbol_count = len(symbols)
    samples_per_symbol = int(sample_rate * symbol_duration)  # 每个符号的采样点数
    total_samples = symbol_count * samples_per_symbol        # 总采样点数
    total_duration = symbol_count * symbol_duration          # 总持续时间

    # --------------------------
    # 3. 生成时间轴和IQ信号容器
    # --------------------------
    t = np.linspace(0, total_duration, total_samples, endpoint=False)
    i = np.zeros(total_samples)  # I路（同相分量）
    q = np.zeros(total_samples)  # Q路（正交分量）

    # --------------------------
    # 4. 符号→频率→IQ映射（0中频核心逻辑）
    # --------------------------
    # 频率映射表：符号0→f0，符号1→f1，符号2→f2，符号3→f3
    freq_map = {0: f0, 1: f1, 2: f2, 3: f3}

    for sym_idx, symbol in enumerate(symbols):
        # 计算当前符号的采样范围
        start = sym_idx * samples_per_symbol
        end = start + samples_per_symbol
        t_symbol = t[start:end]  # 当前符号的局部时间轴

        # 获取当前符号对应的频率偏移（0中频下的频率）
        f = freq_map[symbol]

        # 生成0中频IQ信号：I=cos(2πft), Q=sin(2πft)
        # （频率f决定旋转速度，正频率顺时针旋转，负频率逆时针旋转）
        i[start:end] = np.cos(2 * np.pi * f * t_symbol)
        q[start:end] = np.sin(2 * np.pi * f * t_symbol)

    # --------------------------
    # 5. 生成基带辅助信号（用于显示）
    # --------------------------
    baseband_symbol = np.repeat(symbols, samples_per_symbol)  # 符号序列扩展为采样点长度
    padded_bitstream = np.repeat(padded_bit_array, samples_per_symbol // 2)  # 比特流扩展为采样点长度

    return i, q, t, symbols, padded_bitstream, baseband_symbol


def qam_modulate_iq(bitstream: bytes, sample_rate: int, symbol_duration: float, modulation_order: int = 16):
    """
    QAM正交振幅调制调制（16QAM/64QAM等），生成0中频IQ信号
    :param bitstream: 二进制比特流（bytes类型）
    :param sample_rate: 基带采样率 (Hz)，建议16000
    :param symbol_duration: 每个符号的持续时间 (秒)
    :param modulation_order: 调制阶数（需为2的幂，如16/64/256）
    :return: I路信号、Q路信号（numpy数组），时间轴，符号序列，星座图
    """
    # --------------------------
    # 1. 参数校验与基础计算
    # --------------------------
    if (modulation_order & (modulation_order - 1)) != 0:
        raise ValueError("调制阶数必须是2的幂（如16/64）")
    bits_per_symbol = int(np.log2(modulation_order))  # 每符号比特数（16QAM→4，64QAM→6）
    samples_per_symbol = int(sample_rate * symbol_duration)  # 每个符号的采样点数
    symbol_rate = 1 / symbol_duration  # 符号速率（波特率）

    # --------------------------
    # 2. 比特流处理：转换为比特数组并补零
    # --------------------------
    bit_array = np.unpackbits(np.frombuffer(bitstream, dtype=np.uint8)).astype(int)
    total_bits = len(bit_array)
    # 补零使总比特数为bits_per_symbol的整数倍
    padding = (bits_per_symbol - (total_bits % bits_per_symbol)) % bits_per_symbol
    bit_array = np.pad(bit_array, (0, padding), mode='constant')
    total_symbols = len(bit_array) // bits_per_symbol  # 总符号数

    # --------------------------
    # 3. 生成格雷码星座图（I/Q幅度映射）
    # --------------------------
    def generate_qam_constellation(order):
        """生成格雷码编码的QAM星座图（归一化到±1范围）"""
        n = int(np.log2(order))  # 总比特数
        n_i = n // 2  # I路比特数
        n_q = n - n_i  # Q路比特数
        i_levels = 2 **n_i  # I路幅度等级数
        q_levels = 2** n_q  # Q路幅度等级数

        # 生成I/Q幅度等级（如16QAM：I/Q各4等级，±1, ±3，归一化后±1/3, ±3/3）
        i_vals = np.linspace(-(i_levels-1), i_levels-1, i_levels)
        q_vals = np.linspace(-(q_levels-1), q_levels-1, q_levels)
        i_vals = i_vals / np.max(np.abs(i_vals))  # 归一化到[-1, 1]
        q_vals = q_vals / np.max(np.abs(q_vals))

        # 格雷码编码（相邻等级仅1比特差异）
        def gray_code(n):
            return n ^ (n >> 1)  # 格雷码转换公式

        # 构建星座图：{比特组: (I, Q)}
        constellation = {}
        for idx in range(order):
            # 拆分比特组为I路和Q路比特
            i_bits = (idx >> n_q) & ((1 << n_i) - 1)
            q_bits = idx & ((1 << n_q) - 1)
            # 格雷码映射
            i_gray = gray_code(i_bits)
            q_gray = gray_code(q_bits)
            # 转换为二进制比特组（用于匹配输入比特）
            bit_group = tuple(np.unpackbits(
                np.array([idx], dtype=np.uint8),
                bitorder='little'
            )[:n][::-1])  # 大端模式比特组（如16QAM：4比特 tuple）
            # 映射到I/Q幅度
            constellation[bit_group] = (i_vals[i_gray], q_vals[q_gray])
        return constellation

    constellation = generate_qam_constellation(modulation_order)  # 星座图

    # --------------------------
    # 4. 生成IQ信号和时间轴
    # --------------------------
    total_samples = total_symbols * samples_per_symbol
    i = np.zeros(total_samples)  # I路（同相分量，幅度调制）
    q = np.zeros(total_samples)  # Q路（正交分量，幅度调制）
    t = np.linspace(0, total_symbols * symbol_duration, total_samples, endpoint=False)
    symbols = []  # 存储符号索引（用于调试）

    # 逐符号映射比特组到I/Q幅度
    for sym_idx in range(total_symbols):
        # 提取当前符号的比特组（tuple类型，如(0,0,1,1)）
        bit_start = sym_idx * bits_per_symbol
        bit_end = bit_start + bits_per_symbol
        bit_group = tuple(bit_array[bit_start:bit_end])

        # 从星座图获取I/Q幅度
        i_amp, q_amp = constellation[bit_group]
        symbols.append(bit_group)  # 记录符号

        # 扩展到当前符号的所有采样点
        start = sym_idx * samples_per_symbol
        end = start + samples_per_symbol
        i[start:end] = i_amp  # I路幅度保持恒定（符号持续时间内）
        q[start:end] = q_amp  # Q路幅度保持恒定

    return i, q, t, symbols, constellation

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