# Copyright (c) 2025 Y.MF. All rights reserved.
#
# 本代码及相关文档受著作权法保护，未经授权，禁止任何形式的复制、分发、修改或商业使用。
# 如需使用或修改本代码，请联系版权所有者获得书面许可（联系方式：1428483061@qq.com）。
#
# 免责声明：本代码按"原样"提供，不提供任何明示或暗示的担保，包括但不限于对适销性、特定用途适用性的担保。
# 在任何情况下，版权所有者不对因使用本代码或本代码的衍生作品而导致的任何直接或间接损失承担责任。
#
# 项目名称：DigitalModulation.py
# 项目仓库：https://github.com/YiMuFeng/DigitalRadio.git
# 创建时间：2025/10/9 22:36
# 版权所有者：Y.MF
# 联系方式：1428483061@qq.com
# 许可协议：Apache License 2.0

"""
数字调制与解调模块
提供各种数字调制解调方法
包括数字变频和数字滤波
"""
__version__ = "0.0.0.0"

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import stft,hilbert
import matplotlib.pyplot as plt

class DigitalModulation:
    """
    数字调制模块
    实现了对数字信号的AM，FM，SSB，等调制方法
    """
    def __init__(self):
        pass

    @staticmethod
    def am_modulate(signal:np.ndarray, carrier_freq:int, sample_rate:int,
                    modulation_index:float=0.5,normalization:bool=True):
        """
        调幅（AM）调制
        注意：输入信号和载频必须具有同样的采样率，满足奈奎斯特采样定理或带通采样定理
        :param signal: 输入音频信号（numpy数组）
        :param carrier_freq: 载波频率 (Hz)
        :param sample_rate: 采样率 (Hz)
        :param modulation_index: 调制指数（0-1之间，默认0.5）
        :param normalization: 输入信号是否需要归一化
        :return: 调制后的信号
        """
        if normalization:# 确保信号归一化到[-1, 1]范围
            signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else signal

        # 生成时间轴
        t = np.arange(len(signal)) / sample_rate

        # 生成载波
        carrier = np.cos(2 * np.pi * carrier_freq * t)

        # AM调制：(1 + 调制指数*信号) * 载波
        modulated = (1 + modulation_index * signal) * carrier

        return modulated

    @staticmethod
    def fm_modulate(signal:np.ndarray, carrier_freq:int, sample_rate:int,
                    modulation_index:float=5.0,max_baseband_freq=20000,normalization:bool=True):
        """
        调频（FM）调制
        注意：输入信号和载频必须具有同样的采样率，满足奈奎斯特采样定理或带通采样定理
        :param signal: 输入音频信号（numpy数组）,与载波采样率相同的音频
        :param carrier_freq: 载波频率 (Hz)
        :param sample_rate: 采样率 (Hz)
        :param modulation_index: 调制指数（决定频率偏移程度，默认5.0）
        :param max_baseband_freq:信号最大频率（Hz)语音3.4K，音乐广播15k，高保真音乐20k
        :param normalization: 输入信号是否需要归一化
        :return: 调制后的信号
        """
        if normalization:        # 确保信号归一化到[-1, 1]范围
            signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else signal

        # 生成时间轴
        t = np.arange(len(signal)) / sample_rate

        # 计算最大频偏Δf = β×fₘ，并用于相位调制
        max_freq_deviation = modulation_index * max_baseband_freq  # Δf = β×fₘ

        # 计算信号的积分（用于频率调制）
        integral = np.cumsum(signal) / sample_rate

        # FM调制：cos(2πfc*t + 2π*调制指数*积分)
        modulated = np.cos(2 * np.pi * carrier_freq * t +
                           2 * np.pi * max_freq_deviation * integral)

        return modulated

    @staticmethod
    def ssb_modulate(signal:np.ndarray, carrier_freq:int, sample_rate:int,
                     sideband='upper',normalization:bool=True):
        """
        单边带（SSB）调制
        注意：输入信号和载频必须具有同样的采样率，满足奈奎斯特采样定理或带通采样定理
        :param signal: 输入音频信号（numpy数组）
        :param carrier_freq: 载波频率 (Hz)
        :param sample_rate: 采样率 (Hz)
        :param sideband: 边带选择，'upper'（上边带）或'lower'（下边带）
        :param normalization: 输入信号是否需要归一化
        :return: 调制后的信号
        """
        if normalization:        # 确保信号归一化到[-1, 1]范围
            signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) != 0 else signal

        # 生成时间轴
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

    def signal_spectrum(self,signal:np.ndarray, sample_rate):
        """
        获取信号的频谱
        :param signal:  输入音频信号（numpy数组）
        :param sample_rate: 采样率 (Hz)
        :return: 频谱X轴和频谱，X轴单位是Hz
        """
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, 1 / sample_rate)[:n // 2]
        yf_abs = 2.0 / n * np.abs(yf[:n // 2])  # 幅度归一化

        # plt.plot(xf, yf_abs)
        # plt.show()
        return xf, yf_abs

    def short_signal_spectrum(self,signal:np.ndarray, sample_rate:int,nperseg:int=2048):
        """
        获取信号的时频谱
        :param signal: 输入信号（numpy数组）
        :param sample_rate: 采样率 (Hz)
        :param nperseg: 窗口长度（点数）
        :return:
        """
        f, t, Zxx = stft(signal, fs=sample_rate, nperseg=nperseg)  # 2048点窗口
        # plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')  # STFT时频谱（快速）
        # plt.show()
        return f,t,Zxx

if __name__ == '__main__':
    pass
