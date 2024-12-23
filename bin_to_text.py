import numpy as np
import struct

def bin_to_text(input_bin_file, output_text_file, npts_x, npts_y):
    """
    将二进制文件转换为人类可读的文本文件。

    参数:
    input_bin_file (str): 输入的二进制文件路径。
    output_text_file (str): 输出的文本文件路径。
    npts_x (int): 数据网格的宽度（X方向点数）。
    npts_y (int): 数据网格的高度（Y方向点数）。
    """
    # 打开二进制文件进行读取
    with open(input_bin_file, 'rb') as f_bin:
        # 计算总元素数量
        total_elements = npts_x * npts_y
        
        # 读取整个文件内容并转换为 numpy 数组
        data = np.fromfile(f_bin, dtype=np.float64, count=total_elements)
        
        # 如果数据量小于预期，则报错
        if len(data) < total_elements:
            raise ValueError("文件包含的数据少于预期")
        
        # 重塑数组为二维形式
        data_2d = data.reshape((npts_y, npts_x))
    
    # 打开文本文件进行写入
    with open(output_text_file, 'w') as f_txt:
        for row in data_2d:
            # 将每一行的数据以空格分隔的形式写入文本文件
            line = ' '.join(f'{val:.6f}' for val in row) + '\n'
            f_txt.write(line)

if __name__ == '__main__':
    # 输入总的网格尺寸 npts_x 和 npts_y
    npts_x = 100  # 总的 x 方向点数
    npts_y = 99  # 总的 y 方向点数
    
    # 指定输入和输出文件名
    input_bin_file = 'output.bin'
    output_text_file = 'output.txt'

    # 调用函数进行转换
    bin_to_text(input_bin_file, output_text_file, npts_x, npts_y)

    print(f"转换完成: {input_bin_file} -> {output_text_file}")