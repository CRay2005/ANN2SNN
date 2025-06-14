import os
import re
from datetime import datetime
import numpy as np

# 定义文件夹路径
hook_outputs_dir = 'hook_outputs_ANN'
hook_outputs_snn_dir = 'hook_outputs_SNN'

# 获取文件夹中的文件列表，并按时间排序
def get_sorted_files(directory):
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.txt')]
    # 按文件名中的时间戳排序
    files.sort(key=lambda x: datetime.strptime(re.search(r'\d{8}_\d{6}_\d{6}', x).group(), '%Y%m%d_%H%M%S_%f'))
    return files

# 读取 hook_outputs 文件中的浮点数数据
def read_hook_outputs(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    pattern = r'^Output:\d+\s*$'  # 匹配 "Output:" 后接任意数字的整行
    output_indices = [i for i, line in enumerate(lines) if re.match(pattern, line.strip())]

    if not output_indices:
        raise ValueError("未找到有效的 Output 标记")

    # 取最后一个 Output 标记（可根据需求改为 output_indices[0]）
    output_line_index = output_indices[-1]


    #output_index = lines.index('Output:64\n') + 1
    output_index = output_line_index + 1

    floats = []
    for line in lines[output_index:]:
        if line.strip():
            floats.extend(map(float, line.strip().split()))
    # 将数据转换为 200 * 64 * 32 * 32 的格式
    #floats = np.array(floats).reshape(200, 64, 32, 32)
    floats = np.array(floats).reshape(200, 128, 16, 16)
    return floats

# 读取 hook_outputs-SNN 文件中的浮点数数据，并对重复的4遍数据取平均
def read_hook_outputs_snn(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    pattern = r'^Output:\d+\s*$'  # 匹配 "Output:" 后接任意数字的整行
    output_indices = [i for i, line in enumerate(lines) if re.match(pattern, line.strip())]

    if not output_indices:
        raise ValueError("未找到有效的 Output 标记")

    # 取最后一个 Output 标记（可根据需求改为 output_indices[0]）
    output_line_index = output_indices[-1]


    #output_index = lines.index('Output:64\n') + 1
    output_index = output_line_index + 1


    floats = []
    for line in lines[output_index:]:
        if line.strip():
            floats.extend(map(float, line.strip().split()))
    # 将数据转换为 200 * 4 * 64 * 32 * 32 的格式
    #floats = np.array(floats).reshape(200, 4, 64, 32, 32)
    floats = np.array(floats).reshape(200, 4, 128, 16, 16)
    # 对重复的4遍数据取平均，得到 200 * 64 * 32 * 32 的格式
    floats = np.mean(floats, axis=1)
    return floats

# 统计浮点数的大小关系
def compare_floats(hook_floats, snn_floats):
    stats = {
        'hook_gt_zero': 0,  # hook_outputs 中大于0的浮点数
        'hook_eq_zero': 0,  # hook_outputs 中等于0的浮点数
        'hook_eq_zero_snn_eq': 0,  # hook_outputs 中等于0，且 hook_outputs-SNN 等于 hook_outputs
        'hook_eq_zero_snn_gt': 0,  # hook_outputs 中等于0，且 hook_outputs-SNN 大于 hook_outputs
        'hook_gt_zero_snn_eq': 0,  # hook_outputs 中大于0，且 hook_outputs-SNN 等于 hook_outputs
        'hook_gt_zero_snn_gt': 0,  # hook_outputs 中大于0，且 hook_outputs-SNN 大于 hook_outputs
        'hook_gt_zero_snn_lt': 0,  # hook_outputs 中大于0，且 hook_outputs-SNN 小于 hook_outputs
    }

    # 遍历每个图像、每个特征值、每个像素
    for img_idx in range(hook_floats.shape[0]):
        for feat_idx in range(hook_floats.shape[1]):
            for row in range(hook_floats.shape[2]):
                for col in range(hook_floats.shape[3]):
                    hf = hook_floats[img_idx, feat_idx, row, col]
                    sf = snn_floats[img_idx, feat_idx, row, col]

                    if hf > 0:
                        stats['hook_gt_zero'] += 1
                        if sf == hf:
                            stats['hook_gt_zero_snn_eq'] += 1
                        elif sf > hf:
                            stats['hook_gt_zero_snn_gt'] += 1
                        else:
                            stats['hook_gt_zero_snn_lt'] += 1
                    elif hf == 0:
                        stats['hook_eq_zero'] += 1
                        if sf == hf:
                            stats['hook_eq_zero_snn_eq'] += 1
                        elif sf > hf:
                            stats['hook_eq_zero_snn_gt'] += 1

    return stats

# 主函数
def main():
    hook_files = get_sorted_files(hook_outputs_dir)
    snn_files = get_sorted_files(hook_outputs_snn_dir)

    if len(hook_files) != len(snn_files):
        print("两个文件夹中的文件数量不一致，请检查文件。")
        return

    # 汇总统计
    total_stats = {
        'hook_gt_zero': 0,
        'hook_eq_zero': 0,
        'hook_eq_zero_snn_eq': 0,
        'hook_eq_zero_snn_gt': 0,
        'hook_gt_zero_snn_eq': 0,
        'hook_gt_zero_snn_gt': 0,
        'hook_gt_zero_snn_lt': 0,
    }

    # 总浮点数数量
    total_floats = 0

    for hook_file, snn_file in zip(hook_files, snn_files):
        hook_floats = read_hook_outputs(os.path.join(hook_outputs_dir, hook_file))
        snn_floats = read_hook_outputs_snn(os.path.join(hook_outputs_snn_dir, snn_file))

        if hook_floats.shape != snn_floats.shape:
            print(f"文件 {hook_file} 和 {snn_file} 中的浮点数形状不一致，跳过对比。")
            continue

        # 计算当前文件的浮点数总数
        current_floats = hook_floats.size
        total_floats += current_floats

        stats = compare_floats(hook_floats, snn_floats)

        # 输出单个文件的统计信息
        print(f"文件: {hook_file} 和 {snn_file}")
        print(f"hook_outputs 中大于0的浮点数: {stats['hook_gt_zero']}")
        print(f"hook_outputs 中等于0的浮点数: {stats['hook_eq_zero']}")
        print(f"hook_outputs 中等于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: {stats['hook_eq_zero_snn_eq']}")
        print(f"hook_outputs 中等于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: {stats['hook_eq_zero_snn_gt']}")
        print(f"hook_outputs 中大于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: {stats['hook_gt_zero_snn_eq']}")
        print(f"hook_outputs 中大于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: {stats['hook_gt_zero_snn_gt']}")
        print(f"hook_outputs 中大于0，且 hook_outputs-SNN 小于 hook_outputs 的浮点数: {stats['hook_gt_zero_snn_lt']}")
        print("-" * 40)

        # 更新汇总统计
        for key in total_stats:
            total_stats[key] += stats[key]

    # 输出汇总统计信息
    print("汇总统计:")
    print(f"浮点数总数: {total_floats}")
    print(f"hook_outputs 中大于0的浮点数: {total_stats['hook_gt_zero']} (比例: {total_stats['hook_gt_zero'] / total_floats:.4f})")
    print(f"hook_outputs 中等于0的浮点数: {total_stats['hook_eq_zero']} (比例: {total_stats['hook_eq_zero'] / total_floats:.4f})")
    print(f"hook_outputs 中等于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: {total_stats['hook_eq_zero_snn_eq']} (比例: {total_stats['hook_eq_zero_snn_eq'] / total_floats:.4f})")
    print(f"hook_outputs 中等于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: {total_stats['hook_eq_zero_snn_gt']} (比例: {total_stats['hook_eq_zero_snn_gt'] / total_floats:.4f})")
    print(f"hook_outputs 中大于0，且 hook_outputs-SNN 等于 hook_outputs 的浮点数: {total_stats['hook_gt_zero_snn_eq']} (比例: {total_stats['hook_gt_zero_snn_eq'] / total_floats:.4f})")
    print(f"hook_outputs 中大于0，且 hook_outputs-SNN 大于 hook_outputs 的浮点数: {total_stats['hook_gt_zero_snn_gt']} (比例: {total_stats['hook_gt_zero_snn_gt'] / total_floats:.4f})")
    print(f"hook_outputs 中大于0，且 hook_outputs-SNN 小于 hook_outputs 的浮点数: {total_stats['hook_gt_zero_snn_lt']} (比例: {total_stats['hook_gt_zero_snn_lt'] / total_floats:.4f})")

if __name__ == "__main__":
    main()