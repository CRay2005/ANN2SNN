#!/bin/bash

# 设置脚本权限: chmod +x 0629get_grad_ccc_weight_in_out.sh

# 定义剪枝比例数组
PRUNING_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# 存储结果的数组
RATIO_LIST=""
ACCURACY_LIST=""
LOSS_LIST=""

# 遍历所有剪枝比例
for ratio in "${PRUNING_RATIOS[@]}"; do
    echo "正在执行 r=$ratio..."
    
    # 执行Python脚本并捕获输出
    OUTPUT=$(python 0614get_grad_ccc.py -r "$ratio" 2>&1)
    
    # 提取剪枝后准确率和损失
    POST_ACC=$(echo "$OUTPUT" | grep "剪枝后评估:" -A 2 | grep "平均准确率:" | awk '{print $2}' | sed 's/%//')
    POST_LOSS=$(echo "$OUTPUT" | grep "剪枝后评估:" -A 2 | grep "平均损失:" | awk '{print $2}')
    
    # 控制损失精度为小数点后3位
    POST_LOSS_FORMATTED=$(printf "%.3f" "$POST_LOSS")
    
    # 显示当前进度结果
    echo "r=$ratio 完成: 准确率=${POST_ACC}%, 损失=${POST_LOSS_FORMATTED}"
    
    # 构建结果列表
    if [ -z "$RATIO_LIST" ]; then
        RATIO_LIST="$ratio"
        ACCURACY_LIST="${POST_ACC}%"
        LOSS_LIST="$POST_LOSS_FORMATTED"
    else
        RATIO_LIST="$RATIO_LIST $ratio"
        ACCURACY_LIST="$ACCURACY_LIST ${POST_ACC}%"
        LOSS_LIST="$LOSS_LIST $POST_LOSS_FORMATTED"
    fi
done

# 输出格式化结果
echo ""
echo "========== 最终汇总结果 =========="
echo "$RATIO_LIST"
echo "$ACCURACY_LIST $LOSS_LIST"
