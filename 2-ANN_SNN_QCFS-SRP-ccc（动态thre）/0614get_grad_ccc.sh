#!/bin/bash

# 设置脚本权限: chmod +x 0614get_grad_ccc.sh

# 创建输出目录
OUTPUT_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 创建结果汇总文件
SUMMARY_FILE="$OUTPUT_DIR/experiment_summary.txt"
echo "实验参数和结果汇总" > "$SUMMARY_FILE"
echo "==================" >> "$SUMMARY_FILE"
echo "开始时间: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 定义参数数组
PRUNING_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
ORDERS=("low" "high" "index" "random")
SORT_BYS=("gradient" "weight" "weight_gradient")

# 计数器
TOTAL_EXPERIMENTS=$(( ${#PRUNING_RATIOS[@]} * ${#ORDERS[@]} * ${#SORT_BYS[@]} ))
CURRENT_EXPERIMENT=0

echo "开始执行实验，总共 $TOTAL_EXPERIMENTS 个实验组合"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 遍历所有参数组合
for ratio in "${PRUNING_RATIOS[@]}"; do
    for order in "${ORDERS[@]}"; do
        for sort_by in "${SORT_BYS[@]}"; do
            CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
            
            echo "执行实验 $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
            echo "参数: ratio=$ratio, order=$order, sort_by=$sort_by"
            
            # 创建实验特定的输出文件名
            EXPERIMENT_NAME="ratio_${ratio}_order_${order}_sort_${sort_by}"
            LOG_FILE="$OUTPUT_DIR/${EXPERIMENT_NAME}.log"
            OUTPUT_FILE="$OUTPUT_DIR/${EXPERIMENT_NAME}_output.txt"
            
            # 记录实验开始
            echo "实验 $CURRENT_EXPERIMENTS/$TOTAL_EXPERIMENTS" >> "$SUMMARY_FILE"
            echo "参数: ratio=$ratio, order=$order, sort_by=$sort_by" >> "$SUMMARY_FILE"
            echo "开始时间: $(date)" >> "$SUMMARY_FILE"
            echo "日志文件: $LOG_FILE" >> "$SUMMARY_FILE"
            
            # 执行Python脚本
            echo "执行命令: python 0614get_grad_ccc.py -r $ratio --order $order --sort_by $sort_by"
            
            # 运行Python脚本并捕获输出
            python 0614get_grad_ccc.py \
                -r "$ratio" \
                --order "$order" \
                --sort_by "$sort_by" \
                2>&1 | tee "$LOG_FILE"
            
            # 检查执行结果
            EXIT_CODE=${PIPESTATUS[0]}
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo "✅ 实验成功完成"
                echo "状态: 成功" >> "$SUMMARY_FILE"
                
                # 提取关键结果（准确率和损失）
                if [ -f "$LOG_FILE" ]; then
                    echo "结果提取:" >> "$SUMMARY_FILE"
                    
                    # 提取剪枝前准确率
                    PRE_ACC=$(grep "剪枝前评估:" -A 10 "$LOG_FILE" | grep "平均准确率:" | head -1 | awk '{print $2}' | sed 's/%//')
                    if [ ! -z "$PRE_ACC" ]; then
                        echo "  剪枝前准确率: ${PRE_ACC}%" >> "$SUMMARY_FILE"
                    fi
                    
                    # 提取剪枝前损失
                    PRE_LOSS=$(grep "剪枝前评估:" -A 10 "$LOG_FILE" | grep "平均损失:" | head -1 | awk '{print $2}')
                    if [ ! -z "$PRE_LOSS" ]; then
                        echo "  剪枝前损失: $PRE_LOSS" >> "$SUMMARY_FILE"
                    fi
                    
                    # 提取剪枝后准确率
                    POST_ACC=$(grep "剪枝后评估:" -A 10 "$LOG_FILE" | grep "平均准确率:" | head -1 | awk '{print $2}' | sed 's/%//')
                    if [ ! -z "$POST_ACC" ]; then
                        echo "  剪枝后准确率: ${POST_ACC}%" >> "$SUMMARY_FILE"
                    fi
                    
                    # 提取剪枝后损失
                    POST_LOSS=$(grep "剪枝后评估:" -A 10 "$LOG_FILE" | grep "平均损失:" | head -1 | awk '{print $2}')
                    if [ ! -z "$POST_LOSS" ]; then
                        echo "  剪枝后损失: $POST_LOSS" >> "$SUMMARY_FILE"
                    fi
                    
                    # 计算准确率下降
                    if [ ! -z "$PRE_ACC" ] && [ ! -z "$POST_ACC" ]; then
                        ACC_DROP=$(echo "$PRE_ACC - $POST_ACC" | bc -l)
                        echo "  准确率下降: ${ACC_DROP}%" >> "$SUMMARY_FILE"
                    fi
                fi
            else
                echo "❌ 实验执行失败 (退出码: $EXIT_CODE)"
                echo "状态: 失败 (退出码: $EXIT_CODE)" >> "$SUMMARY_FILE"
            fi
            
            echo "结束时间: $(date)" >> "$SUMMARY_FILE"
            echo "----------------------------------------" >> "$SUMMARY_FILE"
            echo ""
            
            # 添加延迟避免系统负载过高
            sleep 2
        done
    done
done


echo "完成..."
