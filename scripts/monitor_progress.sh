#!/bin/bash
# 监控 korbench 收集进度

LOG_FILE="D:/router/LLMRouterBench/korbench-final-run.log"
STATUS_FILE="D:/router/LLMRouterBench/.korbench_progress"

while true; do
    # 检查进程是否还在运行
    if ! ps aux | grep 6341 | grep -v grep > /dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 进程 6341 已停止" >> "$STATUS_FILE"
        break
    fi
    
    # 提取最新进度
    PROGRESS=$(tail -1 "$LOG_FILE" | grep -oP '\d+/1250' | tail -1)
    SPEED=$(tail -1 "$LOG_FILE" | grep -oP '\d+\.\d+s/record' | tail -1)
    
    if [ -n "$PROGRESS" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 进度: $PROGRESS, 速度: $SPEED" >> "$STATUS_FILE"
    fi
    
    # 每 10 分钟检查一次
    sleep 600
done
