#!/bin/bash

extract_field() {
  local file=$1
  local field=$2
  grep -o "\"$field\": [0-9.]*" "$file" | cut -d: -f2 | tr -d ' ' | head -1
}

echo "## GPT-5 性能统计 (最早测试)"
echo ""
echo "| 数据集 | 准确率 | 样本数 | 耗时(秒) | 成本(\$) | Prompt Tokens | Completion Tokens |"
echo "|--------|--------|--------|----------|---------|---------------|-------------------|"

{
for ds in arcc bbh emorynlp finqa humaneval kandk math500 mathbench mbpp medqa meld winogrande; do
  split="test"
  [ "$ds" = "winogrande" ] && split="valid"
  
  gpt5_dir="results/bench/$ds/$split/gpt-5"
  earliest=$(ls "$gpt5_dir"/*gpt-5-*.json 2>/dev/null | grep -v "gpt-5-chat" | head -1)
  
  if [ -n "$earliest" ]; then
    perf=$(extract_field "$earliest" "performance")
    counts=$(extract_field "$earliest" "counts")
    time=$(extract_field "$earliest" "time_taken")
    cost=$(extract_field "$earliest" "cost")
    prompt=$(extract_field "$earliest" "prompt_tokens")
    completion=$(extract_field "$earliest" "completion_tokens")
    
    printf "| %-10s | %8.4f | %6d | %8.2f | %7.2f | %13d | %17d |\n" "$ds" "$perf" "$counts" "$time" "$cost" "$prompt" "$completion"
    
    # 输出数据以供awk处理
    echo "$perf $time $cost $counts $prompt $completion"
  fi
done
} | tee /tmp/earliest_data.txt | head -12

# 使用awk计算总和
echo "|--------|--------|--------|----------|---------|---------------|-------------------|"
awk '
NF == 6 {
  perf_sum += $1
  time_sum += $2
  cost_sum += $3
  counts_sum += $4
  prompt_sum += $5
  completion_sum += $6
  count++
}
END {
  if (count > 0) {
    avg_perf = perf_sum / count
    printf "| **总计** | **%.4f** | **%d** | **%.2f** | **%.2f** | **%d** | **%d** |\n", avg_perf, counts_sum, time_sum, cost_sum, prompt_sum, completion_sum
  }
}
' /tmp/earliest_data.txt

echo ""
echo ""
echo "## GPT-5 性能统计 (最晚测试)"
echo ""
echo "| 数据集 | 准确率 | 样本数 | 耗时(秒) | 成本(\$) | Prompt Tokens | Completion Tokens |"
echo "|--------|--------|--------|----------|---------|---------------|-------------------|"

{
for ds in arcc bbh emorynlp finqa humaneval kandk math500 mathbench mbpp medqa meld winogrande; do
  split="test"
  [ "$ds" = "winogrande" ] && split="valid"
  
  gpt5_dir="results/bench/$ds/$split/gpt-5"
  latest=$(ls "$gpt5_dir"/*gpt-5-*.json 2>/dev/null | grep -v "gpt-5-chat" | tail -1)
  
  if [ -n "$latest" ]; then
    perf=$(extract_field "$latest" "performance")
    counts=$(extract_field "$latest" "counts")
    time=$(extract_field "$latest" "time_taken")
    cost=$(extract_field "$latest" "cost")
    prompt=$(extract_field "$latest" "prompt_tokens")
    completion=$(extract_field "$latest" "completion_tokens")
    
    printf "| %-10s | %8.4f | %6d | %8.2f | %7.2f | %13d | %17d |\n" "$ds" "$perf" "$counts" "$time" "$cost" "$prompt" "$completion"
    
    echo "$perf $time $cost $counts $prompt $completion"
  fi
done
} | tee /tmp/latest_data.txt | head -12

echo "|--------|--------|--------|----------|---------|---------------|-------------------|"
awk '
NF == 6 {
  perf_sum += $1
  time_sum += $2
  cost_sum += $3
  counts_sum += $4
  prompt_sum += $5
  completion_sum += $6
  count++
}
END {
  if (count > 0) {
    avg_perf = perf_sum / count
    printf "| **总计** | **%.4f** | **%d** | **%.2f** | **%.2f** | **%d** | **%d** |\n", avg_perf, counts_sum, time_sum, cost_sum, prompt_sum, completion_sum
  }
}
' /tmp/latest_data.txt
