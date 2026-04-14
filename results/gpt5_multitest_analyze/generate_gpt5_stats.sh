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

total_cost_e=0
total_time_e=0
total_counts_e=0
total_prompt_e=0
total_completion_e=0
perf_sum_e=0
count_e=0

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
    
    printf "| %-10s | %8s | %6s | %8s | %7s | %13s | %17s |\n" "$ds" "$perf" "$counts" "$time" "$cost" "$prompt" "$completion"
    
    total_cost_e=$(echo "$total_cost_e + $cost" | bc)
    total_time_e=$(echo "$total_time_e + $time" | bc)
    total_counts_e=$((total_counts_e + counts))
    total_prompt_e=$((total_prompt_e + prompt))
    total_completion_e=$((total_completion_e + completion))
    perf_sum_e=$(echo "$perf_sum_e + $perf" | bc)
    count_e=$((count_e + 1))
  fi
done

avg_perf_e=$(echo "scale=4; $perf_sum_e / $count_e" | bc)
echo "|--------|--------|--------|----------|---------|---------------|-------------------|"
printf "| **总计** | **%8s** | **%6d** | **%8s** | **%7s** | **%13d** | **%17d** |\n" "$avg_perf_e" "$total_counts_e" "$total_time_e" "$total_cost_e" "$total_prompt_e" "$total_completion_e"

echo ""
echo ""
echo "## GPT-5 性能统计 (最晚测试)"
echo ""
echo "| 数据集 | 准确率 | 样本数 | 耗时(秒) | 成本(\$) | Prompt Tokens | Completion Tokens |"
echo "|--------|--------|--------|----------|---------|---------------|-------------------|"

total_cost_l=0
total_time_l=0
total_counts_l=0
total_prompt_l=0
total_completion_l=0
perf_sum_l=0
count_l=0

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
    
    printf "| %-10s | %8s | %6s | %8s | %7s | %13s | %17s |\n" "$ds" "$perf" "$counts" "$time" "$cost" "$prompt" "$completion"
    
    total_cost_l=$(echo "$total_cost_l + $cost" | bc)
    total_time_l=$(echo "$total_time_l + $time" | bc)
    total_counts_l=$((total_counts_l + counts))
    total_prompt_l=$((total_prompt_l + prompt))
    total_completion_l=$((total_completion_l + completion))
    perf_sum_l=$(echo "$perf_sum_l + $perf" | bc)
    count_l=$((count_l + 1))
  fi
done

avg_perf_l=$(echo "scale=4; $perf_sum_l / $count_l" | bc)
echo "|--------|--------|--------|----------|---------|---------------|-------------------|"
printf "| **总计** | **%8s** | **%6d** | **%8s** | **%7s** | **%13d** | **%17d** |\n" "$avg_perf_l" "$total_counts_l" "$total_time_l" "$total_cost_l" "$total_prompt_l" "$total_completion_l"
