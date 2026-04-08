#!/bin/bash

extract_field() {
  local file=$1
  local field=$2
  grep -o "\"$field\": [0-9.]*" "$file" | cut -d: -f2 | tr -d ' ' | head -1
}

# 测试
file="results/bench/arcc/test/gpt-5/arcc-test-gpt-5-20260319_000829.json"
echo "File: $file"
echo "performance: $(extract_field "$file" "performance")"
echo "counts: $(extract_field "$file" "counts")"
echo "time_taken: $(extract_field "$file" "time_taken")"
echo "cost: $(extract_field "$file" "cost")"
