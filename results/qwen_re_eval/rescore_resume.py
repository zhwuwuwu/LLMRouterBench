"""支持断点续传的重新评分脚本"""
import json
import time
import sys
from evaluation.factory import EvaluatorFactory

def rescore_with_resume(dataset_name, result_file):
    print(f"开始评分: {dataset_name}")
    print(f"结果文件: {result_file}")
    
    # 加载
    result = json.load(open(result_file, 'r', encoding='utf-8'))
    factory = EvaluatorFactory()
    evaluator = factory.get_evaluator(dataset_name)
    data = evaluator.load_data('test')
    
    old_perf = result['performance']
    total = len(result['records'])
    print(f"记录数: {total}, 旧 performance: {old_perf}")
    
    # 统计已经评分的数量（score != 0.0 的）
    already_scored = sum(1 for r in result['records'] if r.get('score', 0.0) != 0.0)
    if already_scored > 0:
        print(f"[WARN] 发现 {already_scored} 条已评分记录，将跳过")
    
    # 评分
    correct = 0
    start_time = time.time()
    last_save_time = start_time
    
    for i, record in enumerate(result['records']):
        # 如果已经有非零 score，跳过
        if record.get('score', 0.0) != 0.0:
            if record['score'] == 1.0:
                correct += 1
            continue
        
        idx = record['index'] - 1
        try:
            eval_result = evaluator.evaluate(data[idx], record['raw_output'])
            new_score = 1.0 if eval_result['is_correct'] else 0.0
            
            record['score'] = new_score
            record['prediction'] = eval_result.get('prediction', record.get('prediction', ''))
            
            if new_score == 1.0:
                correct += 1
        except Exception as e:
            print(f"  [ERROR] Record {i+1} evaluation failed: {e}")
            record['score'] = 0.0  # 失败标记为 0
        
        # 每 20 条输出一次进度
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1 - already_scored) if (i + 1 - already_scored) > 0 else 0
            remaining = (total - i - 1) * avg_time
            print(f"  {i+1}/{total} ({correct}/{i+1} correct) | "
                  f"已用 {elapsed/60:.1f}分钟 | 预计剩余 {remaining/60:.1f}分钟")
        
        # 每 50 条保存一次（断点续传）
        current_time = time.time()
        if (i + 1) % 50 == 0 or (current_time - last_save_time) > 300:  # 每 50 条或每 5 分钟
            temp_perf = correct / (i + 1)
            result['performance'] = temp_perf
            json.dump(result, open(result_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
            print(f"  [SAVE] 已保存进度 (临时 performance: {temp_perf:.4f})")
            last_save_time = current_time
    
    # 最终更新
    new_perf = correct / total
    result['performance'] = new_perf
    
    # 最终保存
    json.dump(result, open(result_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n[DONE] 完成: {dataset_name}")
    print(f"   新 performance: {new_perf:.4f} ({correct}/{total})")
    print(f"   总用时: {total_time/60:.1f} 分钟")
    return new_perf, correct, total


if __name__ == '__main__':
    start = time.time()
    
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ['humaneval']
    
    if 'humaneval' in datasets:
        rescore_with_resume('humaneval', r'D:\router\LLMRouterBench\results\bench\humaneval\test\qwen3-coder-next\humaneval-test-qwen3-coder-next-20260317_012746.json')
    
    if 'mbpp' in datasets:
        rescore_with_resume('mbpp', r'D:\router\LLMRouterBench\results\bench\mbpp\test\qwen3-coder-next\mbpp-test-qwen3-coder-next-20260317_050836.json')
    
    print(f"\n总用时: {(time.time() - start)/60:.1f} 分钟")
