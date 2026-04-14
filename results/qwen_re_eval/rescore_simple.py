"""简化版重新评分脚本 - 只输出关键信息"""
import json
import time
from evaluation.factory import EvaluatorFactory

def rescore(dataset_name, result_file):
    print(f"开始评分: {dataset_name}")
    
    # 加载
    result = json.load(open(result_file, 'r', encoding='utf-8'))
    factory = EvaluatorFactory()
    evaluator = factory.get_evaluator(dataset_name)
    data = evaluator.load_data('test')
    
    old_perf = result['performance']
    total = len(result['records'])
    print(f"记录数: {total}, 旧 performance: {old_perf}")
    
    # 评分
    correct = 0
    for i, record in enumerate(result['records']):
        idx = record['index'] - 1
        eval_result = evaluator.evaluate(data[idx], record['raw_output'])
        new_score = 1.0 if eval_result['is_correct'] else 0.0
        
        record['score'] = new_score
        record['prediction'] = eval_result.get('prediction', record.get('prediction', ''))
        
        if new_score == 1.0:
            correct += 1
        
        # 每 20 条输出一次进度
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} ({correct}/{i+1} correct)")
    
    # 更新
    new_perf = correct / total
    result['performance'] = new_perf
    
    # 保存
    json.dump(result, open(result_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    
    print(f"完成: {dataset_name}, 新 performance: {new_perf:.4f} ({correct}/{total})")
    return new_perf, correct, total


if __name__ == '__main__':
    import sys
    start = time.time()
    
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ['humaneval']
    
    if 'humaneval' in datasets:
        rescore('humaneval', r'D:\router\LLMRouterBench\results\bench\humaneval\test\qwen3-coder-next\humaneval-test-qwen3-coder-next-20260317_012746.json')
    
    if 'mbpp' in datasets:
        rescore('mbpp', r'D:\router\LLMRouterBench\results\bench\mbpp\test\qwen3-coder-next\mbpp-test-qwen3-coder-next-20260317_050836.json')
    
    print(f"\n总用时: {time.time() - start:.1f}秒")
