"""
重新评分 qwen3-coder-next 的 HumanEval 和 MBPP 结果。
从已有结果文件读取 raw_output，用修复后的 evaluator 重新计算 score。
"""
import json
import os
import sys
import time
from datetime import datetime

# 确保工作目录正确
os.chdir(r'D:\router\LLMRouterBench')

from evaluation.factory import EvaluatorFactory

def rescore_dataset(dataset_name, result_file):
    """对单个数据集重新评分"""
    print(f"\n{'='*60}")
    print(f"重新评分: {dataset_name}")
    print(f"结果文件: {result_file}")
    print(f"{'='*60}")
    
    # 读取旧结果
    with open(result_file, 'r', encoding='utf-8') as f:
        old_result = json.load(f)
    
    print(f"旧 performance: {old_result['performance']}")
    print(f"记录数: {old_result['counts']}")
    
    # 创建 evaluator 并加载数据
    factory = EvaluatorFactory()
    evaluator = factory.get_evaluator(dataset_name)
    data = evaluator.load_data('test')
    
    print(f"数据集大小: {len(data)}")
    
    # 重新评分每条记录
    new_records = []
    correct_count = 0
    total_count = 0
    errors = []
    
    for i, record in enumerate(old_result['records']):
        idx = record['index'] - 1  # 1-based to 0-based
        raw_output = record.get('raw_output', '')
        
        if idx >= len(data):
            print(f"  ⚠️ index {record['index']} 超出数据集范围 ({len(data)}), 跳过")
            errors.append(f"index {record['index']} out of range")
            continue
        
        data_item = data[idx]
        
        try:
            eval_result = evaluator.evaluate(data_item, raw_output)
            new_score = 1.0 if eval_result['is_correct'] else 0.0
            new_prediction = eval_result.get('prediction', record.get('prediction', ''))
        except Exception as e:
            print(f"  ⚠️ index {record['index']} 评分失败: {str(e)[:100]}")
            new_score = 0.0
            new_prediction = record.get('prediction', '')
            errors.append(f"index {record['index']}: {str(e)[:100]}")
        
        total_count += 1
        if new_score == 1.0:
            correct_count += 1
        
        # 更新记录
        new_record = {**record}
        new_record['score'] = new_score
        new_record['prediction'] = new_prediction
        new_records.append(new_record)
        
        # 进度显示
        if (i + 1) % 50 == 0 or (i + 1) == len(old_result['records']):
            print(f"  进度: {i+1}/{len(old_result['records'])} | 正确: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    
    # 计算新 performance
    new_performance = correct_count / total_count if total_count > 0 else 0.0
    
    print(f"\n结果:")
    print(f"  旧 performance: {old_result['performance']:.4f}")
    print(f"  新 performance: {new_performance:.4f}")
    print(f"  正确数: {correct_count}/{total_count}")
    if errors:
        print(f"  错误数: {len(errors)}")
        for e in errors[:5]:
            print(f"    - {e}")
    
    # 构建新结果（直接覆盖旧文件）
    new_result = {**old_result}
    new_result['performance'] = new_performance
    new_result['records'] = new_records
    
    # 覆盖原文件
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(new_result, f, ensure_ascii=False, indent=2)
    
    print(f"  已更新原文件: {result_file}")
    
    return new_performance, correct_count, total_count


if __name__ == '__main__':
    import sys
    
    start_time = time.time()
    
    results = {}
    
    # 只处理命令行指定的数据集，默认只跑 HumanEval
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ['humaneval']
    
    if 'humaneval' in datasets:
        # HumanEval
        he_file = r'D:\router\LLMRouterBench\results\bench\humaneval\test\qwen3-coder-next\humaneval-test-qwen3-coder-next-20260317_012746.json'
        if os.path.exists(he_file):
            perf, correct, total = rescore_dataset('humaneval', he_file)
            results['humaneval'] = {'performance': perf, 'correct': correct, 'total': total}
        else:
            print(f"文件不存在: {he_file}")
    
    if 'mbpp' in datasets:
        # MBPP
        mbpp_file = r'D:\router\LLMRouterBench\results\bench\mbpp\test\qwen3-coder-next\mbpp-test-qwen3-coder-next-20260317_050836.json'
        if os.path.exists(mbpp_file):
            perf, correct, total = rescore_dataset('mbpp', mbpp_file)
            results['mbpp'] = {'performance': perf, 'correct': correct, 'total': total}
        else:
            print(f"文件不存在: {mbpp_file}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name}: {r['performance']*100:.1f}% ({r['correct']}/{r['total']})")
    print(f"  总用时: {elapsed:.1f}秒")
    print(f"\n完成！旧的 performance=0.0 已更新为实际评分结果。")
    
    if 'mbpp' not in datasets:
        print(f"\n提示: MBPP 有 974 条记录，预计需要 30 分钟。")
        print(f"如需评分 MBPP，运行: python rescore.py humaneval mbpp")
