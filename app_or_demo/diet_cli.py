"""真实模型演示入口：输出严格任务二 JSON，无硬编码预测。"""
import argparse
import json
from task2_diet_record.interfaces import TransformersDietModel
from task4_nutrition_estimation.baseline import IngredientNutritionBaseline, NutritionEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'xpu'])
    parser.add_argument('--method', default='direct', choices=['direct', 'ingredients'])
    parser.add_argument('--baseline')
    args = parser.parse_args()
    if args.method == 'ingredients' and not args.baseline:
        parser.error('ingredients method requires --baseline')
    model = TransformersDietModel.from_pretrained(args.model_path, args.adapter, args.device)
    baseline = IngredientNutritionBaseline.load(args.baseline) if args.baseline else None
    print(json.dumps(NutritionEstimator(model, baseline).predict(args.image, args.method), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
