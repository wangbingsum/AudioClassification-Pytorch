import argparse
import functools

from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/panns.yml',    '配置文件')
add_arg("use_gpu",          bool,  True,                       '是否使用GPU评估模型')
add_arg('save_model',       str,   'models/',                  '模型保存的路径')
add_arg('resume_model',     str,   'models/PANNS_CNN10_MelSpectrogram/best_model/', '准备转换的模型路径')
args = parser.parse_args()
print_arguments(args=args)


# 获取训练器
trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)

# 导出预测模型
trainer.export_onnx(save_model_path=args.save_model,
               resume_model=args.resume_model)