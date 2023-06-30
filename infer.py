import argparse
import functools

from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/panns.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('feature_save_path',str,    'output/test_data_1_98_64_1.bin',       '保存实际音频文件提取的特征数据')
add_arg('audio_path',       str,    '/root/autodl-tmp/UrbanSound8K/audio/fold5/156634-5-2-5.wav', '音频路径')
add_arg('model_path',       str,    'models/PANNS_CNN10_MelSpectrogram/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

label, score = predictor.predict(audio_data=args.audio_path, feature_save_path=args.feature_save_path)

print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
