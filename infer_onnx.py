import argparse
import functools

from macls.predict import MAClsPredictor
from macls.utils.utils import (
    add_arguments, 
    print_arguments, 
    dict_to_object
)
from macls.data_utils.audio import AudioSegment
from macls.data_utils.featurizer import AudioFeaturizer

import onnxruntime as ort
import numpy as np
import yaml
import torch

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/panns.yml',   '配置文件')
add_arg('feature_save_path',str,    'output/test_data_1_98_64_1.bin',       '保存实际音频文件提取的特征数据')
add_arg('audio_path',       str,    '/root/autodl-tmp/UrbanSound8K/audio/fold5/156634-5-2-5.wav', '音频路径')
add_arg('model_path',       str,    'models/PANNS_CNN10_MelSpectrogram/end2end.onnx', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

def main():
    #--------------------------加载配置文件和标签信息--------------------------
    with open(args.configs, 'r', encoding='utf-8') as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
    configs = dict_to_object(configs)
    with open(configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    class_labels = [l.replace('\n', '') for l in lines]
    
    #-----------------------------加载音频文件---------------------------------
    audio_segment = AudioSegment.from_file(args.audio_path)
    if configs.dataset_conf.use_dB_normalization:
        audio_segment.normalize(target_db=configs.dataset_conf.target_dB)

    #------------------------------提取特征------------------------------------
    audio_featurizer = AudioFeaturizer(feature_conf=configs.feature_conf, **configs.preprocess_conf)
    input_data = torch.tensor(audio_segment.samples, dtype=torch.float32).unsqueeze(0)
    input_len_ratio = torch.tensor([1], dtype=torch.float32)
    print(input_data.shape)
    audio_feature, _ = audio_featurizer(input_data, input_len_ratio)
    
    # 保存处理好的特征数据
    if args.feature_save_path != "":
        audio_feature.numpy()[:,:98,:].tofile(args.feature_save_path)
    
    #-----------------------------模型预测-------------------------------------
    session = ort.InferenceSession(args.model_path)
    print(audio_feature.shape)
    output = session.run(None, {'input': audio_feature.numpy()[:,:98,:]})
    print(output)
    index = np.argmax(output[0][0, :])
    label = class_labels[index]
    score = output[0][0, index]
    
    print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
    
def load_data(filename):
    from struct import unpack
    data_lst = []
    with open(filename, "rb") as f:
        data = bytes(f.read())
    for i in range(len(data)//4):
        v = unpack("<f", data[4*i:4*i+4])
        data_lst.append(v[0])
    return data_lst

def test_01():
    # input_data = load_data("output/test_data_1_98_64_1.bin")
    # print(input_data)
    output_data = load_data("output/Gemm_result_Y.bin")
    print(output_data)    
    
if __name__ == "__main__":
    main()

