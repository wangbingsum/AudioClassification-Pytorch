import torch
from macls.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_test_data():
    filename = "output/test_data_1_98_64_1.bin"
    
    data = torch.rand((1, 98, 64)).to(torch.int16).numpy()
    data.tofile(filename)
    
    logger.info("测试数据已保存：{}".format(filename))
    
if __name__ == "__main__":
    create_test_data()