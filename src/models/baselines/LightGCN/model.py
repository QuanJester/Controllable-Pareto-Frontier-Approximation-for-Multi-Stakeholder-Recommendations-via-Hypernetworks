import torch
import torch.nn as nn
# Tạo basic model để các model khác kế thừa
class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
# Mô hình nền tảng LightGCN
class LightGCN(BasicModel):
    def __init__(self, config: dict):
        super(LightGCN, self).__init__()
        self.config = config
        self.__init_weight()
    def __init_weight(self):
        # Khởi tạo các tham số quan trọng 
        self.num_users = 5
        self.num_items = 5
        self.latent_dim = self.config['latent_dim_rec']