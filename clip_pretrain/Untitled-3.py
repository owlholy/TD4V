import torch

# 加载第一个.pth文件的参数
state_dict1 = torch.jit.load('/mnt/DataDisk02/lyw/Video/models/ViT-L-14.pt', map_location=torch.device('cpu'))
state_dict1 = state_dict1.state_dict()
torch.save(state_dict1, 'state_dict1.pth')
# 加载第二个.pth文件的参数
#state_dict2 = torch.load('/home/xgy/AdaptFormer-bias/youjiandu/videomae_pretrain_vit_b_1600_new.pth', map_location=torch.device('cpu'))

# 保存第一个.pth文件的参数和值到txt文件
with open('image-huge-0.txt', 'w') as file:
    for key, value in state_dict1.items():
        file.write(f"键: {key}\n")
        file.write(f"值: {value}\n\n")

# 保存第二个.pth文件的参数和值到txt文件
# with open('model2_parameters.txt', 'w') as file:
#     for key, value in state_dict2.items():
#         file.write(f"键: {key}\n")
#         file.write(f"值: {value}\n\n")

print("参数和值已保存到txt文件中。")