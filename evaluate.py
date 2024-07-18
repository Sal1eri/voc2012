import numpy as np
from model.FCN import FCN32s, FCN8x
from model.Unet import UNet
import torch
import torchvision.transforms as transforms
import os
from model.DeepLab import DeepLabV3

NUM_CLASSES = 21


def evaluate(val_image_path, model_e):
    from utils.DataLoade import colormap
    import matplotlib.pyplot as plt
    from PIL import Image
    model_path = './model_result/best_model_{}.mdl'.format(model_e)
    if model_e == 'FCN8x':
        net = FCN8x(NUM_CLASSES)
    elif model_e == 'UNet':
        net = UNet(3, NUM_CLASSES)
    elif model_e == 'DeepLabV3':
        net = DeepLabV3(NUM_CLASSES)
    else:
        net = FCN8x(NUM_CLASSES)
    image_name = os.path.basename(val_image_path)
    image_name = image_name.replace(".jpg", ".png")
    val_image = Image.open(val_image_path)
    tfs = transforms.Compose([
        transforms.Resize((320, 320)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 归一化
    ])
    input_image = tfs(val_image).unsqueeze(0)
    # 加载模型参数并移至GPU
    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    net.cuda()

    # 进行推理
    with torch.no_grad():
        out = net(input_image.cuda())
        pred = out.argmax(dim=1).squeeze().cpu().numpy()
        pred = np.expand_dims(pred, axis=0)
        colormap = np.array(colormap).astype('uint8')
        val_pre = colormap[pred]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(val_image)
    ax[1].imshow(val_pre.squeeze())
    ax[0].axis('off')
    ax[1].axis('off')
    save_path = './user_results/history/pic_{}_{}'.format(model_e, image_name)
    plt.savefig(save_path)
    plt.close()  # 关闭当前图形对象
    # plt.show()
    pre_img = Image.fromarray(val_pre.squeeze())
    pre_path = './user_results/pic_{}_{}'.format(model_e, image_name)
    pre_img.save(pre_path)
    return pre_path
