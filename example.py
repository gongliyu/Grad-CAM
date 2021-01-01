from grad_cam import grad_cam
import numpy as np
import skimage.io
import skimage.transform
from matplotlib import cm

from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import resnet50

# load image
img = skimage.io.imread('example.png')

# load model
model = resnet50(pretrained=True)
model.eval()

# calculate CAM
trans = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
batch = trans(img)
cam = grad_cam(model, batch, 'layer4.2.relu')
cam = cam.detach().cpu().numpy()
cam = skimage.transform.resize(cam, img.shape[:2])


# render CAM
def overlay_cam(img, cam):
    cm_jet = cm.get_cmap('jet')
    heatmap = cm_jet(cam)[:, :, :3]
    ret = heatmap + skimage.img_as_float(img)
    ret = ret / np.max(ret)
    return skimage.img_as_ubyte(ret)


overlay = overlay_cam(img, cam)
skimage.io.imsave('example_cam.png', overlay)
