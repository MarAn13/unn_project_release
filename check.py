import os
import cv2
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
# local libs
from model.udepth import *
from utils.data import *
from utils.utils import *
from CPD.sod_mask import get_sod_mask


def get_depth(image):
    """Generate depth map"""
    # Prepare SOD mask
    mask = np.array(get_sod_mask(image))
    # Prepare data
    image_tensor = totensor(image).unsqueeze(0)
    input_img = torch.autograd.Variable(image_tensor.to(device=device))
    # Generate depth map
    _,out=net(input_img)
    # Apply guidedfilter to depth map
    result = output_result(out, mask)
    
    return result
	
	
def comp_images(path_1, path_2):
  try:
    # Load images as grayscale
    img_1 = cv2.imread(path_1, cv2.COLOR_BGR2RGB)
    img_2 = cv2.imread(path_2, cv2.COLOR_BGR2RGB)
    if img_1.shape != img_2.shape:
      raise BaseException('ERROR! Images shapes do not match!')

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(img_1, img_2, channel_axis=2, full=True)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] image1 we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    cv2.imwrite('diff.png', diff)
    return score * 100
  except BaseException as e:
    print(e)
    return None
	
	
if '__main__' == __name__:
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('path_1', type=str, nargs='?', default='')
  parser.add_argument('path_2', type=str, nargs='?', default='')
  parser.add_argument('-c', '--compare', action='store_true')
  args = parser.parse_args()
  if not args.compare:
    # Define input space
    image_space = "RGB"

    # Create output folder if not exist 
    output_folder = './data/output/test/'
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)

    device = torch.device('cpu')

    # Load specific model
    model_path = "./saved_model/model_RGB.pth"
    import warnings
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      net = UDepth.build(n_bins=80, min_val=0.001, max_val=1, norm="linear")
      net.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded: UDepth")

    net = net.to(device=device)
    net.eval()
    img_fn = "test_img_inference.jpg"
    # Load data
    image = Image.open("test_img.jpg")
    # Generate depth map
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      result = get_depth(image)
    # Save result
    plt.imsave(os.path.join(output_folder, img_fn), result, cmap='inferno')
    args.path_1 = "test_img_result.jpg"
    args.path_2 = os.path.join(output_folder, img_fn)
    
  print('Checking images on similarity...')
  percent = comp_images(args.path_1, args.path_2)
  if percent is not None:
    print(f'Similarity: {np.round(percent, 2)}%')