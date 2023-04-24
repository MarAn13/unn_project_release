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
    output_folder = 'diff'
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
    
    total_score = 0
    for index, [img_path_1, img_path_2] in enumerate(tqdm(zip(sorted(os.listdir(path_1)), sorted(os.listdir(path_2))), desc='Comparing images')):
      # Load images as grayscale
      img_1 = cv2.imread(path_1 + '/' + img_path_1, cv2.COLOR_BGR2RGB)
      img_2 = cv2.imread(path_2 + '/' + img_path_2, cv2.COLOR_BGR2RGB)
      if img_1.shape != img_2.shape:
        raise BaseException(f'ERROR! Images shapes do not match!{path_1 + "/" + img_path_1} \n {path_2 + "/" + img_path_2}')

      # Compute SSIM between the two images
      (score, diff) = structural_similarity(img_1, img_2, channel_axis=2, full=True)

      # The diff image contains the actual image differences between the two images
      # and is represented as a floating point data type in the range [0,1] 
      # so we must convert the array to 8-bit unsigned integers in the range
      # [0,255] image1 we can use it with OpenCV
      diff = (diff * 255).astype("uint8")
      cv2.imwrite(output_folder + '/' + f'diff_{index}.png', diff)
      if args.verbose:
        print('score:', score, path_1 + "/" + img_path_1, ';', path_2 + "/" + img_path_2)
      total_score += score
    return (total_score / (index + 1)) * 100
  except BaseException as e:
    print(e)
    return None
	
	
if '__main__' == __name__:
  import argparse
  from tqdm import tqdm
  parser = argparse.ArgumentParser()
  parser.add_argument('path_1', type=str, nargs='?', default='')
  parser.add_argument('path_2', type=str, nargs='?', default='')
  parser.add_argument('-cf', '--compare_folder', action='store_true')
  parser.add_argument('-vb', '--verbose', action='store_true')
  args = parser.parse_args()
  if not args.compare_folder:
    # Define input space
    image_space = "RGB"
    
    input_folder = './images_test'

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
    
    for index, img_path in enumerate(tqdm(sorted(os.listdir(input_folder)), desc='Image processing')):
      img_fn = f"test_img_inference_{index}.jpg"
      # Load data
      image = Image.open(input_folder + '/' + img_path)
      # Generate depth map
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = get_depth(image)
      # Save result
      plt.imsave(os.path.join(output_folder, img_fn), result, cmap='inferno')
      args.path_1 = 'images_local_result'
      args.path_2 = output_folder
  
  print('Checking images on similarity...')
  avg_percent = comp_images(args.path_1, args.path_2)
  if avg_percent is not None:
    print('\n', f'Average similarity: {np.round(avg_percent, 2)}%')