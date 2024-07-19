import sys
sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from igev_stereo import IGEVStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2


def load_image(imfilel,imfiler, camera_matrix_left=None, distortion_coeffs_left=None, camera_matrix_right=None, distortion_coeffs_right=None,
               R=None,T=None,Rl=None,Rr=None):
    imgl = cv2.imread(imfilel)
    imgr = cv2.imread(imfiler)
    imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)
    imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
    h, w = imgl.shape[:2]

    
    if camera_matrix_left is None:
        camera_matrix_left = np.array([[569.2192625396522, 0, 318.8201824714225],
                                       [0, 568.7043378882039, 269.0695743045546],
                                       [0, 0, 1]])
    if distortion_coeffs_left is None:
        distortion_coeffs_left = np.array([-0.05410510256997902,
                                            0.3182238890934277,
                                            0.001813864426405082,
                                            0.002745408621748306,
                                            -0.4900448373261769])
    if camera_matrix_right is None:
        camera_matrix_right = np.array([[570.0396099218942, 0, 328.8684908782695],
                                        [0, 568.9448763832647, 271.9050984622033],
                                        [0, 0, 1]])
    if distortion_coeffs_right is None:
        distortion_coeffs_right = np.array([-0.09742499150469552,
                                            0.4005082716369824,
                                            0.0005955242858359883,
                                            0.0008864414009363592,
                                            -0.4594590299410194])
    if R is None:
        R=np.array([[0.9999964449210295, -0.0007582482730996474, -0.002556404674323124],
                    [0.0007432033841410097, 0.9999824306050461, -0.005880997360474495],
                    [0.002560819015932416, 0.005879076524459265, 0.9999794391212188]])
    if T is None:
        T = np.array([-65.94946195375267,
                        -0.2174111264396236,
                        0.9964200288795323])
    if Rl is None:
        Rl = np.array([[0.999840657726031, 0.002449207225957617, -0.01768220975845256],
                        [-0.002501096322363262, 0.9999926293907279, -0.002913019293592921],
                        [0.01767494484189013, 0.002956780036272687, 0.999839413994394]])
    if Rr is None:
        Rr = np.array([[0.9998804489900642, 0.003296238184208474, -0.0151070361507702],
                            [-0.003251883258977125, 0.9999903327607969, 0.002959668937699586],
                            [0.01511664588120314, -0.002910188788337555, 0.999881501888358]])


    Rl,Rr,_,_,_,_,_= cv2.stereoRectify(camera_matrix_left,distortion_coeffs_left,camera_matrix_right,distortion_coeffs_right,(w,h),R,T, alpha=0)
    new_camera_matrix_left, _ = cv2.getOptimalNewCameraMatrix(camera_matrix_left, distortion_coeffs_left, (w,h), 0, (w,h))
    new_camera_matrix_right, _ = cv2.getOptimalNewCameraMatrix(camera_matrix_right, distortion_coeffs_right, (w,h), 0, (w,h))
    mapx_left, mapy_left = cv2.initUndistortRectifyMap(camera_matrix_left, distortion_coeffs_left, Rl,new_camera_matrix_left,(w,h), 5)
    mapx_right, mapy_right = cv2.initUndistortRectifyMap(camera_matrix_right, distortion_coeffs_right, Rr ,new_camera_matrix_right,(w,h), 5)
   
   
    img_undistorted_right = cv2.remap(imgr, mapx_right, mapy_right, cv2.INTER_LINEAR)
    img_undistorted_right = np.array(img_undistorted_right).astype(np.uint8)
    img_undistorted_right = img_undistorted_right[0:350, 0:620]

    img_undistorted_left = cv2.remap(imgl, mapx_left, mapy_left, cv2.INTER_LINEAR)
    img_undistorted_left = np.array(img_undistorted_left).astype(np.uint8)
    img_undistorted_left = img_undistorted_left[0:350, 0:620]
    # 转化为PyTorch Tensor并返回
    img_undistorted_left = torch.from_numpy(img_undistorted_left).permute(2, 0, 1).float()
    img_undistorted_right = torch.from_numpy(img_undistorted_right).permute(2, 0, 1).float()

  
    return img_undistorted_left[None].to(DEVICE),img_undistorted_right[None].to(DEVICE)




def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        args.left_imgs = "D:/shuangmu/IGEV-main/IGEV-Stereo/demo-imgs/Motorcycle/im0.jpg"
        args.right_imgs = "D:/shuangmu/IGEV-main/IGEV-Stereo/demo-imgs/Motorcycle/im1.jpg"
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1,image2 = load_image(imfile1,imfile2)


            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            file_stem = imfile1.split('/')[-2]
            filename = os.path.join(output_directory, f"{file_stem}.png")
            plt.imsave(output_directory / f"{file_stem}.png", disp, cmap='jet')
            cv2.imwrite(filename, (disp.squeeze() * 256).astype(np.uint16))
       
            # disp = np.round(disp * 256).astype(np.uint16)
            # cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(disp.squeeze(), alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


            
    disp = cv2.imread('demo-output\Motorcycle.png', cv2.IMREAD_UNCHANGED)
    # 进行伪彩色映射
    disp_jet = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=0.1), cv2.COLORMAP_JET)
    def onclick(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if x >= 0 and x < disp.shape[1] and y >= 0 and y < disp.shape[0]:
                focal_length = 568.8246071357344
                baseline = 1/0.01516131321421901
                x_offset = abs(318.8201824714225 - 328.8684908782695)
                depth = (focal_length * baseline) / (disp[int(y), int(x)] + x_offset)
                text = f"{depth:.2f} meters"
                cv2.putText(disp, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imshow('Depth Image', disp)

    cv2.namedWindow('Depth Image')
    cv2.imshow('Depth Image', disp_jet)
    cv2.setMouseCallback('Depth Image', onclick)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='sceneflow\sceneflow.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.png")

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/Middlebury/trainingH/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/Middlebury/trainingH/*/im1.png")
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/ETH3D/two_view_training/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/ETH3D/two_view_training/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)
