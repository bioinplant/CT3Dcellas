#%%#################################################################################################
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage.io import imread, imshow
import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import torch
from glob import glob
import tifffile as tiff
from ELD.utils import (toImg, preprocess, predict_landmarks, create_target_landmarks, 
                       create_target_images, download_images_urls, downscale_images, plot_images, 
                       mask_background, padImg, crop_non_tissue, downsize_and_save, 
                       rescale_landmarks, pad_image_and_adjust_landmarks, corr, plot_warped_images)
from ELD.model import loadFan, crop, toGrey
from ELD.warp import Homo, Rigid, TPS
from ELD.plugin import load_imgs, scale
from typing import List
#%%
if __name__ == "__main__":

    # 要处理的 ST 文件列表
    st_files = [
        # "ST-002_test.png"
        # # "ST-003.png", 
        # # "ST-004.png", 
        # # "ST-005.png", 
        # # "ST-006.png", 
        # # "ST-007.png", 
        # # "ST-009.png", 
        # # "ST-011.png", 
        # # "ST-012.png", 
        # # "ST-013.png"
        # # "16-6-cuda.png"
        # # "20-5-cuda.png"
        # "001_cp_masks.png",
        # "002_cp_masks.png",
        # "003_cp_masks.png",
        # "004_cp_masks.png",
        # "005_cp_masks.png",
        # "006_cp_masks.png",
        # "007_cp_masks.png",
        # # "008_cp_masks_spinning.png",
        # "009_cp_masks.png",
        # # "010_cp_masks_spinning.png",
        # "011_cp_masks.png",
        # "012_cp_masks.png",
        # "013_cp_masks.png"
        # "1.png"
        # "combined_visualization_3D.png"
        # "combined_visualization_ST.png"
        # "3D-1.png",
        # "3D-2.png",
        # "3D-3.png",
        # "3D-4.png",
        # "3D-5.png",
        # "3D-6.png",
        # "3D-7.png"
        # "ST-6_2_1_cp_masks_spinning.png",
        # "ST-6_2_2_cp_masks_spinning.png",
        # "ST-6_2_3_cp_masks_spinning.png",
        "ST-6HAI_2_3_cp_masks_spinning.png",
        # "ST-6_2_4_cp_masks_spinning.png"
        # "ST-24HAI_2_1_cp_masks_spinning.png",
        # "ST-24HAI_2_2_cp_masks_spinning.png",
        # "ST-24HAI_2_3_cp_masks_spinning.png",
        "ST-24HAI_2_4_cp_masks_spinning.png",
        "ST-24HAI_2_5_cp_masks_spinning.png",
        "ST-24HAI_2_6_cp_masks_spinning.png",
        "ST-24HAI_2_7_cp_masks_spinning.png",
        "ST-24HAI_2_8_cp_masks_spinning.png",
        "ST-36HAI_5_cp_masks_spinning.png",
        # "ST-36HAI_6_cp_masks_spinning.png",
        # "ST-36HAI_7_cp_masks_spinning.png",
        # "ST-36HAI_9_cp_masks_spinning.png",
        # "ST-36HAI_11_cp_masks_spinning.png",
        # "ST-36HAI_12_cp_masks_spinning.png"
        ]  # 可以扩展为更多文件

    for st_file in st_files:
        print(f"Processing {st_file}...")

        # 加载图像的函数
        def load_imgs(path):
            """
            从指定路径加载图像，支持 .tif 和常见图像格式。

            参数:
                path (str): 图像文件夹路径

            返回:
                List[List[str]]: 每组文件列表，每组包含最多99个文件名
                str: ST.png 文件的路径
            """
            # 获取路径中的所有文件
            files = glob(f"{path}*")  # 使用 glob 模块获取指定路径中的所有文件
            
            # 查找路径中的 ST.png 文件
            ST = glob(f"{path}{st_file}")  
            
            # 按照每99个文件一组进行拆分
            grouped_files = [files[i:i+99] for i in range(0, len(files), 99)]
            
            # 将 ST.png 文件添加到每组的第一个位置
            for group in grouped_files:
                group.insert(0, ST[0])
            
            return grouped_files, ST

        if __name__ == "__main__":
            # 下载数据路径
            inpath = "G:/snm/embryo3D/24H_AntherS1_2.07um_recon_Export/zoo/09-tissueSeqBinary/zoo1/spinning01/"
            # inpath = "G:/snm/embryo3D/24H_AntherS1_2.07um_recon_Export/zoo/13/zoo1/spinning03/sub/"
            # inpath = "G:/snm/embryo3D/24H_AntherS1_2.07um_recon_Export/zoo/13/zoo1/"
            # inpath = "G:/snm/embryo3D/24H_AntherS1_2.07um_recon_Export/zoo/13/zoo1/spinning03/sub/"
            # inpath = "G:/snm/embryo3D/24H_AntherS1_2.07um_recon_Export/zoo/13/zoo2/spinning03/"
            # inpath = "G:/snm/embryo3D/24H_AntherS1_2.07um_recon_Export/zoo/09-tissueSeqBinary/zoo1/spinning01/f/"
            # inpath = "G:/snm/embryo3D/24H_AntherS1_2.07um_recon_Export/zoo/ST3Dintegrated/spinning/processed/PIC/" 
            

            # 加载图像列表和文件名映射
            grouped_files, ST = load_imgs(inpath)
            
            # 创建一个空列表，用于存储所有损失数据
            all_loss_data = []
            
            for idx, group in enumerate(grouped_files):
                imgs = []
                index_filename_mapping = []
                
                for i, f in enumerate(group):
                    if f.endswith('.tif'):
                        img = tiff.imread(f)
                        # 检查图像是否加载成功
                        if img is None:
                            print(f"tif Error loading image {f}")
                            continue
                        # 如果图像是 RGBA 格式，只保留 RGB 通道
                        if img.shape[-1] == 4:
                            img = img[..., :3]
                    else:
                        img = cv2.imread(f)
                        # 检查图像是否加载成功
                        if img is None:
                            print(f"png Error loading image {f}")
                            continue
                        # 转换为 RGB 格式
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    imgs.append(img)
                    index_filename_mapping.append((i, f))
                    
                
                imageList = imgs

                # 裁剪非组织区域
                imageList = crop_non_tissue(imageList)
                # print("imageList:")
                # for i, img in enumerate(imageList):
                #     print(f"Image {i+1}: {img.shape}")
                
                #% 对图像进行缩放
                imageList = scale(imageList)

                # print("imageList scale:")
                # for i, img in enumerate(imageList):
                #     print(f"Image {i+1}: {img.shape}")

                # 降采样数据
                imageList = downscale_images(imageList)

                # print("downscale_images:")
                # for i, img in enumerate(imageList):
                #     print(f"Image {i+1}: {img.shape}")
                
                # 裁剪非组织区域
                imageList = crop_non_tissue(imageList)

                # print("crop_non_tissue:")
                # for i, img in enumerate(imageList):
                #     print(f"Image {i+1}: {img.shape}")
                
                # 调整图像大小为 128x128 并保存用于训练
                small_imgs = downsize_and_save(imageList, "./smallImg/")
                
                # 将图像预处理为 Torch 张量
                image = torch.stack([preprocess(img) for img in small_imgs])
                
                # 加载 FAN 模型
                fan = loadFan(npoints=14, n_channels=3, path_to_model="./models/unimodal/MOB_3d/model_120.fan.pth")
                
                # 预测关键点
                pts = predict_landmarks(fan, image)
                
                # 显示关键点
                np_img = toImg(image.cuda()[:,:3], pts, 128)
                
                # 将关键点缩放回原始图像
                scaled_pts = rescale_landmarks(pts, imageList)
                
                # 对所有图像进行零填充以使它们具有相同的大小
                padded_images_torch, adjusted_landmarks = pad_image_and_adjust_landmarks(imageList, scaled_pts)
                
                # 绘制原始形状的图像及其关键点
                np_img = toImg(padded_images_torch.cuda()[:,:3], adjusted_landmarks, 5 * 128)
                
                # 创建目标图像和关键点用于配准
                image = padded_images_torch
                dst_image = create_target_images(image, 0)
                dst_image_size = dst_image.size()
                # print("目标图像的大小：", dst_image_size)
                
                pts = adjusted_landmarks
                dst_pts = create_target_landmarks(pts, 0)
                
                # # 创建图像网格并显示图像########################
                # fig, axs = plt.subplots(10, 10, figsize=(100, 100))  # 调整大小
                # axs = axs.ravel()  # 展平数组
                
                # for i in range(len(np_img)):
                #     img = np_img[i]
                #     axs[i].imshow(img)
                #     axs[i].set_title(f"Image {i+1}")
                #     axs[i].axis('off')  # 隐藏轴
                
                # plt.tight_layout()
                # plt.show()
                ##############################################

                #% 使用不同方法进行图像变换和损失计算
                
                # 使用 Homography 进行图像变换
                homo_transform = Homo()
                
                # 变换图像
                mapped_imgs = homo_transform.warp_img(image.cuda(), pts, dst_pts, size=819)
                
                # 变换关键点
                mapped_pts = homo_transform.warp_pts(pts, dst_pts, pts)
                
                # 计算 Homography 损失
                homo_loss = corr(mapped_imgs, dst_image.cuda()).cpu().numpy()[1:]
                # plot_warped_images(mapped_imgs, mapped_pts, homo_loss, 5 * 128, 'HOMO')
                
                # 使用 Rigid 进行图像变换
                rigid_transform = Rigid()
                # 变换图像
                mapped_imgs = rigid_transform.warp_img(image.cuda(), pts, dst_pts, (819, 819))
                # 变换关键点
                mapped_pts = rigid_transform.warp_pts(pts, dst_pts, pts)
                rigid_loss = corr(mapped_imgs, dst_image.cuda()).cpu().numpy()[1:]
                # plot_warped_images(mapped_imgs, mapped_pts, rigid_loss, 5 * 128, 'RIG')
                
                
                # 使用 Affine transform 进行图像变换
                tps_transform = TPS()
                mapped_imgs = tps_transform.warp_img(image.cuda(), pts, dst_pts, reg=1e20, norm=True, size=863)
                mapped_pts = tps_transform.warp_pts(pts, dst_pts, pts, reg=1e20)
                affine_loss = corr(mapped_imgs, dst_image.cuda()).cpu().numpy()[1:]
                # plot_warped_images(mapped_imgs, mapped_pts, affine_loss, 5 * 128, 'AFF')
                
                # 使用 Thin-plate splines 进行图像变换
                mapped_imgs = tps_transform.warp_img(image.cuda(), pts, dst_pts, reg=0, norm=True, size=819)
                mapped_pts = tps_transform.warp_pts(pts, dst_pts, pts, reg=0)
                tps_loss = corr(mapped_imgs, dst_image.cuda()).cpu().numpy()[1:]
                # plot_warped_images(mapped_imgs, mapped_pts, tps_loss, 5 * 128, 'TPS')


                # #% 绘制每个方法的折线图并标出最大点
                # # 定义颜色列表
                # colors = ['blue', 'green', 'red', 'purple']  # 定义每个方法的颜色
                
                # # 创建图形
                # plt.figure(figsize=(20, 10))
                # image_indices = range(2, len(rigid_loss) + 2)
                
                # # 绘制 Rigid Loss
                # plt.plot(image_indices, rigid_loss, linestyle='-', color=colors[0], label='Rigid Loss')
                # max_idx_rigid = np.argmax(rigid_loss)
                # plt.scatter(image_indices[max_idx_rigid], rigid_loss[max_idx_rigid], color=colors[0], s=100, zorder=5)
                # plt.annotate(f'Max: {rigid_loss[max_idx_rigid]:.2f}', 
                #             xy=(image_indices[max_idx_rigid], rigid_loss[max_idx_rigid]), 
                #             xytext=(image_indices[max_idx_rigid]+1, rigid_loss[max_idx_rigid]+0.05),
                #             fontsize=12, color=colors[0])
                
                # # 绘制 Affine Loss
                # plt.plot(image_indices, affine_loss, linestyle='-', color=colors[1], label='Affine Loss')
                # max_idx_affine = np.argmax(affine_loss)
                # plt.scatter(image_indices[max_idx_affine], affine_loss[max_idx_affine], color=colors[1], s=100, zorder=5)
                # plt.annotate(f'Max: {affine_loss[max_idx_affine]:.2f}', 
                #             xy=(image_indices[max_idx_affine], affine_loss[max_idx_affine]), 
                #             xytext=(image_indices[max_idx_affine]+1, affine_loss[max_idx_affine]+0.05),
                #             fontsize=12, color=colors[1])
                
                # # 绘制 TPS Loss
                # plt.plot(image_indices, tps_loss, linestyle='-', color=colors[2], label='TPS Loss')
                # max_idx_tps = np.argmax(tps_loss)
                # plt.scatter(image_indices[max_idx_tps], tps_loss[max_idx_tps], color=colors[2], s=100, zorder=5)
                # plt.annotate(f'Max: {tps_loss[max_idx_tps]:.2f}', 
                #             xy=(image_indices[max_idx_tps], tps_loss[max_idx_tps]), 
                #             xytext=(image_indices[max_idx_tps]+1, tps_loss[max_idx_tps]+0.05),
                #             fontsize=12, color=colors[2])
                
                # # 绘制 Homography Loss
                # plt.plot(image_indices, homo_loss, linestyle='-', color=colors[3], label='Homography Loss')
                # max_idx_homo = np.argmax(homo_loss)
                # plt.scatter(image_indices[max_idx_homo], homo_loss[max_idx_homo], color=colors[3], s=100, zorder=5)
                # plt.annotate(f'Max: {homo_loss[max_idx_homo]:.2f}', 
                #             xy=(image_indices[max_idx_homo], homo_loss[max_idx_homo]), 
                #             xytext=(image_indices[max_idx_homo]+1, homo_loss[max_idx_homo]+0.05),
                #             fontsize=12, color=colors[3])
                
                # # 添加标签和标题
                # plt.xlabel('Image Index')
                # plt.ylabel('Loss')
                # plt.title('Comparison of Loss Values')
                # plt.legend()
                # plt.grid(True)
                # plt.show()

                
                #% 将损失数据存储到列表中
                all_loss_data.append({
                    'Filename': [os.path.basename(filename) for _, filename in index_filename_mapping[1:]],
                    'Rigid Loss': rigid_loss,
                    'Affine Loss': affine_loss,
                    'TPS Loss': tps_loss,
                    'Homography Loss': homo_loss
                })
                
            #% 循环结束后，整合数据并保存到CSV文件
            # 创建一个空的 DataFrame 用于整合数据
            all_loss_df = pd.DataFrame(columns=['Filename', 'Rigid Loss', 'Affine Loss', 'TPS Loss', 'Homography Loss'])
            
            # 将所有损失数据整合到 DataFrame 中
            for data in all_loss_data:
                all_loss_df = pd.concat([all_loss_df, pd.DataFrame(data)], ignore_index=True)

            # 根据 ST 文件名动态生成 CSV 文件名
            csv_file_name = os.path.splitext(st_file)[0] + '_loss_values.csv'
            
            # 保存整合后的数据到 CSV 文件
            all_loss_df.to_csv(os.path.join(inpath, csv_file_name), index=False)
            print(f"所有损失值已保存到 {csv_file_name} 文件中")

  # %%
