import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
import numpy as np
import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def normalize_keypoints(keypoints, img_width, img_height):

    normalized_keypoints = keypoints.copy()
    normalized_keypoints[:, 0] /= img_width
    normalized_keypoints[:, 1] /= img_height
    
    return normalized_keypoints


def keypoint_local_euclidean_distance(keypoints1, keypoints2,  img_width, img_height,part_up,part_down,visibilities1=None, visibilities2=None, visibility_threshold=0.3):
    """
    计算两个姿态序列之间关键点的平均欧氏距离，忽略可视性低于阈值的关键点
    :param keypoints1: 第一组关键点，形状为 (num_keypoints, dimensions)
    :param keypoints2: 第二组关键点，形状应与keypoints1相同
    :param visibilities1: 第一组关键点的可视性，形状为 (num_keypoints,)，默认为None表示所有点可视
    :param visibilities2: 第二组关键点的可视性，形状应与visibilities1相同，默认为None表示所有点可视
    :param visibility_threshold: 可视性阈值，低于该阈值的关键点不参与计算
    :return: 平均欧氏距离
    """

    # 确保两组关键点的数量相匹配
    assert keypoints1.shape[0] == keypoints2.shape[0], "两组关键点的数量必须一致"
    keypoints1 = normalize_keypoints(keypoints1, img_width, img_height)
    keypoints2 = normalize_keypoints(keypoints2, img_width, img_height)
    if visibilities1 is not None and visibilities2 is not None:
        assert visibilities1.shape == (keypoints1.shape[0],) and visibilities2.shape == (keypoints2.shape[0],), \
            "可视性数组的长度必须与关键点数量一致"

        # 选择可视性大于等于阈值的关键点
        valid_indices = np.where((visibilities1 >= visibility_threshold) & (visibilities2 >= visibility_threshold))[0]
        part_indices_up = np.array(part_up)
        common_elements_up = np.intersect1d(valid_indices, part_indices_up)
        part_indices_down = np.array(part_down)
        common_elements_down = np.intersect1d(valid_indices, part_indices_down)        
        keypoints1_up = keypoints1[common_elements_up]
        keypoints2_up = keypoints2[common_elements_up]
        keypoints1_down = keypoints1[common_elements_down]
        keypoints2_down = keypoints2[common_elements_down]

    # 计算每一对对应关键点之间的欧氏距离
    distances_up = np.linalg.norm(keypoints1_up - keypoints2_up, axis=1)
    distances_down = np.linalg.norm(keypoints1_down - keypoints2_down, axis=1)

    # 如果有至少一个有效关键点，则返回平均距离；否则返回None或自定义的特殊值
    if distances_up.size > 0 and distances_down.size > 0:
        return np.mean(distances_up), np.mean(distances_down)
    elif distances_up.size == 0 and distances_down.size > 0:
        return None, np.mean(distances_down)
    elif distances_up.size > 0 and distances_down.size == 0:
        return np.mean(distances_up), None
    else:
        return None, None # 或者根据实际情况设置2个特殊的表示完全不可比的值


def keypoint_euclidean_distance(keypoints1, keypoints2,  img_width, img_height,visibilities1=None, visibilities2=None, visibility_threshold=0.3):
    """
    计算两个姿态序列之间关键点的平均欧氏距离，忽略可视性低于阈值的关键点
    :param keypoints1: 第一组关键点，形状为 (num_keypoints, dimensions)
    :param keypoints2: 第二组关键点，形状应与keypoints1相同
    :param visibilities1: 第一组关键点的可视性，形状为 (num_keypoints,)，默认为None表示所有点可视
    :param visibilities2: 第二组关键点的可视性，形状应与visibilities1相同，默认为None表示所有点可视
    :param visibility_threshold: 可视性阈值，低于该阈值的关键点不参与计算
    :return: 平均欧氏距离
    """

    # 确保两组关键点的数量相匹配
    assert keypoints1.shape[0] == keypoints2.shape[0], "两组关键点的数量必须一致"
    keypoints1 = normalize_keypoints(keypoints1, img_width, img_height)
    keypoints2 = normalize_keypoints(keypoints2, img_width, img_height)
    if visibilities1 is not None and visibilities2 is not None:
        assert visibilities1.shape == (keypoints1.shape[0],) and visibilities2.shape == (keypoints2.shape[0],), \
            "可视性数组的长度必须与关键点数量一致"

        # 选择可视性大于等于阈值的关键点
        valid_indices = np.where((visibilities1 >= visibility_threshold) & (visibilities2 >= visibility_threshold))[0]           
        keypoints1 = keypoints1[valid_indices]
        keypoints2 = keypoints2[valid_indices]

    # 计算每一对对应关键点之间的欧氏距离
    distances = np.linalg.norm(keypoints1 - keypoints2, axis=1)

    # 如果有至少一个有效关键点，则返回平均距离；否则返回None或自定义的特殊值
    if distances.size > 0:
        return np.mean(distances)
    else:
        return None  # 或者根据实际情况设置一个特殊的表示完全不可比的值



def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = pose_results[0]#merge_data_samples(pose_results)
    # print(data_samples.pred_instances["keypoints"].shape)
    # print(data_samples.pred_instances["keypoints"][0].shape)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)

def process_video(args,video_path,detector,pose_estimator,visualizer):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_folder = '/ssd/tjwu/Person_Skeleton_Estimation/full_data/output_clips'
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    pred_instances_list = []
    frame_idx = 0
    res = []
    
    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1

        if not success:
            break

        # topdown pose estimation
        pred_instances = process_one_image(args, frame, detector,
                                            pose_estimator, visualizer,
                                            0.001)
       
        if pred_instances is not None and frame_idx==1 :
            pre = pred_instances
        
        if pred_instances is not None and frame_idx>1 :                             
                temp_mean = keypoint_euclidean_distance(keypoints1=pre["keypoints"][0],keypoints2=pred_instances["keypoints"][0], img_width=width, img_height=height ,visibilities1=pre["keypoints_visible"][0],visibilities2=pred_instances["keypoints_visible"][0])
                res.append(temp_mean)

        if args.save_predictions:
            # save prediction results
            pred_instances_list.append(
                dict(
                    frame_id=frame_idx,
                    instances=split_instances(pred_instances)))
    cap.release()
    if args.save_predictions:
        pred_save_path = f'{args.output_root}/results_' f'{os.path.splitext(os.path.basename(video_path))[0]}.json'
        with open(pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {pred_save_path}')
    result_dict = {
         'res_sim': res,
         'pred_save_path': pred_save_path,
         'fps': int(fps),
         'img_width': width,
         'img_height' : height
    }
    return result_dict

def find_min_index_after_given_index(lst, start_index):
    sublist = lst[start_index:]
    if not sublist:
        return None
    min_value = min(sublist)
    min_index = start_index + sublist.index(min_value)
    return min_index

def find_max_sum_index(list1, list2):
    # 确保两个列表长度一致
    assert len(list1) == len(list2), "两个列表长度不一致！"

    # 找出两个列表各自的最大5个值及其索引
    sorted1 = sorted(enumerate(list1), key=lambda x: x[1], reverse=True)[:5]
    sorted2 = sorted(enumerate(list2), key=lambda x: x[1], reverse=True)[:5]

    combined = [(i, list1[i] + list2[i]) for i in range(len(list1))]

    

    max_sum = max(combined, key=lambda x: x[1])[1]

    max_sum_index = next(i for i, (_, v) in enumerate(combined) if v == max_sum)

    return max_sum_index

def find_end_frame_idx(args,star_idx,end_idx,json_path,img_width,img_height):
    with open(json_path, 'r') as file:
        data = json.load(file)

    #计算上半身，下半身相似度
    part_up = list(range(0, 5)) + list(range(7, 11)) + list(range(91, 133))#上半身
    part_down = list(range(11, 23))#上半身
    res_up = []
    res_down = []
    for i in range(star_idx+1,end_idx):
        keypoints1 = np.array(data["instance_info"][star_idx+1]["instances"][0]["keypoints"])
        keypoints2 = np.array(data["instance_info"][i+1]["instances"][0]["keypoints"])
        visibilities1= np.array(data["instance_info"][star_idx+1]["instances"][0]["keypoint_scores"])
        visibilities2= np.array(data["instance_info"][i+1]["instances"][0]["keypoint_scores"])
        temp_up,temp_down = keypoint_local_euclidean_distance(keypoints1=keypoints1, keypoints2=keypoints2,img_width=img_width, img_height=img_height,part_up=part_up,part_down=part_down,visibilities1=visibilities1, visibilities2=visibilities2)
        res_up.append(temp_up)
        res_down.append(temp_down)
    return find_max_sum_index(res_up,res_down)


def cut_reverse_video(args,video_file,end_frame_idx):
    video_capture = cv2.VideoCapture(video_file)

    # 获取视频的帧率和总帧数
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    # total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算要保留的帧数（丢弃最后3秒的帧）
    frames_to_keep = end_frame_idx#total_frames - fps * 3

    # 创建一个VideoWriter对象来保存截取后的视频
    name = os.path.join(args.output_root,os.path.basename(video_file))
    output_video = cv2.VideoWriter(name, 
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, 
                                (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                    int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # 读取并保存视频的帧，直到达到要保留的帧数
    frame_count = 0
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame.copy())
        output_video.write(frame)
        frame_count += 1
        if frame_count >= frames_to_keep:
            break

    # 释放资源
    video_capture.release()
    for frame in reversed(frames):
            
            output_video.write(frame)
    output_video.release()   

def process_videos_in_folder(args,detector,pose_estimator,visualizer):
    video_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.mp4')]
    for video_file in video_files:
        print(f'processing {video_file}')
        result_dict = process_video(args,video_file,detector,pose_estimator,visualizer)
        result_index = find_min_index_after_given_index(result_dict['res_sim'], result_dict['fps']*8)
        end_idx = min(result_index+result_dict['fps'],len(result_dict['res_sim']))
        for i in range(result_index,end_idx):
            if result_dict['res_sim'][i]>0.2:
                end_idx = i
                break
        end_frame_idx = find_end_frame_idx(args,result_index,end_idx,result_dict['pred_save_path'],result_dict['img_width'],result_dict['img_height'])
        end_frame_idx = result_index + end_frame_idx + 3
        cut_reverse_video(args,video_file,end_frame_idx)

def main():
    """
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--det_config', type=str, default= "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py",help='Config file for detection')
    parser.add_argument('--det_checkpoint', type=str, default= "/ssd/tjwu/Person_Skeleton_Estimation/models/mmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",help='Checkpoint file for detection')
    parser.add_argument('--pose_config', type=str, default= "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py",help='Config file for pose')
    parser.add_argument('--pose_checkpoint', type=str, default= "/ssd/tjwu/Person_Skeleton_Estimation/models/mmpose/td-hm_hrnet-w32_8xb64-210e_ubody-coco-256x192-7c227391_20230807.pth",help='Checkpoint file for pose')
    #/ssd/tjwu/Person_Skeleton_Estimation/data/lera_boroditsky_v1.mp4
    #/ssd/tjwu/Person_Skeleton_Estimation/full_data/output_clips/clip_572.0.mp4
    #/ssd/tjwu/Person_Skeleton_Estimation/full_data/output_clips/
    parser.add_argument(
        '--input', type=str, default='/ssd/tjwu/Person_Skeleton_Estimation/data/', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='vis_results_reverse/',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0.001, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    
    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    process_videos_in_folder(args,detector,pose_estimator,visualizer)
    
    


    

    




if __name__ == '__main__':
    main()
