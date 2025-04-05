# #objcls4d
import argparse
import glob
from pathlib import Path
import os
import time



try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file 
from pcdet.datasets import DatasetTemplate 
from pcdet.models import build_network, load_data_to_gpu 
from pcdet.utils import common_utils 


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {'points': points, 'frame_id': index}
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='output_frames', help='directory to save output frames')
    parser.add_argument('--fps', type=int, default=10, help='frames per second for the output video')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    # Create a point cloud object
    point_cloud = open3d.geometry.PointCloud()
    vis.add_geometry(point_cloud)

    # Create a list to store bounding box objects
    box_list = []

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
        
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
        
            # Update points
            points = data_dict['points'][:, 1:].cpu().numpy()
            point_cloud.points = open3d.utility.Vector3dVector(points[:, :3])
        
            # Clear previous boxes
            for box in box_list:    
                vis.remove_geometry(box, False)
            box_list.clear()
        
            # Add new bounding boxes
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()  # Get predicted labels

            for box, label in zip(pred_boxes, pred_labels):
                obb = open3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
                    open3d.geometry.AxisAlignedBoundingBox(min_bound=box[:3] - box[3:6]/2,
                                                        max_bound=box[:3] + box[3:6]/2))
                # Use color from box_colormap based on label (handle out-of-range cases)
                color_idx = label % len(V.box_colormap)  # Ensure index is within range
                obb.color = V.box_colormap[color_idx]  # Set color based on label
                vis.add_geometry(obb)
                box_list.append(obb)

            vis.update_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()
            #time.sleep(0.1)

    logger.info('Demo done.')

if __name__ == '__main__':
    main()







