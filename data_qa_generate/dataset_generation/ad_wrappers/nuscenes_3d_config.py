# Base Config
version = 'mini'
version = 'trainval'
length = {'trainval': 28130, 'mini': 323}

dataset_type = "NuScenes3DDataset"
data_root = "dataset_generation/raw_data/nuscenes/"
anno_root = "dataset_generation/raw_data/nuscenes_infos/"

class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

map_class_names = [
    'ped_crossing',
    'divider',
    'boundary',
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

batch_size = 1

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    map_classes=map_class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

train_pipeline = [
    # dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    # dict(type="ResizeCropFlipImage"),
    # dict(type="BBoxRotation"),
    # dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            # "img",
            "timestamp",
            "projection_mat",
            "cam_intrinsic",
            # "image_wh",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_captions_3d",
            # 'gt_map_labels', 
            # 'gt_map_pts',
            # planning
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'fut_boxes',
            
            'ego_status',
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "token", "img_filename", "aug_config"],
    ),
]


input_shape = (704, 256)
data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": False,  # temporally disable this
    "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=train_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=train_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
)