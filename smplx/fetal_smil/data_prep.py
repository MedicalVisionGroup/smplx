#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/20/2024
#
# Distributed under terms of the MIT license.

""" """

import json
import os
import os.path as osp

import numpy as np
from einops import rearrange
from skimage import measure
from tqdm import tqdm


def _get_subj_name_list(group: str):
    if group.startswith("MAP"):
        # this is a single subject case
        from fetal_body_segm_kpt.datasets.datasets import (
            SkeletonMetaData,
            SkeletonSingleton,
        )

        # check if subjex in the dataset
        meta_data = SkeletonMetaData()
        _, date_ID_list, _ = SkeletonSingleton._get_data_path_list(return_flat=True)
        subj_name_list = [meta_data.date2subj_name[d] for d in date_ID_list]
        subj_name_list = sorted(list(set(subj_name_list)))
        assert group in subj_name_list, f"{group} not in {subj_name_list}"

        subj_name_list = [group]

    elif "all_singleton" == group:
        from fetal_body_segm_kpt.datasets.datasets import (
            SkeletonMetaData,
            SkeletonSingleton,
        )

        meta_data = SkeletonMetaData()
        _, date_ID_list, _ = SkeletonSingleton._get_data_path_list(return_flat=True)
        subj_name_list = [meta_data.date2subj_name[d] for d in date_ID_list]
        subj_name_list = sorted(list(set(subj_name_list)))

    elif "nice_5_subjects" == group:
        subj_name_list = [
            "MAP-C507",
            "MAP-B304",
            "MAP-C407",
            "MAP-C502",
            "MAP-C533-L",
        ]

    else:
        raise ValueError(f"Unknown group: {group}")

    return subj_name_list


def main_init_data_preparation(
    exp_data_split_dir, pred_seg_map_dir, subj_name, min_frame, max_frame, resolution
):
    import nibabel as nib

    from fetal_body_segm_kpt.datasets.datasets import SkeletonSingleton

    dataset = SkeletonSingleton(subj_name2split_tag={subj_name: "test"}, transform=None)

    # keypoint (physical and pixel)
    kpt_pixel_seq = np.array([d["kpt"] for d in dataset])
    kpt_pixel_seq = rearrange(kpt_pixel_seq, "t c n -> t n c")
    neck = (kpt_pixel_seq[:, 12] + kpt_pixel_seq[:, 11]) / 2
    kpt_pixel_seq = np.concatenate([kpt_pixel_seq, neck[:, None]], axis=1)
    kpt_seq = kpt_pixel_seq * resolution

    # save at most max_frame frames
    kpt_seq = kpt_seq[min_frame:max_frame]
    kpt_pixel_seq = kpt_pixel_seq[min_frame:max_frame]

    # segmentation map
    subj_pred_seg_map_dir = osp.join(pred_seg_map_dir, subj_name)
    assert osp.exists(subj_pred_seg_map_dir), f"{subj_pred_seg_map_dir} does not exist"
    segm_map_file_list = sorted(os.listdir(subj_pred_seg_map_dir))
    segm_seq = []
    for segm_map_file in segm_map_file_list:
        segm_map_path = osp.join(subj_pred_seg_map_dir, segm_map_file)
        segm_map = nib.load(segm_map_path).get_fdata().squeeze()
        segm_seq.append(segm_map)
    segm_seq = np.array(segm_seq)
    segm_seq = segm_seq[min_frame:max_frame]

    # distance transform
    # f_dist_trans = ndimage.distance_transform_edt
    # segm_dist_map_seq = [
    #     np.abs((f_dist_trans(seg) - f_dist_trans(1 - seg))) for seg in tqdm(segm_seq)
    # ]

    # marching cube to get body segmentation surface vertices and faces
    segm_verts_seq = []
    segm_faces_seq = []
    for segm in tqdm(segm_seq):
        segm_verts_pixel, segm_faces, _, _ = measure.marching_cubes(
            segm, level=0.5, spacing=(1, 1, 1)
        )
        segm_verts_pixel = segm_verts_pixel[..., [1, 0, 2]]
        segm_verts = segm_verts_pixel * resolution
        segm_verts_seq.append(segm_verts)
        segm_faces_seq.append(segm_faces)

    segm_verts_seq = np.array(segm_verts_seq, dtype=object)
    segm_faces_seq = np.array(segm_faces_seq, dtype=object)

    # save data
    save_dir = osp.join(exp_data_split_dir, subj_name)
    os.makedirs(save_dir, exist_ok=True)
    np.save(osp.join(save_dir, "kpt_seq"), kpt_seq)
    np.save(osp.join(save_dir, "kpt_pixel_seq"), kpt_pixel_seq)
    # np.save(osp.join(save_dir, "segm_seq"), segm_seq)
    # np.save(osp.join(save_dir, "dist_map_seq"), segm_dist_map_seq)
    np.save(osp.join(save_dir, "segm_vertex_seq"), segm_verts_seq)
    np.save(osp.join(save_dir, "segm_faces_seq"), segm_faces_seq)


def main_init_data_check(
    exp_data_split_dir, pred_seg_map_dir, subj_name, min_frame, max_frame, resolution
):
    """Modified from main_init_data_preparation to run quality test of
    the dataset. One report per subject is saved to split dir.
    Kpt and segm are not saved.

    Input args are the same as main_init_data_preparation to make consistent.
    """
    test_report_dir = osp.join(exp_data_split_dir, "init_data_check")
    os.makedirs(test_report_dir, exist_ok=True)
    subj_report_path = osp.join(test_report_dir, f"{subj_name}.txt")

    import nibabel as nib

    from fetal_body_segm_kpt.datasets.datasets import SkeletonSingleton

    dataset = SkeletonSingleton(subj_name2split_tag={subj_name: "test"}, transform=None)

    # keypoint (physical and pixel)
    kpt_pixel_seq = np.array([d["kpt"] for d in dataset])
    kpt_pixel_seq = rearrange(kpt_pixel_seq, "t c n -> t n c")
    neck = (kpt_pixel_seq[:, 12] + kpt_pixel_seq[:, 11]) / 2
    kpt_pixel_seq = np.concatenate([kpt_pixel_seq, neck[:, None]], axis=1)
    kpt_seq = kpt_pixel_seq * resolution

    # save at most max_frame frames
    kpt_seq = kpt_seq[min_frame:max_frame]
    kpt_pixel_seq = kpt_pixel_seq[min_frame:max_frame]

    # segmentation map
    subj_pred_seg_map_dir = osp.join(pred_seg_map_dir, subj_name)
    assert osp.exists(subj_pred_seg_map_dir), f"{subj_pred_seg_map_dir} does not exist"
    segm_map_file_list = sorted(os.listdir(subj_pred_seg_map_dir))
    segm_seq = []
    for segm_map_file in segm_map_file_list:
        segm_map_path = osp.join(subj_pred_seg_map_dir, segm_map_file)
        segm_map = nib.load(segm_map_path).get_fdata().squeeze()
        segm_seq.append(segm_map)
    segm_seq = np.array(segm_seq)
    segm_seq = segm_seq[min_frame:max_frame]

    #########################################
    #  test1: kpt must be inside body segm  #
    #########################################

    segm_label_at_kpt_pixel_seq = []
    for kpt_pixel, segm in zip(kpt_pixel_seq, segm_seq):
        # kpt_pixel: (n, 3) (ijk indexing)
        # segm: (h, w, d) (xyz indexing)
        kpt_pixel = kpt_pixel.astype(int)
        kpt_pixel_xyz = kpt_pixel[..., [1, 0, 2]]
        segm_label_at_kpt_pixel = segm[
            kpt_pixel_xyz[:, 0], kpt_pixel_xyz[:, 1], kpt_pixel_xyz[:, 2]
        ]
        segm_label_at_kpt_pixel_seq.append(segm_label_at_kpt_pixel)
    segm_label_at_kpt_pixel_seq = np.array(segm_label_at_kpt_pixel_seq)  # (t, n)

    # if there are any kpt outside the body segm, report it
    if np.any(segm_label_at_kpt_pixel_seq == 0):
        with open(subj_report_path, "w") as f:
            f.write(
                "Test1: kpt must be inside body segm\n"
                "------------------------------------\n"
                "Failed frame-kpt indices:\n"
            )
            failed_frame_kpt_indices = np.where(segm_label_at_kpt_pixel_seq == 0)
            for frame_idx, kpt_idx in zip(*failed_frame_kpt_indices):
                f.write(f"frame: {frame_idx}, kpt: {kpt_idx}\n")

            # sort frame by n_kpt outside body segm
            idx_f2idx_kpt_list: dict[int, list[int]] = {}
            for frame_idx, kpt_idx in zip(*failed_frame_kpt_indices):
                if frame_idx not in idx_f2idx_kpt_list:
                    idx_f2idx_kpt_list[frame_idx] = []
                idx_f2idx_kpt_list[frame_idx].append(kpt_idx)
            idx_f2n_kpt = {k: len(v) for k, v in idx_f2idx_kpt_list.items()}

            # sort from most to least
            idx_f2n_kpt = dict(
                sorted(idx_f2n_kpt.items(), key=lambda item: item[1], reverse=True)
            )
            f.write("\nFrames sorted by n_kpt outside body segm:\n")
            idx_f_list = list(idx_f2n_kpt.keys())
            f.write(",".join([str(i) for i in idx_f_list]))
            f.write("\n")


def main_init_data_preparation_chiari_sebo_ismrm_individual(
    exp_data_split_dir, pred_seg_map_dir
):
    import nibabel as nib

    from fetal_body_segm_kpt.datasets.datasets import ChiariIndividuals
    from smplx.vertex_keypoint_regressor import _kpt_name_list

    # figure out the left right mapping
    new_idx_list = []
    for kn in _kpt_name_list:
        if kn.endswith("_r"):
            new_kn = kn[:-2] + "_l"
            new_idx = _kpt_name_list.index(new_kn)
        elif kn.endswith("_l"):
            new_kn = kn[:-2] + "_r"
            new_idx = _kpt_name_list.index(new_kn)
        else:
            new_idx = _kpt_name_list.index(kn)
        new_idx_list.append(new_idx)

    dataset = ChiariIndividuals(transform=None)

    up_scale_factor = 4 / 3  # we interprete 3mm as if it is 4mm

    need_flip_pid_list = [2, 3, 4, 5, 6, 10, 15, 22, 23, 24, 25, 27, 30]

    # save kpt and segm related data
    split_meta_data_dict = {}
    for i in range(len(dataset)):
        data = dataset[i]

        PID = data["PID"]
        split_meta_data_dict[f"{PID:04d}"] = [0, 2]

        res = data["resolution"]
        assert res[0] == res[1], f"res[0] != res[1]: {res}"
        res = np.array(res) * up_scale_factor
        res = res / 1000  # mm to m

        # in individual dataset, each subject has one frame,
        # we will duplicate to make it 2 frames to be compatible.
        kpt = data["kpt"]
        kpt_pixel_seq = np.stack([kpt, kpt], axis=0)
        kpt_pixel_seq = rearrange(kpt_pixel_seq, "t c n -> t n c")

        # NOTE(YL 11/02):: in sebo's dataset, neck is already added.
        # neck = (kpt_pixel_seq[:, 12] + kpt_pixel_seq[:, 11]) / 2
        # kpt_pixel_seq = np.concatenate([kpt_pixel_seq, neck[:, None]], axis=1)

        # NOTE(YL 11/05):: in sebo's dataset, we need to flip xy.
        # and also flip the left and right label.
        kpt_pixel_seq = kpt_pixel_seq[..., [1, 0, 2]]
        # kpt_pixel_seq = kpt_pixel_seq[:, new_idx_list]

        if int(PID) in need_flip_pid_list:
            kpt_pixel_seq = kpt_pixel_seq[:, new_idx_list]

        kpt_seq = kpt_pixel_seq * res

        # TODO(YL 11/05):: remove this image after debug
        # image
        # img = data["img"]
        # img_seq = np.stack([img, img], axis=0)

        # segmentation map
        segm_map_path = osp.join(pred_seg_map_dir, f"{PID:04d}_pred_seg_map.nii.gz")
        segm_map = nib.load(segm_map_path).get_fdata().squeeze()
        segm_map = np.array(segm_map, dtype=np.float32)

        # we pretend we have 2 frames
        segm_seq = [segm_map, segm_map]

        # marching cube to get body segmentation surface vertices and faces
        segm_verts_seq = []
        segm_faces_seq = []
        for segm in segm_seq:
            segm_verts_pixel, segm_faces, _, _ = measure.marching_cubes(
                segm, level=0.5, spacing=(1, 1, 1)
            )
            segm_verts_pixel = segm_verts_pixel[..., [1, 0, 2]]
            segm_verts = segm_verts_pixel * res

            segm_verts_seq.append(segm_verts)
            segm_faces_seq.append(segm_faces)

        segm_verts_seq = np.array(segm_verts_seq)
        segm_faces_seq = np.array(segm_faces_seq)

        # save data
        save_dir = osp.join(exp_data_split_dir, f"{PID:04d}")
        os.makedirs(save_dir, exist_ok=True)
        np.save(osp.join(save_dir, "kpt_seq"), kpt_seq)
        np.save(osp.join(save_dir, "kpt_pixel_seq"), kpt_pixel_seq)
        np.save(osp.join(save_dir, "segm_vertex_seq"), segm_verts_seq)
        np.save(osp.join(save_dir, "segm_faces_seq"), segm_faces_seq)

        # TODO(YL 11/05):: remove after debug
        # np.save(osp.join(save_dir, "img_seq"), img_seq)
        # np.save(osp.join(save_dir, "segm_seq"), segm_seq)

    # split meta data json
    with open(
        osp.join(exp_data_split_dir, "chiari_sebo_ismrm_individual.json"), "w"
    ) as f:
        json.dump(split_meta_data_dict, f)
