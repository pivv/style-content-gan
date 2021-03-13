import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import Tensor

import numpy as np
import pandas as pd

import yaml

import cv2

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def open_video(file_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(file_path)
    return cap


def read_image(file_path: str, unchanged: bool = False) -> np.ndarray:
    if unchanged:
        return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    else:
        return cv2.imread(file_path)


def write_image(image: np.ndarray, file_path: str) -> None:
    cv2.imwrite(file_path, image)


def read_sha_icon(icon_dir: str) -> np.ndarray:
    file_path = os.path.join(icon_dir, f'sha2.png')
    icon_image = read_image(file_path, unchanged=True)
    icon_image = cv2.resize(icon_image, SHA_SIZE)
    return icon_image


def read_item_icons(icon_dir: str) -> List[np.ndarray]:
    item_icons = [None]  # No image for no_item
    for item in ITEMS[1:]:
        file_path = os.path.join(icon_dir, f'{item}.png')
        icon_image = read_image(file_path, unchanged=True)
        icon_image = cv2.resize(icon_image, ICON_SIZE)
        item_icons.append(icon_image)
    return item_icons


def read_rank_icons(icon_dir: str) -> List[np.ndarray]:
    rank_icons = [None]  # No image for rank 0
    for rank in range(1, 1+NUM_RANK):
        file_path = os.path.join(icon_dir, f'{rank}.png')
        icon_image = read_image(file_path, unchanged=True)
        icon_image = cv2.resize(icon_image, ICON_SIZE)
        rank_icons.append(icon_image)
    return rank_icons


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def read_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def write_yaml(data: dict, file_path: str) -> None:
    with open(file_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def read_video(file_path: str, start_frame: int = 1, end_frame: int = None, dframe: int = 1,
               visualize: bool = True) -> dict:
    cap = open_video(file_path)
    if not cap.isOpened():
        raise FileNotFoundError("Error opening video stream or file")
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    # print(frame_width, frame_height, num_frame, fps, 1. / fps * 1000.)
    assert start_frame >= 1
    pos_frame = start_frame
    end_frame = num_frame if end_frame is None else min(num_frame, end_frame)
    frames = []
    iframes = []
    while cap.isOpened():
        if end_frame is not None and pos_frame >= end_frame:
            break
        ret = cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame)
        if not ret:
            break
        #pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        cur_pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        assert(pos_frame == cur_pos_frame)
        #print(pos_msec, pos_frame)
        ret, frame = cap.read()
        if not ret:
            break
        assert(frame.shape[0] == frame_height and frame.shape[1] == frame_width)
        if visualize:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                visualize = False
        frames.append(frame)
        iframes.append(pos_frame - 1)
        pos_frame += dframe
    #frames = np.stack(frames, axis=0)
    #iframes = np.stack(iframes, axis=0)
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    video_data = {'name': file_name, 'fps': fps, 'frames': frames, 'iframes': iframes,
                  'frame_names': [f'{file_name}_{iframe}' for iframe in iframes]}
    return video_data


def read_videos(file_paths: List[str] or str, dframe: int = 1) -> List[dict]:
    if isinstance(file_paths, str):  # Directory is given.
        file_paths = sorted([os.path.join(file_paths, file_name) for file_name in os.listdir(file_paths)])
    all_data = []
    for file_path in file_paths:
        data = read_video(file_path, dframe=dframe, visualize=False)
        all_data.append(data)
    return all_data


def read_data(file_paths: List[str] or str, frame_type: str = '',
              rect: Tuple[int, int, int, int] = None,
              margin: Tuple[float, float, float, float] = None,
              load_params: bool = False) -> dict:
    if isinstance(file_paths, str):  # Directory is given.
        file_paths = sorted(list(set([os.path.join(file_paths, os.path.splitext(file_name)[0]) for
                                      file_name in os.listdir(file_paths)])))
    if frame_type:
        frames_column = f'{frame_type}_frames'
        frame_names_column = f'{frame_type}_frame_names'
        margin_column = f'{frame_type}_margins'
    else:
        frames_column = 'frames'
        frame_names_column = 'frame_names'
        margin_column = ''

    images = []
    rects = []
    margins = []
    for file_path in file_paths:
        images.append(read_image(f'{file_path}.png'))
        if rect is not None:
            assert(not load_params)
            rects.append(rect)
        if margin is not None:
            assert(not load_params)
            margins.append(margin)
        if load_params:
            yaml_data = read_yaml(f'{file_path}.yaml')
            if 'rect' in yaml_data:
                rects.append(yaml_data['rect'])
            if 'margin' in yaml_data:
                margins.append(yaml_data['margin'])

    file_names = [os.path.basename(file_path) for file_path in file_paths]
    if frame_type in ['team', 'item']:
        images = [[image] for image in images]
        file_names = [[file_name] for file_name in file_names]
    data = {frames_column: images, frame_names_column: file_names}
    if rects:
        data['base_rects'] = rects
    if margins:
        data[margin_column] = margins
    return data


def write_data(data: dict, save_dir: str, frame_type: str = '', save_params: bool = False,
               save_frames: np.ndarray = None) -> None:
    if frame_type:
        frames_column = f'{frame_type}_frames'
        frame_names_column = f'{frame_type}_frame_names'
        margin_column = f'{frame_type}_margins'
    else:
        frames_column = 'frames'
        frame_names_column = 'frame_names'
        margin_column = ''
    assert(frames_column in data and frame_names_column in data)
    frames: List[np.ndarray] or List[List[np.ndarray]] = data[frames_column]
    frame_names: List[np.ndarray] or List[List[np.ndarray]] = data[frame_names_column]
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    for iframe, (frame, frame_name) in enumerate(zip(frames, frame_names)):
        if isinstance(frame, list):
            sub_frames, sub_frame_names = frame, frame_name
            sub_save_frames = save_frames[iframe] if save_frames is not None else None
        else:
            sub_frames, sub_frame_names = [frame], [frame_name]
            sub_save_frames = [save_frames[iframe]] if save_frames is not None else None
        for isub, (frame, frame_name) in enumerate(zip(sub_frames, sub_frame_names)):
            if sub_save_frames is not None and not sub_save_frames[isub]:
                continue
            write_image(frame, os.path.join(save_dir, f'{frame_name}.png'))
            if save_params:
                yaml_data = dict()
                if not frame_type and 'basic_rects' in data:
                    yaml_data['rect']: Tuple[int, int, int, int] = data['basic_rects'][iframe]
                if margin_column and margin_column in data:
                    yaml_data['margin']: Tuple[float, float, float, float] = data[margin_column][iframe]
                write_yaml(yaml_data, os.path.join(save_dir, f'{frame_name}.yaml'))


def slice_data(data: dict, indices: np.ndarray or List[int], inplace=False) -> dict:
    if inplace:
        sliced_data = data
    else:
        sliced_data = dict()
    for key in data:
        if isinstance(data[key], np.ndarray):
            sliced_data[key] = data[key][indices]
        elif isinstance(data[key], list):
            sliced_data[key] = [data[key][index] for index in indices]
        else:
            assert(key in ['name', 'fps'])
            sliced_data[key] = data[key]
    return sliced_data


def apply_margin(rect: Tuple[int, int, int, int],
                 margin: Tuple[float, float, float, float],
                 size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, w, y1, h = rect
    x2, y2 = x1+w, y1+h
    nx1 = max(int(round(x1 - w*margin[0])), 0)
    nx2 = min(int(round(x2 + w*margin[1])), size[1])
    ny1 = max(int(round(y1 - h*margin[2])), 0)
    ny2 = min(int(round(y2 + h*margin[3])), size[0])
    nrect = (nx1, nx2-nx1, ny1, ny2-ny1)
    return nrect


def compute_iou(rect1: Tuple[int, int, int, int],
                rect2: Tuple[int, int, int, int]) -> float:
    x1, w1, y1, h1 = rect1
    x2, w2, y2, h2 = rect2
    w_intersection = min(x1+w1, x2+w2) - max(x1, x2)
    h_intersection = min(y1+h1, y2+h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0:
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I
    return I / U


def crop_frames(frames: List[np.ndarray],
                rects: List[Tuple[int, int, int, int]],
                margins: List[Tuple[float, float, float, float]] = None,
                background: bool = False) -> List[np.ndarray]:
    cropped_frames: List[np.ndarray] = []
    for iframe, (frame, rect) in enumerate(zip(frames, rects)):
        if margins is not None:
            rect = apply_margin(rect, margins[iframe], frame.shape[:2])
        if not background:
            rrect = rect
        else:
            while True:
                rx = np.random.randint(low=0, high=frame.shape[1]-rect[1]+1)
                ry = np.random.randint(low=0, high=frame.shape[0]-rect[3]+1)
                rrect = (rx, rect[1], ry, rect[3])
                if compute_iou(rect, rrect) < 0.05:
                    break
        cropped_frame: np.ndarray = frame[rrect[2]:rrect[2]+rrect[3], rrect[0]:rrect[0]+rrect[1]]
        cropped_frames.append(cropped_frame)
    assert len(cropped_frames) == len(frames)
    return cropped_frames


def frames_to_gray(frames: List[np.ndarray]) -> List[np.ndarray]:
    new_frames = []
    for iframe, frame in enumerate(frames):
        new_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return new_frames


def resize_frames(frames: List[np.ndarray],
                  size: Tuple[int, int] = None,
                  margins: List[Tuple[float, float, float, float]] = None) -> np.ndarray:
    new_frames = []
    for iframe, frame in enumerate(frames):
        w = frame.shape[1]
        h = frame.shape[0]
        if margins is not None:
            margin = margins[iframe]
            w = int(round(w / (1. + margin[0] + margin[1])))
            h = int(round(h / (1. + margin[2] + margin[3])))
        x = np.random.randint(low=0, high=frame.shape[1]-w+1)
        y = np.random.randint(low=0, high=frame.shape[0]-h+1)
        new_frame = frame[y:y+h, x:x+w]
        new_frame = cv2.resize(new_frame, size, interpolation=cv2.INTER_CUBIC)
        new_frames.append(new_frame)
    new_frames = np.stack(new_frames, axis=0)
    assert len(new_frames) == len(frames)
    assert new_frames[0].shape[0] == size[1]
    assert new_frames[0].shape[1] == size[0]
    return new_frames
