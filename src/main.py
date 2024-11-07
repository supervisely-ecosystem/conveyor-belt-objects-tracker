import os
from datetime import datetime
from typing import *

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.cluster import KMeans

import src.globals as g
import supervisely as sly
import supervisely.app.development as sly_app_development
from lightglue import ALIKED, DISK, SIFT, DoGHardNet, LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
from supervisely.api.annotation_api import AnnotationInfo
from supervisely.api.video.video_api import VideoInfo
from supervisely.app.widgets import Button, CheckboxField, Container, Field, Text

button_track = Button("Track objects")

checkbox = CheckboxField(
    title="Enable Autotracking",
    description="Create bbox to interpolate figure instantly. Uncheck to unlock manual button.",
    checked=True,
)

status_image_not_initialized = Text("Please refresh page", "error")
status_image_not_initialized.hide()
layout = Container(
    widgets=[
        checkbox,
        button_track,
        status_image_not_initialized,
    ]
)
button_track.disable()
app = sly.Application(layout=layout)

# Enabling advanced debug mode.
if sly.is_development():
    load_dotenv("local.env")
    team_id = sly.env.team_id()
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    sly_app_development.supervisely_vpn_network(action="up")
    sly_app_development.create_debug_task(team_id, update_status=True)


image_infos_dct = {}
matches = {}
timestamp = None

curr_event_tmp = None  # temporary soultion


@sly.timeit
def get_image_infos(
    api: sly.Api, dataset_id: int, image_id: int, margins_shift: int = 0
) -> Tuple[List[sly.ImageInfo], int]:
    if dataset_id not in image_infos_dct:
        image_infos = api.image.get_list(dataset_id)
        image_infos = sorted(image_infos, key=lambda x: x.name)
        image_infos_dct[dataset_id] = image_infos
    else:
        image_infos = image_infos_dct[dataset_id]

    image_ids = [x.id for x in image_infos]

    event_img_idx = image_ids.index(image_id)
    start_idx = max(0, event_img_idx - 2 + margins_shift)
    end_idx = min(event_img_idx + 3 - margins_shift, len(image_infos))

    image_ids = image_ids[start_idx:end_idx]
    event_img_idx = image_ids.index(image_id)

    return image_infos[start_idx:end_idx], event_img_idx


@sly.timeit
def get_images_and_anns(
    api: sly.Api,
    dataset_id: int,
    image_id: int,
) -> Tuple[List[sly.ImageInfo], List[AnnotationInfo], Tuple[int]]:

    image_infos, event_img_idx = get_image_infos(api, dataset_id, image_id)
    image_ids = [x.id for x in image_infos]

    ann_infos = api.annotation.download_batch(dataset_id, image_ids)

    sly.logger.info(
        "Center Image Labels IDs: %s",
        [x["id"] for x in ann_infos[event_img_idx].annotation["objects"]],
    )

    return image_infos, ann_infos, event_img_idx


@sly.timeit
def get_points_image(
    api: sly.Api, event: sly.Event.ManualSelected.ImageChanged, image_id: int
) -> Optional[Dict[int, dict]]:

    m = matches.get(image_id)
    if m is not None and not all(m.get(k, {}).get("diff") is None for k in [-1, 1]):
        return matches[image_id]

    images, _ = get_image_infos(api, event.dataset_id, event.image_id)
    image_ids = [x.id for x in images]
    for img_id in image_ids:
        if matches.get(img_id) is None:
            matches[img_id] = {-1: {"diff": None, "points": []}, 1: {"diff": None, "points": []}}

    paths = [g.SLY_APP_DATA + "/" + str(id) + ".png" for id in image_ids]

    paths_to_download = [path for path in paths if not sly.fs.file_exists(path)]
    image_ids_to_download = [int(sly.fs.get_file_name(path)) for path in paths_to_download]
    api.image.download_paths(event.dataset_id, image_ids_to_download, paths_to_download)

    def _get_diff(m_kpts0, m_kpts1):
        diffs = m_kpts1 - m_kpts0
        filtered = diffs[diffs[:, 1] > 30]  # filter by y-axis static places
        return filtered.mean(dim=0)

    def _find_static_borders(m_kpts0, m_kpts1, image0_width, image1_width):

        data = torch.cat((m_kpts0, m_kpts1), dim=0).T[0].reshape(-1, 1)
        data = torch.cat((data, torch.tensor([[0], [image0_width], [image1_width]])), dim=0)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_

        # Print the clusters
        cluster_1 = data[labels == 0]
        cluster_2 = data[labels == 1]

        return int(cluster_1.max()), int(cluster_2.min())

    pairs = [(images[i], images[i + 1]) for i in range(len(images) - 1)]
    for idx, pair in enumerate(pairs):
        image_i, image_j = pair
        if matches[image_i.id][1]["diff"] is None:
            image_i_load = load_image(paths[idx])
            image_j_load = load_image(paths[idx + 1])
            m_kpts0, m_kpts1, kpts0, kpts1, matches01 = lightglue(
                image_i_load, image_j_load, max_num_keypoints=512
            )
            diff = _get_diff(m_kpts0, m_kpts1)
            if any(torch.isnan(diff)):
                sly.logger.debug(
                    "lightglue has failed with static points. Cropping images to make image matching better."
                )
                x0, x1 = _find_static_borders(m_kpts0, m_kpts1, image_i.width, image_j.width)
                image_i_load = load_image(paths[idx], crop_vertically=(x0, x1))
                image_j_load = load_image(paths[idx + 1], crop_vertically=(x0, x1))
                try:
                    m_kpts0, m_kpts1, kpts0, kpts1, matches01 = lightglue(
                        image_i_load, image_j_load, max_num_keypoints=512
                    )
                    diff = _get_diff(m_kpts0, m_kpts1)
                except ValueError:
                    pass

            if any(torch.isnan(diff)):
                sly.logger.warning(
                    "Image matching has failed. Possibly, matching images have nothing in common?"
                )
            matches[image_i.id][1]["diff"] = diff.type(torch.int).numpy().tolist()
            matches[image_i.id][1]["points"] = (m_kpts0, m_kpts1)

    for pair in pairs[::-1]:
        image_j, image_i = pair  # reverse order
        if matches[image_i.id][-1]["diff"] is None:
            diff = matches[image_j.id][1]["diff"]
            points = matches[image_j.id][1]["points"]
            matches[image_i.id][-1]["diff"] = diff
            matches[image_i.id][-1]["points"] = points

    return matches.get(image_id)


@sly.timeit
def get_points_video(
    api: sly.Api, event: sly.Event.ManualSelected.VideoChanged, frame_id: str
) -> Optional[Dict[int, dict]]:

    m = matches.get(frame_id)
    if m is not None and not all(m.get(k, {}).get("diff") is None for k in [-1, 1]):
        return matches[frame_id]

    video_info = api.video.get_info_by_id(event.video_id)

    end_frame = min(event.frame + 2, video_info.frames_count)
    start_frame = max(event.frame - 2, 0)

    frame_indexes = [x for x in range(end_frame + 1)][start_frame:]
    frame_ids = [f"{event.video_id}_{x}" for x in frame_indexes]

    for _frame_id in frame_ids:
        if matches.get(_frame_id) is None:
            matches[_frame_id] = {-1: {"diff": None, "points": []}, 1: {"diff": None, "points": []}}

    paths = [g.SLY_APP_DATA + "/" + id + ".png" for id in frame_ids]

    paths_to_download = [path for path in paths if not sly.fs.file_exists(path)]
    frame_indexes_to_download = [
        int(sly.fs.get_file_name(path).split("_")[1]) for path in paths_to_download
    ]
    api.video.frame.download_paths(video_info.id, frame_indexes_to_download, paths_to_download)

    def _get_diff(m_kpts0, m_kpts1):
        diffs = m_kpts1 - m_kpts0
        filtered = diffs[diffs[:, 1] > 30]  # filter by y-axis static places
        return filtered.mean(dim=0)

    def _find_static_borders(m_kpts0, m_kpts1, image0_width, image1_width):

        data = torch.cat((m_kpts0, m_kpts1), dim=0).T[0].reshape(-1, 1)
        data = torch.cat((data, torch.tensor([[0], [image0_width], [image1_width]])), dim=0)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_

        # Print the clusters
        cluster_1 = data[labels == 0]
        cluster_2 = data[labels == 1]

        return int(cluster_1.max()), int(cluster_2.min())

    pairs = [(load_image(paths[i]), load_image(paths[i + 1])) for i in range(len(paths) - 1)]
    pairs_ids = [(frame_ids[i], frame_ids[i + 1]) for i in range(len(frame_ids) - 1)]
    for pair_ids, pair in zip(pairs_ids, pairs):
        image_i, image_j = pair
        frame_id_i, frame_id_j = pair_ids
        if matches[frame_id_i][1]["diff"] is None:
            m_kpts0, m_kpts1, kpts0, kpts1, matches01 = lightglue(
                image_i, image_j, max_num_keypoints=512
            )
            diff = _get_diff(m_kpts0, m_kpts1)
            if any(torch.isnan(diff)):
                sly.logger.debug(
                    "lightglue has failed with static points. Cropping images to make frame matching better."
                )
                x0, x1 = _find_static_borders(m_kpts0, m_kpts1, image_i.shape[1], image_j.shape[1])
                try:
                    m_kpts0, m_kpts1, kpts0, kpts1, matches01 = lightglue(
                        image_i[:, x0:x1, :], image_j[:, x0:x1, :], max_num_keypoints=512
                    )
                    diff = _get_diff(m_kpts0, m_kpts1)
                except ValueError:
                    pass

            if any(torch.isnan(diff)):
                sly.logger.warning(
                    "Frame matching (%s and %s) has failed. Possibly, the frame images have nothing in common?",
                    frame_id_i,
                    frame_id_j,
                )
            matches[frame_id_i][1]["diff"] = diff.type(torch.int).numpy().tolist()
            matches[frame_id_i][1]["points"] = (m_kpts0, m_kpts1)

    for pair_ids in pairs_ids[::-1]:
        frame_id_j, frame_id_i = pair_ids  # reverse order
        if matches[frame_id_i][-1]["diff"] is None:
            diff = matches[frame_id_j][1]["diff"]
            points = matches[frame_id_j][1]["points"]
            matches[frame_id_i][-1]["diff"] = diff
            matches[frame_id_i][-1]["points"] = points

    return matches[frame_id]


@app.event(sly.Event.Entity.FrameChanged)
def video_frame_changed(api: sly.Api, event: sly.Event.Entity.FrameChanged):
    g.api = api
    if event is None:
        return
    # g.curr_event[event.user_id] = event

    global curr_event_tmp
    curr_event_tmp = event

    status_image_not_initialized.hide()
    frame_id = f"{event.video_id}_{event.frame}"
    sly.logger.debug("Event frame ID=%s", frame_id)
    get_points_video(api, event, frame_id)

    if not button_track.is_disabled():
        return
    # if g.previous_frame is not None:
    #     g.event_anns[event.user_id][g.previous_frame] = g.DONE  # TODO previous frame
    # g.previous_frame = frame_id


@app.event(sly.Event.ManualSelected.VideoChanged)
def video_frame_changed(api: sly.Api, event: sly.Event.ManualSelected.VideoChanged):
    g.api = api
    if event is None:
        return
    # g.curr_event[event.user_id] = event

    global curr_event_tmp
    curr_event_tmp = event

    status_image_not_initialized.hide()
    frame_id = f"{event.video_id}_{event.frame}"
    sly.logger.debug("Event frame ID=%s", frame_id)
    get_points_video(api, event, frame_id)

    if not button_track.is_disabled():
        return
    # if g.previous_frame is not None:
    #     g.event_anns[event.user_id][g.previous_frame] = g.DONE  # TODO previous frame
    # g.previous_frame = frame_id


@app.event(sly.Event.ManualSelected.ImageChanged)
def image_changed(api: sly.Api, event: sly.Event.ManualSelected.ImageChanged):
    g.api = api
    if event is None:
        return
    # g.curr_event[event.user_id] = event

    global curr_event_tmp
    curr_event_tmp = event

    status_image_not_initialized.hide()
    sly.logger.debug("Event image ID={}".format(event.image_id))
    get_points_image(api, event, event.image_id)

    if not button_track.is_disabled():
        return
    if g.previous_image is not None:
        g.event_anns[event.user_id][g.previous_image] = g.DONE
    g.previous_image = event.image_id


def process_video_rectangle_changed(api: sly.Api, event: sly.Event.Tools.Rectangle.FigureChanged):
    project_meta = g.get_project_meta(api, event.project_id)
    figure = api.image.figure.get_info_by_id(event.figure_state["id"])
    obj_class = project_meta.get_obj_class_by_id(event.figure_state["classId"])

    ann_json = api.video.annotation.download(event.video_id)
    key_id_map = sly.KeyIdMap()
    video_ann = sly.VideoAnnotation.from_json(ann_json, project_meta, key_id_map)
    video_info = api.video.get_info_by_id(event.video_id)

    end_frame = min(event.frame + 2, video_info.frames_count)
    start_frame = max(event.frame - 2, 0)

    frame_indexes = [x for x in range(end_frame + 1)][start_frame:]
    event_frame_idx = frame_indexes.index(event.frame)
    frame_indexes[:event_frame_idx] = frame_indexes[:event_frame_idx][::-1]

    video_object = sly.VideoObject(
        obj_class,
        key=key_id_map.get_object_key(event.object_id),
        class_id=event.tool_class_id,
    )
    video_figure = sly.VideoFigure(
        video_object,
        figure.bbox,
        event_frame_idx,
        # key=key_id_map.get_object_key(event.object_id),
        # class_id=event.tool_class_id,
        track_id=event.object_id,
    )

    new_figures = []
    drow, dcol = 0, 0
    for idx, frame_index in enumerate(frame_indexes):

        if idx == event_frame_idx:
            drow, dcol = 0, 0
            continue

        direction = -1 if idx < event_frame_idx else 1
        frame_id = f"{event.video_id}_{frame_index}"

        _drow, _dcol = get_increment_by_diff(
            api, event, frame_id, direction, reverse_direction=True, modality="video"
        )

        drow += _drow * direction
        dcol += _dcol * direction

        new_figure = interpolate_video_figure(
            video_figure, video_info, video_object, frame_index, drow, dcol
        )
        if new_figure is None:
            continue
        if (direction < 0 and new_figure.geometry.to_bbox().bottom < 0) or (
            direction > 0 and new_figure.geometry.to_bbox().top > video_info.frame_height
        ):
            continue

        new_figures.append(new_figure)

    api.video.figure.append_bulk(event.video_id, new_figures, key_id_map)


@app.event(sly.Event.Tools.Rectangle.FigureChanged)
def rectangle_changed(api: sly.Api, event: sly.Event.Tools.Rectangle.FigureChanged):

    if not button_track.is_disabled():
        return

    if getattr(event, "frame", None) is not None:
        # process_video_rectangle_changed(api, event)
        return

    t = datetime.now().timestamp()
    global timestamp
    timestamp = t

    project_meta = g.get_project_meta(api, event.project_id)
    figure = api.image.figure.get_info_by_id(event.figure_state["id"])
    obj_class = project_meta.get_obj_class_by_id(event.figure_state["classId"])

    images, ann_infos, event_img_idx = get_images_and_anns(api, event.dataset_id, event.image_id)
    images[:event_img_idx] = images[:event_img_idx][::-1]
    anns = [sly.Annotation.from_json(x.annotation, project_meta) for x in ann_infos]

    if g.event_anns.get(event.user_id) is None:
        return
    _event_anns = g.event_anns[event.user_id].get(event.image_id)
    if _event_anns is None or _event_anns == g.DONE:
        return

    event_ann = None
    for tag in figure.tags:
        if tag.get("value") is not None:
            if _event_anns.get(tag["value"]) is not None:
                event_ann = _event_anns[tag["value"]]
                break

    if event_ann is None:
        return

    for label in event_ann.labels:
        if label.tags.has_key(g.TRACK_TAG_NAME):

            track_tag: sly.Tag = label.tags.get(g.TRACK_TAG_NAME)
            for tag in figure.tags:
                if track_tag.value == tag["value"]:

                    bb: sly.Rectangle = label.geometry
                    dtop = bb.top - figure.bbox.top
                    dleft = bb.left - figure.bbox.left
                    dbottom = bb.bottom - figure.bbox.bottom
                    dright = bb.right - figure.bbox.right

                    new_event_ann = event_ann.delete_label(label)
                    new_event_ann = new_event_ann.add_label(label.clone(geometry=figure.bbox))
                    g.event_anns[event.user_id][event.image_id][tag["value"]] = new_event_ann
                    break

    image_ids, new_anns = [], []
    for image, ann in zip(images, anns):
        if image.id == event.image_id:
            continue
        temp_ann = ann.clone()
        for label in ann.labels:
            track_tag: sly.Tag = label.tags.get(g.TRACK_TAG_NAME)
            for tag in figure.tags:
                if track_tag.value == tag["value"]:

                    bb: sly.Rectangle = label.geometry
                    try:
                        new_bbox = sly.Rectangle(
                            bb.top - dtop, bb.left - dleft, bb.bottom - dbottom, bb.right - dright
                        )
                    except ValueError:  # == figure not exists
                        new_anns.append(temp_ann)
                        image_ids.append(image.id)
                        break
                    temp_ann = temp_ann.delete_label(label)
                    new_anns.append(temp_ann.add_label(label.clone(geometry=new_bbox)))
                    image_ids.append(image.id)
                    break

    if t == timestamp:
        api.annotation.upload_anns(image_ids, new_anns)


def process_video(api: sly.Api, event: sly.Event.FigureCreated):
    project_meta = g.get_project_meta(api, event.project_id)

    ann_json = api.video.annotation.download(event.video_id)
    key_id_map = sly.KeyIdMap()
    video_ann = sly.VideoAnnotation.from_json(ann_json, project_meta, key_id_map)
    video_info = api.video.get_info_by_id(event.video_id)

    processed_vidfigures: List[sly.VideoFigure] = []

    end_frame = min(event.frame + 2, video_info.frames_count)
    start_frame = max(event.frame - 2, 0)

    frame_indexes = [x for x in range(end_frame + 1)][start_frame:]
    event_frame_idx = frame_indexes.index(event.frame)
    frame_indexes[:event_frame_idx] = frame_indexes[:event_frame_idx][::-1]

    try:
        figure = api.image.figure.get_info_by_id(event.figure_state["id"])
        obj_class = project_meta.get_obj_class_by_id(event.figure_state["classId"])
        video_object = sly.VideoObject(
            obj_class,
            key=key_id_map.get_object_key(event.object_id),
            class_id=event.tool_class_id,
        )
        vid_figure = sly.VideoFigure(
            video_object,
            figure.bbox,
            event_frame_idx,
            # key=key_id_map.get_object_key(event.object_id),
            # class_id=event.tool_class_id,
            track_id=event.object_id,
        )
        processed_vidfigures.append(vid_figure)
    except AttributeError:
        for frame in ann_json["frames"]:
            if frame["index"] == event.frame:
                for figure in frame["figures"]:
                    object_json = next(
                        (item for item in ann_json["objects"] if item["id"] == figure["objectId"]),
                        None,
                    )
                    vid_object = sly.VideoObject.from_json(object_json, project_meta).clone(
                        key=key_id_map.get_object_key(figure["objectId"]),
                        class_id=object_json["classId"],
                    )
                    bbox = sly.Rectangle.from_json(figure["geometry"])
                    processed_vidfigures.append(
                        sly.VideoFigure(vid_object, bbox, event.frame, track_id=figure["objectId"])
                    )
                break

    for video_figure in processed_vidfigures:

        # video_object = sly.VideoObject(
        #     obj_class,
        #     key=key_id_map.get_object_key(event.object_id),
        #     class_id=event.tool_class_id,
        # )

        new_figures = []
        drow, dcol = 0, 0
        for idx, frame_index in enumerate(frame_indexes):

            if idx == event_frame_idx:
                drow, dcol = 0, 0
                continue

            direction = -1 if idx < event_frame_idx else 1
            frame_id = f"{event.video_id}_{frame_index}"

            _drow, _dcol = get_increment_by_diff(
                api, event, frame_id, direction, reverse_direction=True, modality="video"
            )

            drow += _drow * direction
            dcol += _dcol * direction

            new_figure = interpolate_video_figure(
                video_figure, video_info, video_figure.video_object, frame_index, drow, dcol
            )
            if new_figure is None:
                continue
            if (direction < 0 and new_figure.geometry.to_bbox().bottom < 0) or (
                direction > 0 and new_figure.geometry.to_bbox().top > video_info.frame_height
            ):
                continue

            new_figures.append(new_figure)

        api.video.figure.append_bulk(event.video_id, new_figures, key_id_map)

        # if g.event_anns[event.user_id][frame_id] == "done":
        #     g.event_anns[event.user_id][frame_id] = {}
        # g.event_anns[event.user_id][frame_id][video_figure.track_id] = video_figure


@app.event(sly.Event.FigureCreated)
def figure_created(api: sly.Api, event: sly.Event.FigureCreated):

    if not button_track.is_disabled():
        return

    t = datetime.now().timestamp()
    global timestamp
    timestamp = t

    project_meta = g.get_project_meta(api, event.project_id)
    figure = api.image.figure.get_info_by_id(event.figure_state["id"])
    obj_class = project_meta.get_obj_class_by_id(event.figure_state["classId"])

    if figure.geometry_type != sly.Rectangle.name():
        sly.logger.warning(
            "Created figure %r is not a rectangle. Skipping figure.", figure.geometry_type
        )
        return
    sly.logger.info("Start figure interpolation: ID=%s", figure.id)

    if event.frame is not None:
        process_video(api, event)
        return

    images, event_img_idx = get_image_infos(api, event.dataset_id, event.image_id)
    images[:event_img_idx] = images[:event_img_idx][::-1]
    project_meta = g.get_project_meta(api, event.project_id)

    track_tag_meta = project_meta.get_tag_meta(g.TRACK_TAG_NAME)
    # if track_tag_meta is None:
    #     track_tag_meta = sly.TagMeta(g.TRACK_TAG_NAME, sly.TagValueType.ANY_STRING)
    #     project_meta = api.project.update_meta(
    #         event.project_id, project_meta.add_tag_meta(track_tag_meta)
    #     )

    track_tag = sly.Tag(track_tag_meta, sly.rand_str(10))

    new_labels_dct = {}
    drow, dcol = 0, 0
    for idx, image in enumerate(images):
        if idx == event_img_idx:
            drow, dcol = 0, 0
            bb = sly.Rectangle(
                figure.bbox.top,
                figure.bbox.left,
                figure.bbox.bottom,
                figure.bbox.right,
                sly_id=figure.id,
            )
            new_label = sly.Label(bb, obj_class, [track_tag])

            new_labels_dct[image.id] = new_label
            continue

        direction = -1 if idx < event_img_idx else 1
        _drow, _dcol = get_increment_by_diff(
            api, event, image.id, direction, reverse_direction=True
        )
        drow += _drow * direction
        dcol += _dcol * direction

        new_label = interpolate_label(image, figure, obj_class, track_tag, drow, dcol)
        if new_label is None:
            continue
        if (direction < 0 and new_label.geometry.to_bbox().bottom < 0) or (
            direction > 0 and new_label.geometry.to_bbox().top > image.height
        ):
            continue

        new_labels_dct[image.id] = new_label

    update_annotations(api, event, new_labels_dct, track_tag, t)


@sly.timeit
def update_annotations(
    api: sly.Api,
    event: sly.Event.FigureCreated,
    new_labels_dct: Dict[int, sly.Label],
    track_tag: sly.Tag,
    t: float,
):
    images, event_img_idx = get_image_infos(api, event.dataset_id, event.image_id)

    images, ann_infos, event_img_idx = get_images_and_anns(api, event.dataset_id, event.image_id)
    project_meta = g.get_project_meta(api, event.project_id)
    anns = [sly.Annotation.from_json(x.annotation, project_meta) for x in ann_infos]

    image_ids, new_anns = [], []
    for image, ann in zip(images, anns):

        new_label = new_labels_dct.get(image.id)
        if new_label is not None:
            tmp_ann = ann.clone()

            for label in ann.labels:
                if label.geometry.sly_id == new_label.geometry.sly_id:
                    tmp_ann = tmp_ann.delete_label(label)

            image_ids.append(image.id)
            new_anns.append(tmp_ann.add_label(new_label))

    if t == timestamp:
        if len(new_anns) > 0:
            sly.logger.debug("Upload anns for images: {}".format(image_ids))
            api.annotation.upload_anns(image_ids, new_anns)

        if g.event_anns[event.user_id][event.image_id] == "done":
            g.event_anns[event.user_id][event.image_id] = {}
        event_ann = new_anns[image_ids.index(event.image_id)]
        g.event_anns[event.user_id][event.image_id][track_tag.value] = event_ann


@checkbox.value_changed
def checkbox(value):
    if value is False:
        button_track.enable()
    else:
        button_track.disable()


@button_track.click
def manual_track():
    api: sly.Api = g.api
    # user_id = sly.env.user_id()
    # event: sly.Event.ManualSelected.ImageChanged = g.curr_event.get(user_id)

    global curr_event_tmp
    event = curr_event_tmp

    if event is None:
        sly.logger.info("Please refresh the page")
        status_image_not_initialized.show()
        return
    status_image_not_initialized.hide()

    if getattr(event, "frame", None) is not None:
        process_video(api, event)
        return

    project_meta = g.get_project_meta(api, event.project_id)

    track_tag_meta = project_meta.get_tag_meta(g.TRACK_TAG_NAME)
    # if track_tag_meta is None:
    #     track_tag_meta = sly.TagMeta(g.TRACK_TAG_NAME, sly.TagValueType.ANY_STRING)
    #     project_meta = api.project.update_meta(
    #         event.project_id, project_meta.add_tag_meta(track_tag_meta)
    #     )

    images, ann_infos, event_img_idx = get_images_and_anns(api, event.dataset_id, event.image_id)
    images[:event_img_idx] = images[:event_img_idx][::-1]
    anns = [sly.Annotation.from_json(x.annotation, project_meta) for x in ann_infos]
    anns[:event_img_idx] = anns[:event_img_idx][::-1]

    img_center: sly.ImageInfo = images[event_img_idx]
    ann_center: sly.Annotation = anns[event_img_idx]

    sly.logger.info("Start figure interpolation")

    track_tag_vals = {}
    for label in ann_center.labels:
        if not any(tag.name == track_tag_meta.name for tag in label.tags):
            track_tag_vals[label.geometry.sly_id] = sly.rand_str(10)

    new_anns = []
    new_image_ids = []
    drow, dcol = 0, 0

    direction = -1
    for image, ann in zip(images, anns):
        if image.id == img_center.id:
            drow, dcol = 0, 0
            direction = 1
            new_labels = []
            for label in ann.labels:
                if label.tags.has_key(track_tag_meta.name):
                    new_labels.append(label)
                else:
                    new_labels.append(
                        label.add_tag(
                            sly.Tag(track_tag_meta, track_tag_vals[label.geometry.sly_id])
                        )
                    )
            new_anns.append(ann.clone(labels=new_labels))
            new_image_ids.append(image.id)
            continue

        _drow, _dcol = get_increment_by_diff(
            api, event, image.id, direction, reverse_direction=True
        )

        drow += _drow * direction
        dcol += _dcol * direction

        new_labels = [label for label in ann.labels]
        for label in ann_center.labels:
            if label.geometry.sly_id in track_tag_vals:

                label_drow, label_dcol = drow, dcol
                # res = get_increment_by_points(image.id, label, direction, reverse_direction=True)
                # if res is not None:
                #     label_drow, label_dcol = res

                val = track_tag_vals[label.geometry.sly_id]
                new_label = interpolate_label(
                    image,
                    label,
                    label.obj_class,
                    sly.Tag(track_tag_meta, val),
                    label_drow,
                    label_dcol,
                )
                if new_label is None:
                    continue
                if (direction < 0 and new_label.geometry.to_bbox().bottom < 0) or (
                    direction > 0 and new_label.geometry.to_bbox().top > image.height
                ):
                    continue
                new_labels.append(new_label)

        new_anns.append(ann.clone(labels=new_labels))
        new_image_ids.append(image.id)

    api.annotation.upload_anns(new_image_ids, new_anns)


@sly.timeit
def lightglue(image0, image1, max_num_keypoints=1024, device="cpu"):

    extractor = (
        SuperPoint(max_num_keypoints=max_num_keypoints, model_dir=g.MODEL_DIR).eval().to(device)
    )
    matcher = (
        LightGlue(
            features="superpoint",
            depth_confidence=0.9,
            width_confidence=0.95,
            filter_threshold=0.1,
            # n_layers=3,
            model_dir=g.MODEL_DIR,
        )
        .eval()
        .to(device)
    )

    feats0 = extractor.extract(image0.to(device), resize=256)
    feats1 = extractor.extract(image1.to(device), resize=256)

    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, _matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[_matches[..., 0]], kpts1[_matches[..., 1]]

    return m_kpts0, m_kpts1, kpts0, kpts1, matches01


def interpolate_label(
    image: sly.ImageInfo,
    label: Union[sly.FigureInfo, sly.Label],
    obj_class: sly.ObjClass,
    track_tag: sly.Tag,
    drow: int,
    dcol: int,
):
    res_tags = [tag for tag in label.tags]
    if track_tag.name in [x.name for x in res_tags]:
        return None

    res_tags.append(track_tag)
    figure_bbox: sly.Rectangle = (
        label.bbox if isinstance(label, sly.FigureInfo) else label.geometry.to_bbox()
    )
    bbox = figure_bbox.translate(drow, dcol)
    try:
        bbox = bbox.crop(sly.Rectangle(*(0, 0, image.height, image.width)))[0]
    except IndexError:
        return None

    return sly.Label(bbox, obj_class, res_tags)


def interpolate_video_figure(
    video_figure: sly.VideoFigure,
    video_info: VideoInfo,
    video_object: sly.VideoObject,
    frame_index: int,
    drow: int,
    dcol: int,
):
    figure_bbox: sly.Rectangle = video_figure.geometry.to_bbox()
    bbox = figure_bbox.translate(drow, dcol)
    try:
        bbox = bbox.crop(sly.Rectangle(*(0, 0, video_info.frame_height, video_info.frame_width)))[0]
    except IndexError:
        return None

    return sly.VideoFigure(video_object, bbox, frame_index, track_id=video_figure.track_id)


def get_mask(coordinates, xmin, ymin, xmax, ymax):
    return (
        (coordinates[:, 0] >= xmin)
        & (coordinates[:, 0] <= xmax)
        & (coordinates[:, 1] >= ymin)
        & (coordinates[:, 1] <= ymax)
    )


def get_increment_by_diff(
    api,
    event,
    image_or_frame_id: Union[str, int],
    direction: int,
    reverse_direction: bool = False,
    modality: Literal["images", "video"] = "images",
):
    if modality == "images":
        points = get_points_image(api, event, image_or_frame_id)
    elif modality == "video":
        points = get_points_video(api, event, image_or_frame_id)
    _dir = direction * -1 if reverse_direction else direction

    _dcol, _drow = points[_dir]["diff"]
    return _drow, _dcol


def get_increment_by_points(
    api, event, image_id: int, label: sly.Label, direction: int, reverse_direction: bool = False
) -> Optional[Tuple[int]]:

    points = get_points_image(api, event, image_id)
    if points is None:
        return

    _dir = direction * -1 if reverse_direction else direction
    if len(points[_dir]["points"]) == 0:
        return
    m_kpts0, m_kpts1 = points[_dir]["points"]

    bbox: sly.Rectangle = label.geometry.to_bbox()

    mask = get_mask(m_kpts0, bbox.left, bbox.top, bbox.right, bbox.bottom)
    if mask.sum() < 3:
        return

    # bbox_pts0 = m_kpts0[mask]
    # bbox_pts_indices = torch.nonzero(mask).squeeze()

    bbox_pts0, bbox_pts1 = m_kpts0[mask], m_kpts1[mask]

    # curr_image_path = g.SLY_APP_DATA + "/" + str(g.curr_event.image_id) + ".png"
    # image_path = g.SLY_APP_DATA + "/" + str(image_id) + ".png"
    # image0 = load_image(curr_image_path).cpu()
    # image1 = load_image(image_path).cpu()

    # viz2d.plot_images([image0, image1])
    # viz2d.plot_matches(bbox_pts0, bbox_pts1, color="lime", lw=0.2)
    # plot_path = (
    #     g.SLY_APP_DATA
    #     + "/"
    #     + str(g.curr_event.image_id)
    #     + "_"
    #     + str(image_id)
    #     + "_"
    #     + str(label.geometry.sly_id)
    #     + ".png"
    # )
    # viz2d.save_plot(plot_path)

    topleft_pts = []

    for arr in [bbox_pts0, bbox_pts1]:
        sorted_indices = torch.argsort(arr[:, 0])
        filtered_coords_sorted_by_x = arr[sorted_indices]
        topleft_pts.append(filtered_coords_sorted_by_x[0])

    dcol, drow = (topleft_pts[1] - topleft_pts[0]).type(torch.int).numpy().tolist()

    return drow, dcol
