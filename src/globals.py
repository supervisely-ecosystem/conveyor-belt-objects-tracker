from collections import defaultdict
from typing import *

from dotenv import load_dotenv

import supervisely as sly

if sly.is_development():
    load_dotenv("local.env")

TRACK_TAG_NAME = "track_id"
DONE = "done"

SLY_APP_DATA = sly.app.get_data_dir()
sly.fs.clean_dir(SLY_APP_DATA)

api: sly.Api = None
# curr_event: Dict[int, Event.ManualSelected.ImageChanged] = dict()
MODEL_DIR = "./checkpoints"
previous_image: int = None
previous_frame: int = None

event_anns: Dict[int, Dict[int, Dict[str, sly.Annotation]]] = defaultdict(lambda: defaultdict(dict))

project_metas = {}


def get_project_meta(api: sly.Api, project_id: int) -> sly.ProjectMeta:
    if project_id not in project_metas:
        project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
        if project_meta.get_tag_meta(TRACK_TAG_NAME) is None:
            track_tag_meta = sly.TagMeta(TRACK_TAG_NAME, sly.TagValueType.ANY_STRING)
            project_meta = api.project.update_meta(
                project_id, project_meta.add_tag_meta(track_tag_meta)
            )
        project_metas[project_id] = project_meta
    else:
        project_meta = project_metas[project_id]
    return project_meta
