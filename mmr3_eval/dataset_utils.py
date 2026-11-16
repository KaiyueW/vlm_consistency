"""Loads the items of each task folder under data/.

Expected layout (per task):
    data/<task>/image/<id>.jpg     the question rendered as an image
    data/<task>/text/<id>          the same question as text      (<id>.txt also works)
    data/<task>/answer/<id>        the ground-truth answer (plain text)   (<id>.txt also works)
"""

import os


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# <changed> new: the text / answer file may be named "<id>" or "<id>.txt"; use whichever exists
def _find(folder, i):
    for name in (i, i + ".txt"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def list_tasks(data_root):
    return sorted(os.listdir(data_root))


def load_task(data_root, task, limit=None):
    """id, image_path, question, answer"""
    task_dir = os.path.join(data_root, task)
    text_dir = os.path.join(task_dir, "text")
    answer_dir = os.path.join(task_dir, "answer")

    # get ids from the images' names
    # <changed> plain sorted(): key=int crashes if any id is not a pure number
    ids = sorted(f[:-4] for f in os.listdir(os.path.join(task_dir, "image")))

    # <changed> uses _find (see above) to check that the text and the answer exist for this image
    ids = [i for i in ids if _find(text_dir, i) and _find(answer_dir, i)]

    if limit:
        ids = ids[:limit]

    return [
        {
            "id": i,
            "image_path": os.path.join(task_dir, "image", f"{i}.jpg"),
            "question": _read(_find(text_dir, i)),
            "answer": _read(_find(answer_dir, i)),
        }
        for i in ids
    ]