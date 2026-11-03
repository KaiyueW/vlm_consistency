"""Loads the items of each task folder under data/.

Expected layout (per task):
    data/<task>/image/<id>.jpg     the question rendered as an image
    data/<task>/text/<id>          the same question as text
    data/<task>/answer/<id>        the ground-truth answer (plain text)
"""

import os

def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def list_tasks(data_root):
    return sorted(os.listdir(data_root))


def load_task(data_root, task, limit=None):
    """id, image_path, question, answer"""
    task_dir = os.path.join(data_root, task)

    # get ids from the images' names
    ids = sorted(
        [f[:-4] for f in os.listdir(os.path.join(task_dir, "image"))],
        key=int,
    )

    ids = [
        i for i in ids
        if os.path.exists(os.path.join(text_dir, i)) and os.path.exists(os.path.join(answer_dir, i))
    ] # check if answers and texts all exist for this image

    if limit:
        ids = ids[:limit]

    return [
        {
            "id": i,
            "image_path": os.path.join(task_dir, "image", f"{i}.jpg"),
            "question": _read(os.path.join(task_dir, "text", i)),
            "answer": _read(os.path.join(task_dir, "answer", i)),
        }
        for i in ids
    ]