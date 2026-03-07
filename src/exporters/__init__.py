# coding: utf-8

from __future__ import annotations


def run_face_export(*args, **kwargs):
    from .export_face import run_face_export as _run_face_export

    return _run_face_export(*args, **kwargs)


def run_audio_export(*args, **kwargs):
    from .export_audio import run_audio_export as _run_audio_export

    return _run_audio_export(*args, **kwargs)


def run_scene_export(*args, **kwargs):
    from .export_scene import run_scene_export as _run_scene_export

    return _run_scene_export(*args, **kwargs)


def run_text_export(*args, **kwargs):
    from .export_text import run_text_export as _run_text_export

    return _run_text_export(*args, **kwargs)


__all__ = ["run_face_export", "run_audio_export", "run_scene_export", "run_text_export"]
