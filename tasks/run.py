import os
# 強制禁用 TensorFlow 的 GPU 可見性，防止它搶佔 PyTorch 的資源
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    class DummyImpImporter:
        def __init__(self, path=None): pass
    pkgutil.ImpImporter = DummyImpImporter

import importlib
from utils.hparams import set_hparams, hparams


def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()


if __name__ == '__main__':
    set_hparams()
    run_task()
