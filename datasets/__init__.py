# -*- coding: utf-8 -*-

from .datafeeder_wavenet import DataFeederWavenet


class BaseDataset:
    def __init__(self, in_path, out_path) -> None:
        self.in_path = in_path
        self.out_path = out_path
