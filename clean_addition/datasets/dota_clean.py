

from .builder import DATASETS
from .voc import VOCDatasetPNG

@DATASETS.register_module()
class DotaDatasetHBB(VOCDatasetPNG):

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 
               'harbor', 'swimming-pool', 'helicopter', 'container-crane')

    def __init__(self, **kwargs):
        super(VOCDatasetPNG, self).__init__(**kwargs)
        self.year = 2012