# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MosaicTrainingHook(Hook):
    """Switch mosaic off for the final training epochs."""

    def __init__(self, close_mosaic_epochs=10):
        self.close_mosaic_epochs = int(close_mosaic_epochs)
        self._last_enabled = None

    def _update_dataset(self, dataset, enabled):
        if hasattr(dataset, 'dataset'):
            self._update_dataset(dataset.dataset, enabled)

        pipeline = getattr(dataset, 'pipeline', None)
        transforms = getattr(pipeline, 'transforms', [])
        for transform in transforms:
            if hasattr(transform, 'set_mosaic_enabled'):
                transform.set_mosaic_enabled(enabled)

    def before_train_epoch(self, runner):
        enabled = runner.epoch < (runner.max_epochs - self.close_mosaic_epochs)
        self._update_dataset(runner.data_loader.dataset, enabled)

        if enabled != self._last_enabled:
            state = 'enabled' if enabled else 'disabled'
            runner.logger.info(
                'Mosaic augmentation %s at epoch %d/%d', state,
                runner.epoch + 1, runner.max_epochs)
            self._last_enabled = enabled
