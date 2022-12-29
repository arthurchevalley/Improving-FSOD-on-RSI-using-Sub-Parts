# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import logging
from mmcv.utils import get_logger
from mmcv.runner import HOOKS, master_only
from mmcv.runner.hooks import LoggerHook


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger


def get_caller_name():
    """Get name of caller method."""
    # this_func_frame = inspect.stack()[0][0]  # i.e., get_caller_name
    # callee_frame = inspect.stack()[1][0]  # e.g., log_img_scale
    caller_frame = inspect.stack()[2][0]  # e.g., caller of log_img_scale
    caller_method = caller_frame.f_code.co_name
    try:
        caller_class = caller_frame.f_locals['self'].__class__.__name__
        return f'{caller_class}.{caller_method}'
    except KeyError:  # caller is a function
        return caller_method


def log_img_scale(img_scale, shape_order='hw', skip_square=False):
    """Log image size.

    Args:
        img_scale (tuple): Image size to be logged.
        shape_order (str, optional): The order of image shape.
            'hw' for (height, width) and 'wh' for (width, height).
            Defaults to 'hw'.
        skip_square (bool, optional): Whether to skip logging for square
            img_scale. Defaults to False.

    Returns:
        bool: Whether to have done logging.
    """
    if shape_order == 'hw':
        height, width = img_scale
    elif shape_order == 'wh':
        width, height = img_scale
    else:
        raise ValueError(f'Invalid shape_order {shape_order}.')

    if skip_square and (height == width):
        return False

    logger = get_root_logger()
    caller = get_caller_name()
    logger.info(f'image shape: height={height}, width={width} in {caller}')

    return True



@HOOKS.register_module()
class CometMLLoggerHook(LoggerHook):

    def __init__(self,
                 project_name=None,
                 hyper_params=None,
                 import_comet=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True,
                 api_key=None):
        """Class to log metrics to Comet ML.
        It requires `comet_ml` to be installed.
        Args:
            project_name (str, optional):
                Send your experiment to a specific project. 
                Otherwise will be sent to Uncategorized Experiments. 
                If project name does not already exists Comet.ml will create 
                a new project.
            hyper_params (dict, optional): Logs a dictionary 
                (or dictionary-like object) of multiple parameters.
            import_comet (bool optional): Whether to import comet_ml before run.
                WARNING: Comet ML have to be imported before sklearn and torch,
                or COMET_DISABLE_AUTO_LOGGING have to be set in the environment.
            interval (int): Logging interval (every k iterations).
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`.
            reset_flag (bool): Whether to clear the output buffer after logging
            by_epoch (bool): Whether EpochBasedRunner is used.
        """
        super(CometMLLoggerHook, self).__init__(interval, ignore_last,
                                                reset_flag, by_epoch)
        self._import_comet = import_comet
        if import_comet:
            self.import_comet()
        self.project_name = project_name
        self.hyper_params = hyper_params
        self._api_key = api_key

    def import_comet(self):
        try:
            import comet_ml
        except ImportError:
            raise ImportError(
                'Please run "pip install comet_ml" to install Comet ML')
        self.comet_ml = comet_ml

    #@master_only
    def before_run(self, runner):
        if self._import_comet:
            self.experiment = self.comet_ml.Experiment(
                api_key=self._api_key,
                project_name=self.project_name
            )
        else:
            self.experiment = comet_ml.Experiment(
                api_key=self._api_key,
                project_name=self.project_name,
            )
        if self.hyper_params is not None:
            self.experiment.log_parameters(self.hyper_params)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)

        for tag, val in tags.items():
            self.experiment.log_metric(name=tag,
                                       value=val,
                                       step=self.get_iter(runner),
                                       epoch=self.get_epoch(runner))
                

    @master_only
    def after_run(self, runner):
        self.experiment.end()

      