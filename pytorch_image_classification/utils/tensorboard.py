import pathlib

from torch.utils.tensorboard import SummaryWriter
import yacs.config


class DummyWriter(SummaryWriter):
    def __init__(self):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass

    def add_scalars(self,
                    main_tag,
                    tag_scalar_dict,
                    global_step=None,
                    walltime=None):
        pass

    def export_scalars_to_json(self, path):
        pass

    def add_histogram(self,
                      tag,
                      values,
                      global_step=None,
                      bins='tensorflow',
                      walltime=None,
                      max_bins=None):
        pass

    def add_histogram_raw(self,
                          tag,
                          min,
                          max,
                          num,
                          sum,
                          sum_squares,
                          bucket_limits,
                          bucket_counts,
                          global_step=None,
                          walltime=None):
        pass

    def add_image(self,
                  tag,
                  img_tensor,
                  global_step=None,
                  walltime=None,
                  dataformats='CHW'):
        pass

    def add_images(self,
                   tag,
                   img_tensor,
                   global_step=None,
                   walltime=None,
                   dataformats='NCHW'):
        pass

    def add_image_with_boxes(self,
                             tag,
                             img_tensor,
                             box_tensor,
                             global_step=None,
                             walltime=None,
                             dataformats='CHW',
                             **kwargs):
        pass

    def add_figure(self,
                   tag,
                   figure,
                   global_step=None,
                   close=True,
                   walltime=None):
        pass

    def add_video(self,
                  tag,
                  vid_tensor,
                  global_step=None,
                  fps=4,
                  walltime=None):
        pass

    def add_audio(self,
                  tag,
                  snd_tensor,
                  global_step=None,
                  sample_rate=44100,
                  walltime=None):
        pass

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        pass

    def add_onnx_graph(self, prototxt):
        pass

    def add_graph(self, model, input_to_model=None, verbose=False, **kwargs):
        pass

    def add_embedding(self,
                      mat,
                      metadata=None,
                      label_img=None,
                      global_step=None,
                      tag='default',
                      metadata_header=None):
        pass

    def add_pr_curve(self,
                     tag,
                     labels,
                     predictions,
                     global_step=None,
                     num_thresholds=127,
                     weights=None,
                     walltime=None):
        pass

    def add_pr_curve_raw(self,
                         tag,
                         true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         global_step=None,
                         num_thresholds=127,
                         weights=None,
                         walltime=None):
        pass

    def add_custom_scalars_multilinechart(self,
                                          tags,
                                          category='default',
                                          title='untitled'):
        pass

    def add_custom_scalars_marginchart(self,
                                       tags,
                                       category='default',
                                       title='untitled'):
        pass

    def add_custom_scalars(self, layout):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def create_tensorboard_writer(config: yacs.config.CfgNode,
                              output_dir: pathlib.Path,
                              purge_step: int) -> SummaryWriter:
    if config.train.use_tensorboard:
        if config.train.start_epoch == 0:
            return SummaryWriter(output_dir.as_posix())
        else:
            return SummaryWriter(output_dir.as_posix(), purge_step=purge_step)
    else:
        return DummyWriter()
