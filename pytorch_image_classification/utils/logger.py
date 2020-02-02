from typing import List, Optional

import logging
import pathlib
import sys

import termcolor


def create_logger(name: str,
                  distributed_rank: int,
                  output_dir: Optional[pathlib.Path] = None,
                  filename: str = 'log.txt') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if distributed_rank > 0:
        return logger

    fvcore_logger = logging.getLogger('fvcore')
    fvcore_logger.setLevel(logging.INFO)

    handlers = _create_handlers(output_dir, filename)
    for handler in handlers:
        logger.addHandler(handler)
        fvcore_logger.addHandler(handler)

    return logger


def _create_handlers(output_dir: Optional[pathlib.Path] = None,
                     filename: str = 'log.txt') -> List[logging.Handler]:
    handlers = []
    color_formatter = _create_color_formatter()
    handlers.append(_create_stream_handler(color_formatter))
    if output_dir is not None:
        handlers.append(
            _create_file_handler(output_dir / filename, color_formatter))

        plain_log_name_parts = filename.split('.')
        plain_log_name_parts[-2] = plain_log_name_parts[-2] + '_plain'
        plain_log_name = '.'.join(plain_log_name_parts)
        plain_formatter = _create_plain_formatter()
        handlers.append(
            _create_file_handler(output_dir / plain_log_name, plain_formatter))
    return handlers


def _create_plain_formatter() -> logging.Formatter:
    return logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S")


def _create_color_formatter() -> logging.Formatter:
    return logging.Formatter(
        termcolor.colored('[%(asctime)s] %(name)s %(levelname)s: ', 'green') +
        '%(message)s',
        datefmt="%Y-%m-%d %H:%M:%S")


def _create_stream_handler(
        formatter: logging.Formatter) -> logging.StreamHandler:
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    return stream_handler


def _create_file_handler(file_path: pathlib.Path,
                         formatter: logging.Formatter) -> logging.FileHandler:
    file_handler = logging.FileHandler(file_path.as_posix())
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    return file_handler
