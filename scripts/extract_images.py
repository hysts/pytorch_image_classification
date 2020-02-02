#!/usr/bin/env python

import argparse
import pathlib
import numpy as np
import cv2
from tensorboard.backend.event_processing import event_accumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str)
    args = parser.parse_args()

    event_acc = event_accumulator.EventAccumulator(args.path,
                                                   size_guidance={'images': 0})
    event_acc.Reload()

    if args.output_dir is not None:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = pathlib.Path(args.path).parent / 'images'
    output_dir.mkdir(exist_ok=True, parents=True)

    for tag in event_acc.Tags()['images']:
        events = event_acc.Images(tag)

        tag_name = tag.replace('/', '_')
        dirpath = output_dir / tag_name
        dirpath.mkdir(exist_ok=True, parents=True)

        for index, event in enumerate(events):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(s, cv2.IMREAD_COLOR)
            outpath = dirpath / f'{index:04}.jpg'
            cv2.imwrite(outpath.as_posix(), image)


if __name__ == '__main__':
    main()
