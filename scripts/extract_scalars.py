#!/usr/bin/env python

import argparse
import json
import pathlib
from tensorboard.backend.event_processing import event_accumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str)
    args = parser.parse_args()

    event_acc = event_accumulator.EventAccumulator(
        args.path, size_guidance={'scalars': 0})
    event_acc.Reload()

    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        scalars[tag] = [event.value for event in events]

    if args.output_dir is not None:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = pathlib.Path(args.path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    outpath = output_dir / 'all_scalars.json'
    with open(outpath, 'w') as fout:
        json.dump(scalars, fout)


if __name__ == '__main__':
    main()
