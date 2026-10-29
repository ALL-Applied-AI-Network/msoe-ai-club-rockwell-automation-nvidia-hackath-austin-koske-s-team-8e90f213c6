"""Append the replay owner's public QA export to an existing review batch."""
import argparse
import json
from pathlib import Path
from .batch import append_replay_qa


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--batch',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(append_replay_qa(args.manifest,args.dataset,args.batch),indent=2))


if __name__=='__main__':main()
