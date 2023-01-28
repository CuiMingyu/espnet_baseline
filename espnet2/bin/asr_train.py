#!/usr/bin/env python3
import sys
import os
# this is for debugging in vscode only, vscode seems get some
# issue when parsing python path
espnetdir = os.path.dirname(os.path.abspath(__file__)) + "/../../"
sys.path = [espnetdir] + sys.path
from espnet2.train.build_task import ASRTask




def get_parser():
    parser = ASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
