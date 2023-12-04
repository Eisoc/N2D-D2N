import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # verbose
    parser.add_argument("--load-path", default=None, help="load model")
    parser.add_argument("--save", action="store_true", help="save model chekcpoints")
    parser.add_argument("--save-path", default="./", help="save model")
    parser.add_argument("--save-interval", default=1000, type=int, help="frequency to save checkpoint")
    parser.add_argument("--output-path", default=None, type=str, help="directory to save those visualization results")
    parser.add_argument("--verbose", action="store_true", help="log output to tensorboard.")
    parser.add_argument("--debug-seed", type=int, default=0,  help="random seed (default: 0)")

    # train
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs to train")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--split", default=-1, type=float)

    # model
    parser.add_argument(
        "--with-ppm", action="store_true", help="if to run with ppm (Pyramid Pooling Module) in the decoder"
    )
    parser.add_argument(
        "--flow-refinement",
        default="none",
        choices=["none", "lightweight", "hourglass"],
        help="type of the optical flow refiment module",
    )
    parser.add_argument("--corr-radius", type=int, default=4, help="search radius of the correlation layer")
    parser.add_argument("--no-occ", action="store_true", help="no occlusion prediction.")
    parser.add_argument("--cat-occ", action="store_true", help="use occlusion predictions in the decoder hierarchy")
    parser.add_argument("--upsample-flow-output", action="store_true")
    parser.add_argument("--n-ref", type=int, default=1)

    # optimizer
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument(
        "--lr-steps", default=[100000000000], type=int, nargs="+", help="stepsize of changing the learning rate"
    )
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="learning rate will be multipled by this gamma")

    # loss
    parser.add_argument("--min-depth", type=float, help="minimum depth", default=0.1)
    parser.add_argument("--max-depth", type=float, help="maximum depth", default=100000.0)
    parser.add_argument("--depth-weight", type=float, default=1)
    parser.add_argument("--flow-weight", type=float, default=1)
    parser.add_argument("--flow-smooth-weight", type=float, default=0)
    parser.add_argument("--smooth", type=float, default=0)
    parser.add_argument("--occ", type=float, default=0)

    return parser
