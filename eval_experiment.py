import glob
from functools import partial
import pandas as pd
from eval_cluster_utils import *


def plot_scatter(x_axis, values, outdir, fname, xlab="epoch",
                 ylab="AUROC", title="OOD AUROC & score CIFAR100 -> CIFAR10"):
    plt.figure(figsize=(10,10))
    plt.plot(x_axis, values, "-o")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(Path(outdir) / fname)


def _eval_setting_to_str(s):
    if not isinstance(s, tuple):
        return str(s)
    s = [str(x) for x in s]
    return '-'.join(s)


def print_results(d):
    for k, d_inner in d.items():
        print(k)
        for k_inner, v in d_inner.items():
            s = f'{_eval_setting_to_str(k_inner)}:'
            print(f'\t{s:<22} {v[-1]:.2f}')


def load_tensorboard_loss(path):
    tag = 'Train loss epoch'
    event_acc = EventAccumulator(str(next(path.glob('event*'))))
    event_acc.Reload()
    if tag in event_acc.Tags()['scalars']:
        return pd.DataFrame([{'Epoch': ev.step, 'loss': ev.value}
                             for ev in event_acc.Scalars(tag)]).set_index('Epoch')
    # Multihead case
    dfs = []
    for p in path.rglob('Train loss*/event*'):
        event_acc = EventAccumulator(str(p))
        event_acc.Reload()
        dfs.append(pd.DataFrame([{'Epoch': ev.step, 'loss': ev.value}
                                 for ev in event_acc.Scalars(tag)]).set_index('Epoch'))
    df = pd.concat(dfs)
    return df.groupby('Epoch').min()


def main():
    args = get_eval_args()
    cudnn.deterministic = True

    auroc_results = defaultdict(partial(defaultdict, list))
    cluster_results = {"cluster_acc": [], "nmi": [], "anmi": [], "ari": [],
                       "cluster_acc-train": [], "nmi-train": [], "anmi-train": [], "ari-train": []}
    loss_results = {"train_loss": []}

    checkpoint_list = glob.glob(os.path.join(args.ckpt_folder, "*.pth"))
    outdir = Path(args.ckpt_folder).expanduser().resolve()

    # Read hparams
    with open(outdir / 'hp.json', 'r') as f:
        hparams = json.load(f)
    if not args.ignore_hp_file:
        args.__dict__.update({k: v for k, v in hparams.items() if v is not None})

    # Load loss history
    # losses_df = load_tensorboard_loss(outdir)

    # replace last saved checkpoint name to be last
    checkpoint_list = list(map(lambda st: str.replace(st, "checkpoint.pth", "checkpoint9999.pth"), checkpoint_list))
    checkpoint_list = sorted(checkpoint_list)
    checkpoint_list = list(map(lambda st: str.replace(st, "checkpoint9999.pth", "checkpoint.pth", ), checkpoint_list))
    epochs = []

    print(f"dataset: {args.dataset} \n Checkpoints found {len(checkpoint_list)}  \n {checkpoint_list} ")
    assert len(checkpoint_list) >= 1
    args.datapath = './data' if  args.dataset in ["CIFAR10", "CIFAR100", "STL10", "CIFAR20"] else args.datapath
    extractor = None
    for ckpt in checkpoint_list:
        print(ckpt)
        # Epoch number for next epoch is saved in the checkpoint
        epoch = torch.load(ckpt, map_location='cpu')['epoch'] - 1
        epochs.append(epoch)
        if extractor is None or args.no_cache:
            extractor = FeatureExtractionPipeline(args, cache_backbone=not args.no_cache, datapath=args.datapath)
        train_features, test_features, train_labels, val_labels = \
            extractor.get_features(ckpt)

        # Cluster performance test
        ( _ , max_indices) = torch.max(test_features, dim=1)
        max_indices = max_indices.cpu().numpy()
        cluster_acc, nmi, anmi, ari = utils.compute_metrics(val_labels, max_indices, min_samples_per_class=5)
        print(f'acc={cluster_acc}, nmi={nmi}')

        print('\n', '-'*100, '\n')


if __name__ == '__main__':
    main()

"""
python eval_experiment.py --ckpt_folder ./experiments/TEMI-output-test
"""