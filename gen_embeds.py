"""
Precomputes embeddings for a given model and dataset.
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np

from eval_cluster_utils import knn_classifier
from loaders import get_dataset
from model_builders import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def compute_embedding(model, loader):
    embeds = []
    labels = []
    for images, label in tqdm(loader):
        images = images.cuda()
        image_features = model(images).float()
        embeds.append(image_features.cpu())
        labels.append(label)
    return torch.cat(embeds), torch.cat(labels)

@torch.no_grad()
def compute_neighbors(embedding, k):
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    num_embeds = embedding.shape[0]
    if num_embeds <= 8*1e3:
        dists = embedding @ embedding.permute(1, 0)
        # exclude self-similarity
        dists.fill_diagonal_(-torch.inf)
        return dists.topk(k, dim=-1)   
    else:
        topk_knn_ids = []
        topk_knn_dists = []
        print("Chunk-wise implementation of k-nn in GPU")
        # num_chunks = 12000 
        step_size = 64 # num_embeds // num_chunks
        embedding = embedding.to(DEVICE)
        for idx in tqdm(range(0, num_embeds, step_size)):
            idx_next_chunk = min((idx + step_size), num_embeds)
            features = embedding[idx : idx_next_chunk, :]
            # calculate the dot product dist
            dists_chunk = torch.mm(features, embedding.T).cpu()
            dists_chunk.fill_diagonal_(-torch.inf)
            max_dists, indices = dists_chunk.topk(k, dim=-1)
            topk_knn_ids.append(indices)
            topk_knn_dists.append(max_dists)
        return torch.cat(topk_knn_dists), torch.cat(topk_knn_ids)
    
        
def get_outpath(arch, dataset, datapath='data'):
    datapath = Path(datapath).expanduser().resolve()
    arch = arch.replace('/', '_')
    dataset = dataset.replace('/', '_')
    return datapath / 'embeddings' / f'{dataset}-{arch}'

def load_embeddings(datapath):
    datapath = Path(datapath)
    embeddings = np.load(datapath / 'embeddings.npy')
    labels = np.load(datapath / 'labels.npy')
    label_translation = {text: i for i, text in enumerate(set(labels))}
    labels = np.array([label_translation[text] for text in labels])
    return torch.from_numpy(embeddings), torch.from_numpy(labels)

def get_nn(args, preprocess, model, test=False):
    datapath = args.datapath
    embeddings, label = None, None
    n_classes = 0
    if args.only_knn:
        embeddings, label = load_embeddings(datapath)
        n_classes = label.unique().shape[0]
    else:
        dset = get_dataset(args.dataset, datapath=datapath, train=not test)
        dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=16)
        embeddings, label = compute_embedding(model, dataloader)    
        n_classes = len(dset.classes)
    embeddings = embeddings.squeeze()
    k = args.k or embeddings.shape[0] // n_classes 
    nn_dists, neighbors = compute_neighbors(embeddings, k)
    return embeddings, label, nn_dists, neighbors, n_classes 

def compute_stats(outpath):
    for test in True, False:
        test_str = '-test' if test else ''
        embeddings = torch.load(outpath / f'embeddings{test_str}.pt', map_location='cpu')
        torch.save(embeddings.mean(dim=0), outpath / f'mean{test_str}.pt')
        torch.save(embeddings.std(dim=0), outpath / f'std{test_str}.pt')

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelname = args.arch

    outpath = get_outpath(modelname, args.dataset)
    if args.stats_only:
        compute_stats(outpath)
        return

    model, preprocess = load_model(args, head=False)
    if model is not None:
        model = model.to(device)
        model.eval()

    outpath.mkdir(parents=True, exist_ok=True)

    embs = {}
    labels = {}
    
    for test in True, False:
        print('Computing', 'test' if test else 'train', 'dataset embedding')
        embeddings, label, nn_dists, neighbors, num_classes = get_nn(args, preprocess, model, test)
        embeddings, label, nn_dists, neighbors = embeddings.cpu(), label.cpu(), nn_dists.cpu(), neighbors.cpu()

        embs[test] = embeddings
        labels[test] = label
        test_str = '-test' if test else ''
        torch.save(embeddings, outpath / f'embeddings{test_str}.pt')
        torch.save(label, outpath / f'label{test_str}.pt')
        torch.save(neighbors, outpath / f'knn{test_str}.pt')
        torch.save(nn_dists, outpath / f'knn_dists{test_str}.pt')
        torch.save(embeddings.mean(dim=0), outpath / f'mean{test_str}.pt')
        torch.save(embeddings.std(dim=0), outpath / f'std{test_str}.pt')
    
    if not args.no_eval_knn:
        print('Computing KNN accuracy')
        top1, top5 = knn_classifier(
            train_features=embs[False],
            train_labels=labels[False],
            test_features=embs[True],
            test_labels=labels[True],
            k=args.classifier_k,
            T=args.temperature,
            num_classes=num_classes
        )
        print(f'Top-1 accuracy: {top1}, Top-5 accuracy: {top5}')
        with open(outpath / 'accuracy.json', 'w') as f:
            json.dump({'top1': top1, 'top5': top5}, f)
    # empty gpu memory
    if model is not None:
        model = model.cpu()
        del model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='DCASE2018_TASK5', choices=["DCASE2018_TASK5"], type=str)
    parser.add_argument('--arch', default='PaSST')
    parser.add_argument('--only_knn', action='store_true', help='Compute knn for existing embeddings', default=False)
    parser.add_argument('--outpath', type=Path, default=Path('data'))
    parser.add_argument('--temperature', default=0.02, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--classifier-k', default=20, type=int, help='Numbers of neighbors to use in the classifier')
    parser.add_argument('-k', type=int, default=None, help='total NNs to compute. Default: num images / num classes')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--datapath', default='./data', type=str)
    parser.add_argument('--no_eval_knn', action='store_true', help='Do not evaluate k-nn accuracy', default=False)
    parser.add_argument('--stats_only', action='store_true',
                        help='Only compute the mean and std of the dataset for precomputed embeddings')

    main(parser.parse_args())


# python gen_embeds.py --arch PaSST --dataset DCASE2018_TASK5 --datapath data/DCASE_TASK5-PaSST --only_knn --no_eval_knn