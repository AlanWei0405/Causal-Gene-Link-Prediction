import os
import torch
import torch_geometric
from torch_geometric.nn import MetaPath2Vec


def get_features(train_data):

    metapath = [
        ('gene', 'interact', 'gene'),
        ('gene', 'associate', 'disease'),
        ('disease', 'relate', 'disease'),
        ('disease', 'rev_associate', 'gene')
    ]

    for edge_type in metapath:
        if edge_type not in train_data.edge_index_dict:
            print(f"Edge type {edge_type} is missing in the graph.")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists("metapath2vec_model.pt"):
        model = load_metapath2vec_model("metapath2vec_model.pt", train_data.edge_index_dict, device)
    else:
        model = MetaPath2Vec(train_data.edge_index_dict, embedding_dim=128,
                             metapath=metapath, walk_length=25, context_size=7,
                             walks_per_node=3, num_negative_samples=1,
                             sparse=True).to(device)

        loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        for epoch in range(1, 5):
            train(model, optimizer, loader, device, epoch)
            # acc = test()
            # print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
        save_metapath2vec_model(model, "metapath2vec_model.pt")

    train_data['gene'].x = model('gene')
    train_data['disease'].x = model('disease')

    return train_data


def train(model, optimizer, loader, device, epoch, log_steps=100, eval_steps=2000):
    model.train()
    total_loss = 0

    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                  f'Loss: {total_loss / log_steps:.4f}')
            total_loss = 0

        # if (i + 1) % eval_steps == 0:
        #     acc = test()
        #     print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
        #           f'Acc: {acc:.4f}')


# @torch.no_grad()
# def test(model, hetero_data, device, train_ratio=0.1):
#     model.eval()
#
#     z = model('author', batch=hetero_data['author'].y_index.to(device))
#     y = hetero_data['author'].y
#
#     perm = torch.randperm(z.size(0))
#     train_perm = perm[:int(z.size(0) * train_ratio)]
#     test_perm = perm[int(z.size(0) * train_ratio):]
#
#     return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
#                       max_iter=150)

# Save the trained model
def save_metapath2vec_model(model, file_path):
    torch.save({
        'state_dict': model.state_dict(),  # Save model parameters
        'embedding_dim': model.embedding_dim,
        'metapath': model.metapath,
        'walk_length': model.walk_length,
        'context_size': model.context_size,
        'walks_per_node': model.walks_per_node,
        'num_negative_samples': model.num_negative_samples,
        'sparse': True
    }, file_path)

def load_metapath2vec_model(file_path, edge_index_dict, device):
    checkpoint = torch.load(file_path)
    model = MetaPath2Vec(
        edge_index_dict,
        embedding_dim=checkpoint['embedding_dim'],
        metapath=checkpoint['metapath'],
        walk_length=checkpoint['walk_length'],
        context_size=checkpoint['context_size'],
        walks_per_node=checkpoint['walks_per_node'],
        num_negative_samples=checkpoint['num_negative_samples'],
        sparse=checkpoint['sparse']
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model



