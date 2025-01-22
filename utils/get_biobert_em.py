import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def generate_embeddings_batch(text_list, tokenizer, model, device):
    # Tokenize the batch of text
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,  # Pads to the longest sequence in the batch
        truncation=True,  # Truncates inputs to the model's maximum length
        max_length=512
    ).to(device)  # Move inputs to the specified device (CPU or GPU)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the `[CLS]` token embeddings from the last hidden state
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

    # Move embeddings back to CPU (if necessary) and convert to NumPy
    return cls_embeddings.cpu().numpy()


def generate_embeddings_for_dataframe(df, text_column, tokenizer, model, device, batch_size):

    embeddings = []
    texts = df[text_column].tolist()

    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = generate_embeddings_batch(batch_texts, tokenizer, model, device)
        embeddings.extend(batch_embeddings)

    return embeddings

def get_biobert_em(gene_sum, disease_sum):

    gene_sum_path = './gene_embeddings.pkl'
    disease_sum_path = './disease_embeddings.pkl'

    if (os.path.exists(gene_sum_path) and os.path.exists(disease_sum_path)):
        with open(gene_sum_path, 'rb') as f:
            gene_sum = pickle.load(f)

        with open(disease_sum_path, 'rb') as f:
            disease_sum = pickle.load(f)

    else:
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Generate embeddings for gene summaries
        gene_sum["Embedding"] = generate_embeddings_for_dataframe(
            gene_sum,
            text_column="Summary",
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=64
        )

        # Generate embeddings for disease summaries
        disease_sum["Embedding"] = generate_embeddings_for_dataframe(
            disease_sum,
            text_column="Summary",
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=64
        )

        gene_sum.to_pickle("gene_embeddings.pkl")
        disease_sum.to_pickle("disease_embeddings.pkl")

    return gene_sum, disease_sum




