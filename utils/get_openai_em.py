import openai
import os
import pickle
from tqdm import tqdm
import time
from openai import OpenAI


client = OpenAI()

def generate_embeddings_batch_openai(text_list, model="text-embedding-3-small", max_retries=5):
    embeddings = []
    for text in text_list:
        text = text.replace("\n", " ")
        for attempt in range(max_retries):
            try:
                embeddings.append(client.embeddings.create(input = [text], model=model).data[0].embedding)
            except openai.RateLimitError:
                wait_time = 2 ** attempt
                print(f"Rate limited. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Failed to get embedding for: {text[:30]}... Error: {e}")
                embeddings.append([0.0] * 1536)
                break
    return embeddings


def generate_embeddings_for_dataframe_openai(df, text_column, batch_size):
    embeddings = []
    texts = df[text_column].tolist()

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = generate_embeddings_batch_openai(batch_texts)
        embeddings.extend(batch_embeddings)

    return embeddings

def get_openai_embeddings(gene_sum, disease_sum):
    gene_path = './gene_embeddings_openai.pkl'
    disease_path = './disease_embeddings_openai.pkl'

    if os.path.exists(gene_path) and os.path.exists(disease_path):
        with open(gene_path, 'rb') as f:
            gene_sum = pickle.load(f)
        with open(disease_path, 'rb') as f:
            disease_sum = pickle.load(f)
    else:
        gene_sum["Embedding"] = generate_embeddings_for_dataframe_openai(
            gene_sum,
            text_column="Summary",
            batch_size=8
        )

        disease_sum["Embedding"] = generate_embeddings_for_dataframe_openai(
            disease_sum,
            text_column="Summary",
            batch_size=8
        )

        gene_sum.to_pickle(gene_path)
        disease_sum.to_pickle(disease_path)

    return gene_sum, disease_sum
