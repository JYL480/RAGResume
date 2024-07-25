import random
import torch
import numpy as np
import pandas as pd
from time import perf_counter as timer
from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import texts and embedding df
file_path_resume = "app/text_chunks_and_embeddings_df.csv"
# file_path_resume = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embedding_df = pd.read_csv(file_path_resume)

file_path_aboutme = "app/embeddings_aboutMe.csv"
# file_path_aboutme = "embeddings_aboutMe.csv"
embeddings_aboutMe = pd.read_csv(file_path_aboutme)

def read_embeddings() -> torch.Tensor:
    # Check if the first element in the embedding column is a string
    if isinstance(text_chunks_and_embedding_df["embedding"].iloc[0], str):
        # Convert embedding column back to np.array
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    if isinstance(embeddings_aboutMe["embedding"].iloc[0], str):
        # Convert embedding column back to np.array
        embeddings_aboutMe["embedding"] = embeddings_aboutMe["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    
    # Convert texts and embedding df to list of dicts
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    aboutMe_chunks = embeddings_aboutMe.to_dict(orient="records")

    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings_resume = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

    about_me_embeddings = torch.tensor(np.array(embeddings_aboutMe["embedding"].tolist()), dtype=torch.float32).to(device) 

    return embeddings_resume, pages_and_chunks, about_me_embeddings, aboutMe_chunks



# The get the relevant scores and indices of the source
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def retrieve_relevant_resources(query: str,
                                embeddings_resume: torch.tensor,
                                embeddings_aboutme: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                print_time: bool = True):

  query_embedding = model.encode(query, convert_to_tensor=True)
#   print(query_embedding.shape)
#   print(embeddings.shape)
  # Get the time to do the semantic search, which compares to our source PDF embeddings!
  start_time = timer()
  dot_scores = util.dot_score(a=query_embedding, b=embeddings_resume)[0] # The index zero is just to remove the outer list

  dot_scores_about_me = util.dot_score(a=query_embedding, b=embeddings_aboutme)[0]
  end_time = timer()
  time_taken = end_time-start_time

#   if print_time:
    # print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

  # Get the top k scores of the semantic search
  score_resume, indices_resume = torch.topk(dot_scores, 4)

  score_about_me, indices_about_me = torch.topk(dot_scores_about_me, 4)
  
  return score_resume, indices_resume, score_about_me, indices_about_me, time_taken

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What is your major?
Answer: I am currently studying Computer Engineering.
\nExample 2:
Query: What are your hobbies?
Answer: I like sports and music. More specifically, basketball and playing the French Horn!
\nExample 3:
Query: How do I contact you?
Answer: These are details I can provide you, email: [email@example.com], phone: [123-456-7890].
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    prompt = base_prompt.format(context=context, query=query)

    return prompt
