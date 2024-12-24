import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image
from transformers import AutoTokenizer, AutoModel

class MultimodalDataset(Dataset):
  def __init__(self, dataframe):
    # self.transform = transform
    self.dataframe = dataframe
    self.tokenizer = AutoTokenizer.from_pretrained('efederici/sentence-bert-base')
    self.bert_model = AutoModel.from_pretrained('efederici/sentence-bert-base')

  # Mean Pooling - Take attention mask into account for correct averaging
  def mean_pooling(self, model_output, attention_mask):
      token_embeddings = model_output[0] #First element of model_output contains all token embeddings
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  def __len__(self):
      return len(self.dataframe)

  def get_bert_embeddings(self, text):
    encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = self.bert_model(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

  def __getitem__(self, idx):

    x_title_batch = self.dataframe.iloc[idx]['clean_title'] # key_value
    x_img_batch = self.dataframe.iloc[idx]['clean_img_label']  # query
    y_batch = torch.tensor(self.dataframe.iloc[idx]['2_way_label'])

    title_embeddings = self.get_bert_embeddings(x_title_batch) 
    img_embeddings = self.get_bert_embeddings(x_img_batch) 

    return img_embeddings, title_embeddings, y_batch