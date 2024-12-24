import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd
import urllib
from PIL import Image
import argparse

# Filter out rows where the image URL fails to load
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

# Ref https://github.com/Garima13a/YOLO-Object-Detection
from yolo.darknet import Darknet
from yolo.utils import *

from data.dataset import MultimodalDataset
from torch.utils.data import DataLoader
from model.model import CrossAttentionModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}!.")

def add_image_labels(df_valid):
    # Set the location and name of the cfg file
    cfg_file = './yolo/yolov3.cfg'

    # Set the location and name of the pre-trained weights file
    weight_file = './yolo/yolov3.weights'

    # Set the location and name of the COCO object classes file
    namesfile = './yolo/coco.names'

    # Load the network architecture
    m = Darknet(cfg_file)

    # Load the pre-trained weights
    m.load_weights(weight_file)

    # Load the COCO object classes
    class_names = load_class_names(namesfile)
    # Print the neural network used in YOLOv3
    # m.print_network()

    nms_thresh = 0.6 # Set the NMS threshold
    iou_thresh = 0.4 # Set the IOU threshold

    # concat_objectlables
    def concat_labels(boxes):
        labels_str = []
        for i in range(len(boxes)):
            box = boxes[i]
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                labels_str.append(class_names[cls_id])
        return " ".join(labels_str)

    def extract_obj_labels(img_url):
        req = urllib.request.urlopen(img_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1) # 'Load it as it is'
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # We resize the image to the input width and height of the first layer of the network.
        resized_image = cv2.resize(original_image, (m.width, m.height))

        # Detect objects in the image
        boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
        img_labels_str = concat_labels(boxes)
        return img_labels_str

    df_valid['clean_img_label'] = df_valid['image_url'].apply(extract_obj_labels)

    return df_valid

def preprocess_data(dataset_tsv):
    # Function to strip query parameters from URL
    def clean_url(img_url):
        parsed_url = urlparse(img_url)
        return parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path

    # Function to check and load the image from URL
    def load_image_from_url(img_url):
        try:
            # Strip query parameters before attempting to load the image
            clean_img_url = clean_url(img_url)
            # Try opening the URL
            req = urllib.request.urlopen(clean_img_url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)  # 'Load it as it is'
            if img is not None:
                original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return True  # Image loaded successfully
            else:
                return False  # Failed to load image
            
        except (HTTPError, URLError) as e:
            print(f"Error fetching {img_url}: {e}")
            return False  # Image failed to load
        except Exception as e:
            print(f"Unexpected error with {img_url}: {e}")
            return False  # Any other exception
    
    # read data
    df = pd.read_csv(dataset_tsv, sep='\t')
    df = df.replace(np.nan, '', regex=True)
    df.fillna('', inplace=True)
    df = df[df['hasImage']==True]
    df = df[df['image_url'] != '']
    df = df[["clean_title", "hasImage", "image_url", "2_way_label"]]
    print(len(df))

    valid_urls = []  # List to store valid URLs
    for idx, row in df.iterrows():
        img_url = row['image_url']
        if load_image_from_url(img_url):
            valid_urls.append(img_url)  # Keep valid URLs

    # Update the DataFrame with only valid URLs
    df_valid = df[df['image_url'].isin(valid_urls)]
    print(len(df_valid))

    df_updated = add_image_labels(df_valid)
    print(len(df_updated))
    
    return df_updated

def train_model(args, model, optimizer, criterion, train_dataloader):
    # training loop using dataloader
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_dataloader):
            query, key_value, y_batch = data

            output, attention_weights = model(query, key_value)
            output_sq = output.squeeze(1)

            loss = criterion(output_sq, y_batch)

            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            running_loss += loss.item()

        # Print epoch statistics
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / len(train_dataloader):.4f}")
    print("Training complete!")

def eval_model(model, test_dataloader):
    # testing loop using dataloader
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(test_dataloader):
            query, key_value, y_batch = data

            output, attention_weights = model(query, key_value)
            output_sq = output.squeeze(1)
            predictions = torch.argmax(output_sq, dim=1)
            all_labels.extend(y_batch)
            all_predictions.extend(predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="binary")
    recall = recall_score(all_labels, all_predictions, average="binary")
    f1_score = f1_score(all_labels, all_predictions, average="binary")

    return accuracy, precision, recall, f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_tsv',
        type=str,
        default='.\data\multimodal_train.tsv',
        required=True,
        hint="dataset tsv file path"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        hint="number of training epochs"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        hint="batch size"
    )
    parser.add_argument(
        '--num_heads',
        type=int,
        default=8,
        hint="number of attention heads"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        hint="learning rate"
    )

    args = parser.parse_args()

    df = preprocess_data(args.dataset_tsv)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # initialize dataset
    train_data = MultimodalDataset(train_df)
    test_data = MultimodalDataset(test_df)

    # Iterating over Dataset 
    # for i in range(1, 5):
    #     query, key_value, y_batch = train_data[i]
    #     print(f"Query shape: {query.shape}")
    #     print(f"Key Value shape: {key_value.shape}")
    #     print(f"Y batch shape: {y_batch.shape}")

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Iterating over dataloader
    # query, key_value, y_batch = next(iter(train_dataloader))
    # print(f"Query shape: {query.shape}")
    # print(f"Key_Value Shape: {key_value.shape}")
    # print(f"Y Batch shape: {y_batch.shape}") 


    # Note: sentence bert gives embedding of size: [1, 768] 
    model = CrossAttentionModel(embed_dim=768, num_heads=args.num_heads)
    criterion = nn.CrossEntropyLoss()    # Combines softmax and cross-entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train the model
    train_model(args, model, optimizer, criterion, train_dataloader)

    # evaluate the model
    accuracy, precision, recall, f1_score = eval_model(model, test_dataloader)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")


if "__name__" == "__main__":
    main()