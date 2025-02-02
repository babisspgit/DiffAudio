import os
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from transformers import ASTForAudioClassification, TrainingArguments, Trainer, ASTFeatureExtractor
import evaluate
import numpy as np
import accelerate
import pandas as pd
from datasets import Dataset, Audio, ClassLabel
from transformers import ASTForAudioClassification


# Load your CSV file into a pandas DataFrame
df = pd.read_csv('../data/raw/1000_test_class.csv')

# Verify column names
print(df.head())  # Ensure columns are named correctly (e.g., 'audio' and 'label')

# Rename columns if necessary
df.rename(columns={'image_path': 'audio', 'class': 'label'}, inplace=True)

# Ensure the labels are integers or strings
df['label'] = df['label'].astype(str)

# Create a Dataset from the DataFrame
dataset = Dataset.from_pandas(df)

# Cast the 'audio' and 'label' columns to their respective feature types
dataset = dataset.cast_column('audio', Audio(sampling_rate=44100))
dataset = dataset.cast_column('label', ClassLabel(names=list(df['label'].unique())))

# Verify the dataset structure
print(dataset)
print(dataset.features["label"])


# Convert labels to ClassLabel (ensures integer encoding)
unique_labels = sorted(df["label"].unique().tolist())  # Get sorted unique labels
dataset = dataset.cast_column("label", ClassLabel(names=unique_labels))

# Now split into train/validation sets
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']


# Load the feature extractor from a pretrained AST model
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593",
                                                         sampling_rate=44100,
                                                         num_mels_bins=512
                                                            )


audio_path = dataset[0]["audio"]  # The first entry’s audio path
print("First audio path:", audio_path)



import librosa
import numpy as np

def preprocess_function(batch):
    audio_array = batch["audio"]["array"]
    sampling_rate = batch["audio"]["sampling_rate"]

    # 🔹 Resample to 44.1 kHz if needed
    if sampling_rate != 44100:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=44100)

    # 🔹 Ensure audio is exactly 5 seconds long
    target_length = 5 * 44100  # 5 seconds * 44.1 kHz
    if len(audio_array) < target_length:
        audio_array = np.pad(audio_array, (0, target_length - len(audio_array)), mode="constant")
    else:
        audio_array = audio_array[:target_length]

    # 🔹 Convert audio to spectrogram using AST feature extractor
    inputs = feature_extractor(
        audio_array, 
        sampling_rate=44100, 
        return_attention_mask=True,
        return_tensors="np"
    )

    batch["input_values"] = inputs["input_values"][0]
    batch["attention_mask"] = inputs["attention_mask"][0] if "attention_mask" in inputs else None

    # 🔹 Ensure labels are cast to int64 (torch.long)
    batch["label"] = np.int64(batch["label"])  # ✅ Convert to int64# ✅ Ensure Labels are Scalars and Converted to int64
    #batch["label"] = np.int64(batch["label"]).item()  # ✅ Fixes "list" issue
    # transofrm label to type torch.LongTensor
    #batch["label"] = torch.LongTensor([batch["label"]])

    return batch

train_dataset = train_dataset.map(preprocess_function)
eval_dataset = eval_dataset.map(preprocess_function)


# 🔹 Check dataset labels before training
print("Sample label:", train_dataset[0]["label"])
print("Label type:", type(train_dataset[0]["label"]))  


# Retrieve the label list from your dataset
label_list = train_dataset.features["label"].names
num_labels = len(label_list)

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)


# 9. Define training args
training_args = TrainingArguments(
    output_dir="./ast-finetune-output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)


# 10. Metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)



# 11. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)



# 12. Train
trainer.train()


#-----------------------------------------


model.save_pretrained("./ast_finetuned")