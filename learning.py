# -*- coding: utf-8 -*-

# 개발 환경: Windows 11 (또는 사용 중인 OS로 수정)
# 라이브러리 버전 (pip list로 확인 후 실제 버전으로 업데이트하세요):
# - torch: 2.0.1
# - transformers: 4.35.0
# - pandas: 2.1.0
# - numpy: 1.26.0
# - torchvision: 0.15.2
# - PIL (Pillow): 10.0.0
# - tqdm: 4.66.1

import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# transformers import 오류 방지 (라이브러리 로딩 포함)
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("transformers 라이브러리가 설치되지 않았습니다. pip install transformers를 실행하세요.")

# 시드 고정으로 재현성 높임
torch.manual_seed(42)
np.random.seed(42)

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드
MODEL_NAME = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# 기본 경로 설정 (상대 경로로 변경)
BASE_PATH = "."
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")  # 훈련 데이터 경로 추가

# 이미지 경로 생성 함수 (상대 경로 적용)
def get_image_path(base_path, img_path):
    img_path = img_path.lstrip('./').lstrip('.\\').replace('\\', '/')
    if 'test_input_images/' in img_path:
        img_path = img_path.split('test_input_images/', 1)[1]
        full_path = os.path.join(base_path, 'test_input_images', img_path)
    elif 'train_input_images/' in img_path:
        img_path = img_path.split('train_input_images/', 1)[1]
        full_path = os.path.join(base_path, 'train_input_images', img_path)
    else:
        full_path = os.path.join(base_path, 'test_input_images', img_path)
    return os.path.normpath(full_path)

# 데이터셋 클래스 (전처리 수정: ToTensor와 Normalize 제거, processor가 내부적으로 처리하도록 함)
class VQADataset(Dataset):
    def __init__(self, df, processor, tta_transforms=None, is_train=False):
        self.df = df
        self.processor = processor
        self.is_train = is_train
        self.tta_transforms = tta_transforms or transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1),
            # ToTensor와 Normalize 제거: CLIPProcessor가 내부적으로 처리
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = get_image_path(BASE_PATH, row["img_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224))
        image = self.tta_transforms(image)  # PIL 이미지 상태로 유지

        question = row["Question"]
        choices = [row["A"], row["B"], row["C"], row["D"]]
        item = {"image": image, "question": question, "choices": choices}
        if self.is_train:
            answer = row["answer"]  # 훈련 시 정답 레이블 추가
            pred_index = ord(answer) - ord('A')
            item["label"] = pred_index
        return item

# 커스텀 collate_fn (프롬프트 최적화)
def collate_fn(batch):
    if not batch:
        return None
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    choices = [item["choices"] for item in batch]

    # 프롬프트 강화: 대회 취지에 맞게 일상 사진 이해 강조
    prompts = [f"A daily life photo: {q}. The correct answer is: {c}" for q, cs in zip(questions, choices) for c in cs]

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77)
    if "label" in batch[0]:
        labels = torch.tensor([item["label"] for item in batch])
        inputs["labels"] = labels
    return inputs

# Fine-tuning 함수 (머신러닝 적용: 훈련 데이터로 모델 학습)
def fine_tune_model(train_df, epochs=3, batch_size=8, lr=1e-5):
    dataset = VQADataset(train_df, processor, is_train=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"훈련 중 (에폭 {epoch+1})"):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(DEVICE)
            with autocast():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image  # (batch_size, batch_size * 4)
                batch_size = logits_per_image.shape[0]
                num_choices = 4
                train_logits = []
                for i in range(batch_size):
                    start_idx = i * num_choices
                    end_idx = start_idx + num_choices
                    train_logits.append(logits_per_image[i, start_idx:end_idx])
                train_logits = torch.stack(train_logits)  # (batch_size, 4)
                loss = loss_fn(train_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"에폭 {epoch+1} 평균 손실: {total_loss / len(train_loader)}")
    model.save_pretrained(os.path.join(BASE_PATH, "fine_tuned_clip"))  # 모델 저장
    print("Fine-tuning 완료")

# 훈련 데이터 로딩 및 fine-tuning 실행 (대회 취지에 맞게 학습)
train_df = pd.read_csv(TRAIN_CSV_PATH)
fine_tune_model(train_df)
