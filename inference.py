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

# 기본 경로 설정 (상대 경로로 변경)
BASE_PATH = "."
TEST_CSV_PATH = os.path.join(BASE_PATH, "test.csv")
OUTPUT_CSV_PATH = os.path.join(BASE_PATH, "submission.csv")

# 모델 로드 (학습된 가중치 로드 - 제출 시 fine_tuned_clip 폴더 포함 필수)
model = CLIPModel.from_pretrained(os.path.join(BASE_PATH, "fine_tuned_clip")).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")  # 프로세서는 원본 모델에서 로드

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

# 데이터셋 클래스 (추론용)
class VQADataset(Dataset):
    def __init__(self, df, processor, tta_transforms=None, is_train=False):
        self.df = df
        self.processor = processor
        self.is_train = is_train
        self.tta_transforms = tta_transforms or transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1),
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
        image = self.tta_transforms(image)

        question = row["Question"]
        choices = [row["A"], row["B"], row["C"], row["D"]]
        item = {"image": image, "question": question, "choices": choices}
        return item

# 커스텀 collate_fn (프롬프트 최적화)
def collate_fn(batch):
    if not batch:
        return None
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    choices = [item["choices"] for item in batch]

    prompts = [f"A daily life photo: {q}. The correct answer is: {c}" for q, cs in zip(questions, choices) for c in cs]

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77)
    return inputs

# 테스트셋 로딩
df = pd.read_csv(TEST_CSV_PATH)

# TTA 설정: 반복 횟수 2로 안정화
tta_transforms_list = [
    transforms.Compose([transforms.RandomRotation(5), transforms.ColorJitter(brightness=0.1)]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
]
num_tta = len(tta_transforms_list)

# 예측 루프
preds = []
num_choices = 4
temperature = 0.8

for tta_idx in range(num_tta):
    print(f"TTA 반복 {tta_idx + 1}/{num_tta}")
    dataset = VQADataset(df, processor, tta_transforms=tta_transforms_list[tta_idx])
    test_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    tta_probs = []
    for batch in tqdm(test_loader, desc="예측 중"):
        if batch is None or 'input_ids' not in batch:
            print(f"잘못된 배치: {batch}")
            continue
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad(), autocast():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image / temperature
            
            batch_size = logits_per_image.shape[0]
            for i in range(batch_size):
                start_idx = i * num_choices
                end_idx = start_idx + num_choices
                logits = logits_per_image[i, start_idx:end_idx]
                probs = logits.softmax(dim=0).cpu().numpy()
                tta_probs.append(probs)
    
    if tta_idx == 0:
        avg_probs = np.array(tta_probs)
    else:
        avg_probs += np.array(tta_probs)

# 평균 확률로 최종 예측
avg_probs /= num_tta
for prob in avg_probs:
    pred_index = np.argmax(prob)
    preds.append(chr(ord('A') + pred_index))

# 제출 파일 생성
submission = pd.DataFrame({
    "ID": df["ID"],
    "answer": preds
})
submission.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

print(f"\n✅ 제출 파일 저장 완료: {OUTPUT_CSV_PATH}")
