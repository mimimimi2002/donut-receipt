import os
import torch
from datetime import datetime
from pathlib import Path
from typing import cast

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from src.domain.dataset import Dataset
from src.domain.model import Model

BASE_MODEL = "naver-clova-ix/donut-base"
TRAIN_PATH = Path("dataset/train")
VALIDATION_PATH = Path("dataset/test")
IMAGE_SIZE = (768, 1024)  # Must be multiples of 32

def train() -> None:
    batch_size = 1
    lr = 5e-5
    epoch_num = 10
    model_output_path = Path(
        f"model_output/donut_{batch_size}_{lr}_{round(datetime.now().timestamp())}"
    )

    config = VisionEncoderDecoderConfig.from_pretrained(BASE_MODEL)
    config.encoder.image_size = list(IMAGE_SIZE)
    print(config)

    base_model = cast(
        VisionEncoderDecoderModel,
        VisionEncoderDecoderModel.from_pretrained(
            BASE_MODEL,
            config=config,
        ),
    )

    processor = cast(DonutProcessor, DonutProcessor.from_pretrained(BASE_MODEL))
    processor.image_processor.size = {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]}
    if hasattr(processor, "feature_extractor"):
        processor.feature_extractor.size = {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]}

    model = Model(processor, base_model, lr, epoch_num)

    training_dataset = Dataset.load(TRAIN_PATH, model, training=True)
    validation_dataset = Dataset.load(VALIDATION_PATH, model, training=False)

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
    )

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
    )

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=epoch_num,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        precision=32,  # Use full precision (fp32)
        #precision="16-mixed",
        num_sanity_val_steps=0,
        accumulate_grad_batches=2,
        callbacks=[
            ModelCheckpoint(
                dirpath=model_output_path,
                filename="every_{epoch}_{v_num}",
                every_n_epochs=1,
                save_top_k=-1,
            ),
        ],
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    model.model.save_pretrained(model_output_path)
    model.processor.save_pretrained(model_output_path)


if __name__ == "__main__":
    train()

