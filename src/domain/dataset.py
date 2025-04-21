import json
from pathlib import Path
from typing import cast

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2.functional import pil_to_tensor, to_grayscale, to_pil_image

from src.domain.receipt import Receipt
from src.domain.model import Model

# return image piexels, labels, business_card.xml
class Dataset(TorchDataset[tuple[Tensor, Tensor, str]]):
    def __init__(
        self,
        data: list[Receipt],
        model: Model,
        *,
        training: bool = True,
    ) -> None:
        self.data = data
        self.model = model
        self.training = training

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        receipt = self.data[index]

        image = Image.open(receipt.image_path)

        pixel_values = self._image_to_tensor(image, random_padding=self.training)
        # labels are converted from json to xml here
        labels = self._target_string_to_tensor(receipt.xml)

        return pixel_values, labels, receipt.xml

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def load(
        cls,
        path: Path,
        model: Model,
        *,
        training: bool = True,
    ) -> "Dataset":
        entities_path = path / "entities"
        labels_jsons = []
        for txt_file in entities_path.glob("*.txt"):
            with txt_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                labels_jsons.append({
                    "filename": txt_file.name,
                    "content": data
                })
        print(labels_jsons)
        return cls(
            [
                Receipt(
                    image_path = path / "img" / (labels_json["filename"].replace(".txt", ".jpg")),
                    company = labels_json["content"]["company"],
                    date = labels_json["content"]["date"],
                    address = labels_json["content"].get("address", ""),
                    total = labels_json["content"]["total"],
                )
                for labels_json in labels_jsons
            ],
            model,
            training=training,
        )

    def _gray_scaling_image(self, image: Image.Image) -> Image.Image:
        return to_pil_image(to_grayscale(pil_to_tensor(image)))

    def _image_to_tensor(self, image: Image.Image, *, random_padding: bool) -> Tensor:
        preprocess_image = self._gray_scaling_image(image)
        pixel_values = cast(
            Tensor,
            self.model.processor(
                preprocess_image.convert("RGB"),
                random_padding=random_padding,
                return_tensors="pt",
            ).pixel_values,
        )

        return pixel_values.squeeze()

    def _target_string_to_tensor(self, target: str) -> Tensor:
        ignore_id = -100
        input_ids = cast(
            Tensor,
            self.model.tokenizer(
                target,
                add_special_tokens=False,
                max_length=self.model.model.config.decoder.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
            ).input_ids,
        ).squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.model.tokenizer.pad_token_id] = ignore_id

        return labels
