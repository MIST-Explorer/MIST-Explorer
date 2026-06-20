from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ImageMetadata:
    uuid: str
    name: str
    channel_count: int
    original_filename: str = ""
    contrast_settings: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    channel_display_names: Dict[str, str] = field(default_factory=dict)
    channel_cmaps: Dict[str, str] = field(default_factory=dict)
    default_channel: str = "Channel 1"

    def to_dict(self) -> dict:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "channel_count": self.channel_count,
            "original_filename": self.original_filename,
            "contrast_settings": self.contrast_settings,
            "channel_display_names": self.channel_display_names,
            "channel_cmaps": self.channel_cmaps,
            "default_channel": self.default_channel,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageMetadata":
        return cls(
            uuid=data["uuid"],
            name=data["name"],
            channel_count=data["channel_count"],
            original_filename=data.get("original_filename", ""),
            contrast_settings=data.get("contrast_settings", {}),
            channel_display_names=data.get("channel_display_names", {}),
            channel_cmaps=data.get("channel_cmaps", {}),
            default_channel=data.get("default_channel", "Channel 1"),
        )


@dataclass
class ProjectMetadata:
    name: str
    path: Path
    created_at: datetime
    modified_at: datetime
    is_temporary: bool = False
    images: List[ImageMetadata] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "is_temporary": self.is_temporary,
            "images": [img.to_dict() for img in self.images],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectMetadata":
        return cls(
            name=data["name"],
            path=Path(data["path"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            is_temporary=data.get("is_temporary", False),
            images=[ImageMetadata.from_dict(img) for img in data.get("images", [])],
        )

    def add_image(self, image: ImageMetadata):
        for img in self.images:
            if img.uuid == image.uuid:
                return
        self.images.append(image)
        self.modified_at = datetime.now()

    def remove_image(self, uuid: str):
        self.images = [img for img in self.images if img.uuid != uuid]
        self.modified_at = datetime.now()

    def get_image(self, uuid: str) -> Optional[ImageMetadata]:
        for img in self.images:
            if img.uuid == uuid:
                return img
        return None

    def update_image(self, image: ImageMetadata):
        for i, img in enumerate(self.images):
            if img.uuid == image.uuid:
                self.images[i] = image
                self.modified_at = datetime.now()
                break
