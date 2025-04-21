from dataclasses import dataclass


@dataclass
class Receipt:
    image_path: str
    company: str
    date: str
    address: str
    total: str

    @property
    def xml(self) -> str:
        return (
            "<s>"
            f"<s_company>{self.company}</s_company>"
            f"<s_date>{self.date}</s_date>"
            f"<s_address>{self.address}</s_address>"
            f"<s_total>{self.total}</s_total>"
            "</s>"
        )

    @classmethod
    def get_xml_tags(cls) -> list[str]:
        return [
            "<s>",
            "<s_company>",
            "</s_company>",
            "<s_date>",
            "</s_date>",
            "<s_address>",
            "</s_address>",
            "<s_total>",
            "</s_total>"
            "</s>",
        ]
