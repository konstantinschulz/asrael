from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base


class Sentence:
    def __init__(self, id: str, content: str):
        self.id: str = id
        self.content: str = content


class SentenceRelation:
    def __init__(self, id: str, content: str = "", token_list: list[str] = None, token_set: set[str] = None,
                 related: dict = None):  # dict[str, Sentence]
        self.id: str = id
        self.content: str = content
        self.related: dict = dict() if related is None else related  # dict[str, Sentence]
        self.token_list: list[str] = token_list
        self.token_set: set[str] = token_set

    def to_sentence(self):  # -> Sentence
        return Sentence(self.id, self.content)


Base = declarative_base()


class Exclusion(Base):
    __tablename__ = "exclusion"

    id = Column(String, primary_key=True)


class SentencePair:
    def __init__(self, sr_1: SentenceRelation, sr_2: SentenceRelation, id: str = "", is_match: bool = False):
        self.sr_1: SentenceRelation = sr_1
        self.sr_2: SentenceRelation = sr_2
        self.id: str = id
        self.is_match: bool = is_match
