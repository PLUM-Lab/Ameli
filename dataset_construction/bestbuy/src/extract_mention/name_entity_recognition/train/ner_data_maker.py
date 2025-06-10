
import re

from numpy import pad
def get_tokens_with_entities(raw_text: str):
    raw_tokens = re.split(r"\s(?![^\[]*\])", raw_text)
    entity_value_pattern = r"\[(?P<value>.+?)\]\((?P<entity>.+?)\)"
    entity_value_pattern_compiled = re.compile(entity_value_pattern, flags=re.I|re.M)

    tokens_with_entities = []

    for raw_token in raw_tokens:
        match = entity_value_pattern_compiled.match(raw_token)
        if match:
            raw_entity_name, raw_entity_value = match.group("entity"), match.group("value")

            for i, raw_entity_token in enumerate(re.split("\s", raw_entity_value)):
                entity_prefix = "B" if i == 0 else "I"
                entity_name = f"{entity_prefix}-{raw_entity_name}"
                tokens_with_entities.append((raw_entity_token, entity_name))
        else:
            tokens_with_entities.append((raw_token, "O"))

    return tokens_with_entities


class NERDataMaker:
    def __init__(self, texts):
        self.unique_entities = []
        self.processed_texts = []

        temp_processed_texts = []
        for text in texts:
            tokens_with_entities = get_tokens_with_entities(text)
            for _, ent in tokens_with_entities:
                if ent not in self.unique_entities:
                    self.unique_entities.append(ent)
            temp_processed_texts.append(tokens_with_entities)

        self.unique_entities.sort(key=lambda ent: ent if ent != "O" else "")

        for tokens_with_entities in temp_processed_texts:
            self.processed_texts.append([(t, self.unique_entities.index(ent)) for t, ent in tokens_with_entities])

    @property
    def id2label(self):
        return dict(enumerate(self.unique_entities))

    @property
    def label2id(self):
        return {v:k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        def _process_tokens_for_one_text(id, tokens_with_encoded_entities):
            ner_tags = []
            tokens = []
            for t, ent in tokens_with_encoded_entities:
                ner_tags.append(ent)
                tokens.append(t)

            return {
                "id": id,
                "ner_tags": ner_tags,
                "tokens": tokens
            }

        tokens_with_encoded_entities = self.processed_texts[idx]
        if isinstance(idx, int):
            return _process_tokens_for_one_text(idx, tokens_with_encoded_entities)
        else:
            return [_process_tokens_for_one_text(i+idx.start, tee) for i, tee in enumerate(tokens_with_encoded_entities)]

    def as_hf_dataset(self, tokenizer):
        from datasets import Dataset, Features, Value, ClassLabel, Sequence
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        ids, ner_tags, tokens = [], [], []
        for i, pt in enumerate(self.processed_texts):
            ids.append(i)
            pt_tokens,pt_tags = list(zip(*pt))
            ner_tags.append(pt_tags)
            tokens.append(pt_tokens)
        data = {
            "id": ids,
            "ner_tags": ner_tags,
            "tokens": tokens
        }
        features = Features({
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=self.unique_entities)),
            "id": Value("int32")
        })
        ds = Dataset.from_dict(data, features)
        tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
        return tokenized_ds

# usage
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# dm = NERDataMaker(["I come from [Kathmanduu valley,](location) [Nepal](location)"])  
# test_dm.as_hf_dataset(tokenizer=tokenizer)