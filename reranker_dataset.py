import json
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TripletDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_name='bert-base-uncased', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'query' in item and 'positive' in item and 'negative' in item:
                    self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        query = item['query']

        pos_doc = parse_media_doc(item['positive'])
        neg_doc = parse_media_doc(item['negative'])

        pos_text = format_document_full(pos_doc)
        neg_text = format_document_full(neg_doc)

        pos_inputs = self.tokenizer(f"QUERY: {query}\n{pos_text}", truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        neg_inputs = self.tokenizer(f"QUERY: {query}\n{neg_text}", truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids_pos': pos_inputs['input_ids'].squeeze(0),
            'attention_mask_pos': pos_inputs['attention_mask'].squeeze(0),
            'token_type_ids_pos': pos_inputs['token_type_ids'].squeeze(0),
            'input_ids_neg': neg_inputs['input_ids'].squeeze(0),
            'attention_mask_neg': neg_inputs['attention_mask'].squeeze(0),
            'token_type_ids_neg': neg_inputs['token_type_ids'].squeeze(0),

        }

    def print_sample(self, idx):
        item = self.samples[idx]
        query = item['query']
        pos_doc = parse_media_doc(item['positive'])
        neg_doc = parse_media_doc(item['negative'])
        pos_text = format_document_full(pos_doc)
        neg_text = format_document_full(neg_doc)

        print("[QUERY]\n" + query)
        print("\n[POSITIVE]\n" + pos_text)
        print("\n[NEGATIVE]\n" + neg_text)

    def print_decoded_inputs(self, idx):
        item = self.samples[idx]
        query = item['query']
        pos_doc = parse_media_doc(item['positive'])
        neg_doc = parse_media_doc(item['negative'])
        pos_text = format_document_full(pos_doc)
        neg_text = format_document_full(neg_doc)

        print("[RAW QUERY]\n", query)

        pos_input_ids = self.tokenizer(f"QUERY: {query}\n{pos_text}", truncation=True, padding='max_length', max_length=self.max_length)['input_ids']
        neg_input_ids = self.tokenizer(f"QUERY: {query}\n{neg_text}", truncation=True, padding='max_length', max_length=self.max_length)['input_ids']

        print("\n[DECODED POSITIVE]\n")
        print(self.tokenizer.decode(pos_input_ids, skip_special_tokens=True))
        print("\n[DECODED NEGATIVE]\n")
        print(self.tokenizer.decode(neg_input_ids, skip_special_tokens=True))

    def preview_batch(self, count=5):
        for i in range(min(count, len(self.samples))):
            print("=" * 60)
            print(f"Sample #{i}")
            self.print_sample(i)
            encoded = self[i]
            print("\n-- Token Counts --")
            print(f"POS: {encoded['input_ids_pos'].shape[0]} tokens")
            print(f"NEG: {encoded['input_ids_neg'].shape[0]} tokens")
            print("\n-- Decoded POSITIVE --")
            print(self.tokenizer.decode(encoded['input_ids_pos'], skip_special_tokens=True))
            print("\n-- Decoded NEGATIVE --")
            print(self.tokenizer.decode(encoded['input_ids_neg'], skip_special_tokens=True))


def parse_media_doc(doc_str: str) -> dict:
    fields = ["Title", "Genres", "Overview", "Tagline", "Director", "Stars", "Release Date", "Keywords", "Franchise"]
    doc = {}
    for field in fields:
        pattern = rf"{field}:\s*(.*?)(?=\n(?:{'|'.join(fields)}):|\Z)"
        match = re.search(pattern, doc_str, re.DOTALL)
        if match:
            doc[field] = match.group(1).strip()
    return doc

def format_document_full(doc: dict) -> str:
    title = doc.get("Title", "")
    genres = doc.get("Genres", "")
    overview = doc.get("Overview", "")[:600]
    tagline = doc.get("Tagline", "")
    director = doc.get("Director", "")
    stars = ', '.join(doc.get("Stars", "").split(', ')[:2])  # Trim to 2 actors
    release_year = doc.get("Release Date", "")[:4]
    kw_list = doc.get("Keywords", "").split(', ')
    keywords = ', '.join(kw_list[:30])

    lines = [
        f"Title: {title}",
        f"Genres: {genres}",
        f"Overview: {overview}",
        f"Tagline: {tagline}" if tagline else "",
        f"Director: {director}",
        f"Stars: {stars}",
        f"Release Year: {release_year}",
        f"Keywords: {keywords}",
    ]
    return "\n".join([l for l in lines if l.strip()])

def process_sample(raw_sample: dict, tokenizer=None) -> dict:
    query = raw_sample.get("query", "")
    pos_doc = parse_media_doc(raw_sample.get("positive", ""))
    neg_doc = parse_media_doc(raw_sample.get("negative", ""))

    pos_formatted = format_document_full(pos_doc)
    neg_formatted = format_document_full(neg_doc)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    pos_tokens = len(tokenizer(query, pos_formatted, truncation=False)["input_ids"])
    neg_tokens = len(tokenizer(query, neg_formatted, truncation=False)["input_ids"])

    return {
        "query": query,
        "positive_formatted": pos_formatted,
        "positive_token_count": pos_tokens,
        "negative_formatted": neg_formatted,
        "negative_token_count": neg_tokens
    }

""" TEST """

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
doc = {"query": "Heartfelt films that center around the bond between humans and their loyal animal companions, capturing the essence of love and loss.", "positive": "Title: Marley & Me\nGenres: Comedy, Family\nOverview: A newly married couple, in the process of starting a family, learn many of life's important lessons from their trouble-loving retriever, Marley. Packed with plenty of laughs to lighten the load, the film explores the highs and lows of marriage, maturity and confronting one's own mortality, as seen through the lens of family life with a dog.\nTagline: Heel the love.\nDirector: David Frankel\nStars: Owen Wilson, Jennifer Aniston, Eric Dane\nRelease Date: 2008-12-25\nKeywords: journalist, based on novel or book, affectation, grave, melancholy, puppy, love, dog, cynical, macabre, nostalgic, taunting, duringcreditsstinger, speculative, hard, angry, detached, paranoid, aggressive, pets, frantic, zealous, columnist, calm, candid, philosophical, unassuming, satirical, animal lead, desperate, lyrical, pet ownership, malicious, anxious, playful, dreary, cautionary, relaxed, clinical, inspirational, lighthearted, casual, grand, intimate, provocative, factual, didactic, negative, understated, absurd, critical, egotistical, sentimental, hilarious, romantic, whimsical, admiring, adoring, ambiguous, ambivalent, amused, antagonistic, apathetic, apologetic, appreciative, approving, arrogant, assertive, audacious, authoritarian, awestruck, baffled, bold, callous, celebratory, commanding, compassionate, complicated, defiant, demeaning, derogatory, disapproving, disdainful, disheartening, earnest, embarrassed, empathetic, enchant, enraged, enthusiastic, exhilarated, exuberant, familiar, farcical, frightened, frustrated, ghoulish, harsh, impartial, inflammatory, informative, joyful, matter of fact, mean spirited, melodramatic, pessimistic, sarcastic, sardonic, scathing, sceptical, skeptical, straightforward, sympathetic, tragic, vibrant", "negative": "Title: Behind Enemy Lines\nGenres: Action, Drama, Thriller, War\nOverview: While flying a routine reconnaissance mission over Bosnia, fighter pilot Lt. Chris Burnett photographs something he wasn't supposed to see and gets shot down behind enemy lines, where he must outrun an army led by a ruthless Serbian general. With time running out and a deadly tracker on his trail, Burnett's commanding officer, Admiral Reigart, decides to risk his career and launch a renegade rescue mission to save his life.\nTagline: His only weapon is his will to survive.\nDirector: John Moore\nStars: Owen Wilson, Gene Hackman, Gabriel Macht\nRelease Date: 2001-11-30\nKeywords: sniper, war crimes, bosnia and herzegovina, rescue mission, u.s. navy, based on true story, north atlantic treaty organization (nato), naval officer, bosnian war (1992-95), serbian, atrocity, combat, behind enemy lines, jet fighter pilot, mine field, hunted, jet crash, aircraft carrier, reconnaissance, mass grave, marines, enemy territory, naval flight officer, navy pilot, running for one's life, no fly zone", "source": "semantic"}
process_sample(doc)

dataset = TripletDataset("movie_train_30k_structured_text_0508.jsonl")

sample = dataset[0]
print(sample["input_ids_pos"].shape)  # torch.Size([512])
print(sample["input_ids_neg"].shape)  # torch.Size([512])

dataset.print_sample(998)
dataset.print_decoded_inputs(998)

dataset.preview_batch(count=10)