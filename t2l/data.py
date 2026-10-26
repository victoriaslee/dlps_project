"""Datasets, prompts, description banks, and LoRA weight I/O."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def _hf_cache_dir() -> str:
    """Keep caches from incompatible `datasets` versions separate."""
    import datasets

    version = datasets.__version__.replace(".", "_")
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "hf_cache" / version
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


#prompt templates
TEMPLATES = {
    "boolq": (
        "{passage}\n\nQuestion: {question}?\n\n"
        "Please answer with either 'true' or 'false' without any explanation."
    ),
    "arc_easy": (
        "Answer the question below by choosing the correct choice.\n\n"
        "{question}\n\n{choices}\n\n"
        "You must respond with the letter corresponding to the correct choice "
        "without any explanation."
    ),
    "arc_challenge": (
        "Answer the question below by choosing the correct choice.\n\n"
        "{question}\n\n{choices}\n\n"
        "You must respond with the letter corresponding to the correct choice "
        "without any explanation."
    ),
    "piqa": (
        "Choose the option that answers the question or solves the problem.\n\n"
        "{goal}\n\n0: {sol1}\n1: {sol2}\n\n"
        "Respond with either '0' or '1' without any explanation."
    ),
    "winogrande": (
        "Given the following situation:\n\n{sentence}\n\nWhich option is correct?\n\n"
        "Option 1: {option1}\n\nOption 2: {option2}\n\n"
        "Respond with either '1' or '2' without any explanation."
    ),
    "openbookqa": (
        "Complete the following passage or answer the question by choosing "
        "the correct choice.\n\n{question_stem}\n\n{choices}\n\n"
        "Respond with the letter (A,B,C,D) without any explanation."
    ),
    "hellaswag": (
        "Choose the correct ending for the passage. Endings are labelled 0,1,2,3.\n\n"
        "Passage: {ctx}\n\n"
        "0: {e0}\n1: {e1}\n2: {e2}\n3: {e3}\n\n"
        "Respond with only the number 0,1,2, or 3."
    ),
    "gsm8k": "{question}\n\nLet's think step by step.",
    "sni": "{definition}\n\nInput: {input}\n\nOutput:",
}

# default natural-language task descriptions (generated w ChatGPT following paper's procedure)
DEFAULT_DESCRIPTIONS: Dict[str, List[str]] = {
    "boolq": [
        "Answer yes/no questions about a passage with true or false.",
        "Given a paragraph, decide if a claim is true or false.",
        "Boolean question answering over short passages.",
        "Read the text and answer the question with true or false only.",
        "Verify factual claims against a provided passage.",
        "Binary reading-comprehension: true if supported, else false.",
        "Passage-based boolean QA without explanations.",
        "Determine whether the answer to the question is true or false.",
    ],
    "arc_easy": [
        "Answer easy science multiple-choice questions with a single letter.",
        "Grade-school science quiz: pick A, B, C, or D.",
        "Choose the correct science fact from four options.",
        "Elementary science reasoning with multiple choices.",
        "Select the right answer letter for ARC-Easy items.",
        "Science knowledge questions with letter-only answers.",
        "Multiple-choice science: respond with the correct choice letter.",
        "Answer ARC Easy exam questions concisely.",
    ],
    "arc_challenge": [
        "Hard science multiple-choice: answer with one letter.",
        "Challenging science QA with A/B/C/D responses.",
        "ARC-Challenge style science reasoning, letter only.",
        "Difficult science exam questions, choose the best option.",
        "Advanced science multiple-choice without explanations.",
        "Pick the correct letter for hard science questions.",
        "Science challenge set: letter answer only.",
        "Answer difficult ARC questions with a single choice.",
    ],
    "piqa": [
        "Physical commonsense: choose solution 0 or 1.",
        "Pick the more sensible way to achieve a physical goal.",
        "PIQA-style physical reasoning with binary options.",
        "Choose the practical solution for everyday physical tasks.",
        "Commonsense physics: respond 0 or 1.",
        "Select the better procedure for a physical goal.",
        "Binary physical QA without explanations.",
        "Decide which solution makes physical sense.",
    ],
    "winogrande": [
        "Winograd-style pronoun resolution: choose option 1 or 2.",
        "Fill the blank with the correct entity (1 or 2).",
        "Commonsense coreference: answer 1 or 2.",
        "Resolve ambiguous pronouns by picking the right option.",
        "Winogrande binary choice without explanation.",
        "Choose which noun the blank refers to.",
        "Sentence completion with two candidate fillers.",
        "Pick option 1 or 2 for the Winograd schema.",
    ],
    "openbookqa": [
        "Open-book science QA: answer with A, B, C, or D.",
        "Science facts multiple-choice with letter responses.",
        "OpenBookQA: choose the correct science answer letter.",
        "Elementary science with open-book style facts.",
        "Select A/B/C/D for science comprehension items.",
        "Letter-only answers for open-book science questions.",
        "Science MCQ grounded in basic facts.",
        "Answer OpenBookQA items with a single letter.",
    ],
    "hellaswag": [
        "Choose the most plausible continuation (0-3) of a passage.",
        "HellaSwag sentence completion with numbered endings.",
        "Commonsense ending selection: respond 0, 1, 2, or 3.",
        "Pick the natural next event for the given context.",
        "Activity continuation multiple-choice.",
        "Select the coherent ending index for the passage.",
        "Everyday situation completion, number only.",
        "HellaSwag-style ending prediction without explanation.",
    ],
    "gsm8k": [
        "Solve grade-school math word problems with step-by-step reasoning.",
        "GSM8K arithmetic reasoning; show work then give the answer.",
        "Multi-step math word problems for elementary students.",
        "Compute the numeric answer to a math story problem.",
        "Chain-of-thought math QA ending with a final number.",
        "Grade-school math with intermediate reasoning steps.",
        "Solve math word problems carefully and accurately.",
        "Elementary math reasoning with a clear final answer.",
    ],
}

#curated English SNI task ids (optional via T2L_DOWNLOAD_SNI=1)
SNI_TRAIN_CANDIDATES = [
    "task280_stereoset_classification_stereotype_type",
    "task290_tellmewhy_question_answerability",
    "task1344_glue_entailment_classification",
    "task1389_hellaswag_completion",
    "task1502_commongen_sentence_generation",
    "task1516_tomqa_find_location_easy_clean",
    "task155_count_nouns_verbs",
    "task156_codah_classification_adversarial",
    "task1612_sick_label_classification",
    "task1615_sick_relatedness_score",
    "task190_snli_classification",
    "task199_mnli_classification",
    "task200_robustlr_entailment_classification",
    "task201_mnli_neutrality_classification",
    "task202_mnli_contradiction_classification",
    "task279_storycloze_incorrect_ending_classification",
    "task284_imdb_classification",
    "task285_imdb_answer_generation",
    "task296_storycloze_correct_end_classification",
    "task317_hybridqa_answer_generation",
    "task322_jigsaw_classification_toxic",
    "task325_jigsaw_classification_threat",
    "task327_jigsaw_classification_obscene",
    "task329_gap_classification",
    "task333_hatexplain_classification",
    "task335_hatexplain_answer_generation",
    "task346_hybridqa_classification",
    "task349_squad2.0_answerable_unanswerable_question_classification",
    "task391_causal_relationship",
    "task392_inverse_causal_relationship",
    "task488_extract_country_of_nationality",
    "task520_aquamuse_answer_given_in_passage",
    "task564_discofuse_classification",
    "task580_socialiqa_answer_generation",
    "task605_find_the_longest_common_subsequence",
    "task614_glucose_cause_event_detection",
]

SNI_HOLDOUT = [
    "task035_winogrande_question_modification_object",
    "task039_qasc_find_overlapping_words",
    "task304_numeric_fused_head_resolution",
    "task362_spoc_code_comment_generation",
]

# Diverse, non-overlapping stand-ins for an SNI multi-task mix.  These tasks
# deliberately cover textual labels, booleans, letters, and numeric choices.
REAL_SFT_TASKS = [
    {"key": "imdb_sentiment", "path": "imdb", "config": None, "kind": "classify",
     "fields": ["text"], "labels": ["negative", "positive"],
     "definition": "Classify movie review sentiment as negative or positive."},
    {"key": "ag_news", "path": "ag_news", "config": None, "kind": "classify",
     "fields": ["text"], "labels": ["World", "Sports", "Business", "Science and technology"],
     "definition": "Classify a news article as World, Sports, Business, or Science and technology."},
    {"key": "sst2", "path": "glue", "config": "sst2", "kind": "classify",
     "fields": ["sentence"], "labels": ["negative", "positive"],
     "definition": "Classify sentence sentiment as negative or positive."},
    {"key": "emotion", "path": "emotion", "config": None, "kind": "classify",
     "fields": ["text"], "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
     "definition": "Classify the emotion as sadness, joy, love, anger, fear, or surprise."},
    {"key": "yelp_polarity", "path": "yelp_polarity", "config": None, "kind": "classify",
     "fields": ["text"], "labels": ["negative", "positive"],
     "definition": "Classify a Yelp review as negative or positive."},
    {"key": "dbpedia", "path": "dbpedia_14", "config": None, "kind": "classify",
     "fields": ["content"], "labels": [
         "Company", "Educational institution", "Artist", "Athlete", "Office holder",
         "Mean of transportation", "Building", "Natural place", "Village", "Animal",
         "Plant", "Album", "Film", "Written work",
     ], "definition": "Classify Wikipedia content into its ontology category."},
    {"key": "amazon_polarity", "path": "amazon_polarity", "config": None, "kind": "classify",
     "fields": ["title", "content"], "labels": ["negative", "positive"],
     "definition": "Classify an Amazon review as negative or positive."},
    {"key": "tweet_sentiment", "path": "tweet_eval", "config": "sentiment", "kind": "classify",
     "fields": ["text"], "labels": ["negative", "neutral", "positive"],
     "definition": "Classify tweet sentiment as negative, neutral, or positive."},
    {"key": "mnli", "path": "glue", "config": "mnli", "kind": "classify",
     "fields": ["premise", "hypothesis"], "labels": ["entailment", "neutral", "contradiction"],
     "definition": "Classify the relation between a premise and hypothesis."},
    {"key": "qnli_boolean", "path": "glue", "config": "qnli", "kind": "qnli",
     "definition": "Decide whether the sentence answers the question; return true or false."},
    {"key": "mrpc_boolean", "path": "glue", "config": "mrpc", "kind": "mrpc",
     "definition": "Decide whether two sentences are paraphrases; return true or false."},
    {"key": "rte", "path": "glue", "config": "rte", "kind": "classify",
     "fields": ["sentence1", "sentence2"], "labels": ["entailment", "not entailment"],
     "definition": "Decide whether the first sentence entails the second."},
    {"key": "sciq", "path": "allenai/sciq", "config": None, "kind": "sciq",
     "definition": "Answer a science multiple-choice question with A, B, C, or D."},
    {"key": "commonsense_qa", "path": "tau/commonsense_qa", "config": None, "kind": "choices",
     "definition": "Answer a commonsense multiple-choice question with its choice letter."},
    {"key": "qasc", "path": "allenai/qasc", "config": None, "kind": "choices",
     "definition": "Answer a science multiple-choice question with its choice letter."},
    {"key": "hellaswag", "path": "Rowan/hellaswag", "config": None, "kind": "hellaswag",
     "definition": "Choose the most plausible continuation and return 0, 1, 2, or 3."},
]


def _format_real_example(spec: Dict[str, Any], ex: Dict[str, Any]) -> Tuple[str, str]:
    kind = spec["kind"]
    if kind == "classify":
        parts = [f"{field}: {ex[field]}" for field in spec["fields"]]
        label = int(ex["label"])
        return "\n".join(parts), spec["labels"][label]
    if kind == "qnli":
        text = f"Question: {ex['question']}\nSentence: {ex['sentence']}"
        return text, "true" if int(ex["label"]) == 0 else "false"
    if kind == "mrpc":
        text = f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}"
        return text, "true" if int(ex["label"]) == 1 else "false"
    if kind == "sciq":
        # Rotate the correct answer deterministically so it is not always choice A.
        options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        shift = sum(ord(c) for c in ex["question"]) % 4
        options = options[shift:] + options[:shift]
        correct = "ABCD"[(-shift) % 4]
        choices = "\n".join(f"{letter}: {value}" for letter, value in zip("ABCD", options))
        return f"Question: {ex['question']}\n{choices}", correct
    if kind == "choices":
        choices = ex["choices"]
        labels, texts = choices["label"], choices["text"]
        rendered = "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))
        return f"Question: {ex['question']}\n{rendered}", str(ex["answerKey"])
    if kind == "hellaswag":
        rendered = "\n".join(f"{i}: {ending}" for i, ending in enumerate(ex["endings"]))
        return f"Passage: {ex['ctx']}\n{rendered}", str(ex["label"])
    raise ValueError(f"Unknown SFT formatter: {kind}")


def _load_real_task(spec: Dict[str, Any], max_samples: int) -> List[Dict]:
    from datasets import load_dataset

    path, config = spec["path"], spec["config"]
    # All repositories in REAL_SFT_TASKS are explicitly curated above.
    kwargs = {
        "split": "train",
        "cache_dir": _hf_cache_dir(),
        "trust_remote_code": True,
    }
    ds = load_dataset(path, config, **kwargs) if config else load_dataset(path, **kwargs)
    rows = []
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        input_text, output_text = _format_real_example(spec, ex)
        rows.append({
            "input": input_text,
            "output": output_text,
            "definition": spec["definition"],
        })
    return rows


def prepare_sni_split(
    n_train: int,
    n_holdout: int,
    samples_per_task: int,
    data_dir: Path,
    try_download: bool = False,
) -> Dict[str, Any]:
    """
    Build multi-task corpus for SFT.

    Default: a synthetic smoke-test corpus.
    Set T2L_USE_HF_TASKS=1 for the diverse real task suite.
    Set T2L_DOWNLOAD_SNI=1 to prefer Lots-of-LoRAs SNI slugs instead.
    """
    import os

    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "sni_manifest.json"
    try_sni = try_download or os.environ.get("T2L_DOWNLOAD_SNI") == "1"
    use_hf = os.environ.get("T2L_USE_HF_TASKS") == "1"
    expected_source = "sni" if try_sni else ("hf_diverse" if use_hf else "synthetic")

    if manifest_path.exists():
        with open(manifest_path) as f:
            cached = json.load(f)
        cache_matches = (
            cached.get("schema_version") == 2
            and cached.get("source") == expected_source
            and len(cached.get("train_tasks", [])) == n_train
            and len(cached.get("holdout_tasks", [])) == n_holdout
        )
        if cache_matches:
            return cached
        print("Cached SFT manifest does not match this run; rebuilding it.")

    definitions: Dict[str, str] = {}
    loaded_train, loaded_hold = [], []

    if try_sni:
        train_slugs = SNI_TRAIN_CANDIDATES[:n_train]
        hold_slugs = SNI_HOLDOUT[:n_holdout]
        for slug in train_slugs + hold_slugs:
            rows = try_load_sni_task(slug, samples_per_task)
            if not rows:
                rows = _synthetic_sni(slug, samples_per_task)
            definitions[slug] = rows[0].get("definition") or slug.replace("_", " ")
            with open(data_dir / f"{slug}.json", "w") as f:
                json.dump(rows, f)
            (loaded_train if slug in train_slugs else loaded_hold).append(slug)
        source = "sni"
    elif use_hf:
        base_specs = list(REAL_SFT_TASKS)
        requested = n_train + n_holdout
        if requested > len(base_specs):
            raise ValueError(
                f"Requested {requested} real SFT tasks, but only "
                f"{len(base_specs)} unique tasks are configured. "
                "Reduce n_train_tasks/n_holdout_tasks instead of duplicating tasks."
            )
        specs = base_specs[:requested]
        for i, spec in enumerate(specs):
            key = spec["key"]
            try:
                rows = _load_real_task(spec, samples_per_task)
            except Exception as e:
                print(f"Warn: failed {key} ({e}); using format-diverse synthetic data")
                rows = _synthetic_sni(key, samples_per_task)
                for row in rows:
                    row["definition"] = spec["definition"]
            definitions[key] = spec["definition"]
            with open(data_dir / f"{key}.json", "w") as f:
                json.dump(rows, f)
            if i < n_train:
                loaded_train.append(key)
            else:
                loaded_hold.append(key)
        source = "hf_diverse"
    else:
        # Fast default: synthetic multi-task corpus (no network)
        for i in range(n_train + n_holdout):
            slug = f"synth_task_{i:02d}"
            kind = [
                "sentiment classification",
                "topic classification",
                "entailment",
                "paraphrase detection",
                "question answering",
                "summarization style transfer",
            ][i % 6]
            definition = f"Perform {kind} on the given input and return a short label or answer."
            rows = []
            for j in range(samples_per_task):
                rows.append(
                    {
                        "input": f"{kind} example {j}: the quick brown fox jumps {j}.",
                        "output": f"{j % 4}",
                        "definition": definition,
                    }
                )
            definitions[slug] = definition
            with open(data_dir / f"{slug}.json", "w") as f:
                json.dump(rows, f)
            if i < n_train:
                loaded_train.append(slug)
            else:
                loaded_hold.append(slug)
        source = "synthetic"

    if not loaded_hold and len(loaded_train) > n_holdout:
        loaded_hold = loaded_train[-n_holdout:]
        loaded_train = loaded_train[:-n_holdout]

    manifest = {
        "schema_version": 2,
        "train_tasks": loaded_train,
        "holdout_tasks": loaded_hold,
        "definitions": definitions,
        "source": source,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_config(path: str | Path) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def format_choices(labels: List[str], texts: List[str]) -> str:
    return "\n".join(f"{lab}: {txt}" for lab, txt in zip(labels, texts))


def example_to_prompt(task: str, ex: Dict[str, Any]) -> Tuple[str, str]:
    """Return (prompt, answer_string) for a benchmark example."""
    if task == "boolq":
        prompt = TEMPLATES["boolq"].format(
            passage=ex["passage"], question=ex["question"]
        )
        ans = "true" if ex["answer"] else "false"
        return prompt, ans
    if task in {"arc_easy", "arc_challenge"}:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        prompt = TEMPLATES[task].format(
            question=ex["question"], choices=format_choices(labels, texts)
        )
        return prompt, ex["answerKey"]
    if task == "piqa":
        prompt = TEMPLATES["piqa"].format(
            goal=ex["goal"], sol1=ex["sol1"], sol2=ex["sol2"]
        )
        return prompt, str(ex["label"])
    if task == "winogrande":
        prompt = TEMPLATES["winogrande"].format(
            sentence=ex["sentence"],
            option1=ex["option1"],
            option2=ex["option2"],
        )
        return prompt, str(ex["answer"])
    if task == "openbookqa":
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        prompt = TEMPLATES["openbookqa"].format(
            question_stem=ex["question_stem"],
            choices=format_choices(labels, texts),
        )
        return prompt, ex["answerKey"]
    if task == "hellaswag":
        ends = ex["endings"]
        prompt = TEMPLATES["hellaswag"].format(
            ctx=ex["ctx"], e0=ends[0], e1=ends[1], e2=ends[2], e3=ends[3]
        )
        return prompt, str(ex["label"])
    if task == "gsm8k":
        prompt = TEMPLATES["gsm8k"].format(question=ex["question"])
        # gold after ####
        gold = ex["answer"].split("####")[-1].strip()
        return prompt, gold
    raise ValueError(task)


def load_hf_benchmark(task: str, split: str = "train", max_samples: Optional[int] = None):
    from datasets import load_dataset

    mapping = {
        "boolq": ("google/boolq", None, split if split != "validation" else "validation"),
        "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "train" if split == "train" else "test"),
        "arc_challenge": (
            "allenai/ai2_arc",
            "ARC-Challenge",
            "train" if split == "train" else "test",
        ),
        "piqa": ("piqa", None, "train" if split == "train" else "validation"),
        "winogrande": (
            "winogrande",
            "winogrande_xl",
            "train" if split == "train" else "validation",
        ),
        "openbookqa": (
            "allenai/openbookqa",
            "main",
            "train" if split == "train" else "test",
        ),
        "hellaswag": (
            "Rowan/hellaswag",
            None,
            "train" if split == "train" else "validation",
        ),
        "gsm8k": ("openai/gsm8k", "main", "train" if split == "train" else "test"),
    }
    path, config, sp = mapping[task]
    kwargs = {"split": sp, "cache_dir": _hf_cache_dir()}
    # PIQA is still distributed through an official Hugging Face dataset script.
    # `datasets<4` supports it, but requires explicit opt-in in non-interactive runs.
    if path == "piqa":
        kwargs["trust_remote_code"] = True
    if config:
        ds = load_dataset(path, config, **kwargs)
    else:
        ds = load_dataset(path, **kwargs)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


class BenchmarkSFTDataset(Dataset):
    """Tokenized prompt→answer pairs for oracle / multi-task LoRA SFT."""

    def __init__(
        self,
        task: str,
        tokenizer,
        max_seq_len: int = 512,
        max_samples: int = 2000,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        raw = load_hf_benchmark(task, split=split, max_samples=max_samples)
        self.rows: List[Tuple[str, str]] = []
        for ex in raw:
            try:
                self.rows.append(example_to_prompt(task, ex))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, answer = self.rows[idx]
        full = prompt + "\n" + answer
        tok = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        labels = tok["input_ids"].clone().squeeze(0)
        # mask prompt tokens
        prompt_len = len(
            self.tokenizer(prompt + "\n", add_special_tokens=True)["input_ids"]
        )
        prompt_len = min(prompt_len, labels.size(0))
        labels[:prompt_len] = -100
        labels[tok["attention_mask"].squeeze(0) == 0] = -100
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "labels": labels,
        }

#description banks
def paraphrase_templates(base: str, n: int) -> List[str]:
    """Cheap template paraphrases without an external LLM API."""
    prefixes = [
        "",
        "Your job: ",
        "Task: ",
        "Please ",
        "In this setting, ",
        "Objective — ",
        "Instruction: ",
        "Goal: ",
    ]
    suffixes = [
        "",
        " Keep answers short.",
        " Do not explain.",
        " Be precise.",
        " Follow the output format.",
        " Prefer a single token when possible.",
    ]
    out = []
    for i in range(n):
        p = prefixes[i % len(prefixes)]
        s = suffixes[(i * 3) % len(suffixes)]
        text = f"{p}{base[0].lower() + base[1:] if p.startswith('Please') else base}{s}"
        out.append(re.sub(r"\s+", " ", text).strip())
    # dedupe preserve order
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:n]


def build_description_bank(
    tasks: List[str],
    n_per_task: int,
    out_path: Path,
    sni_definitions: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    bank: Dict[str, List[str]] = {}
    for t in tasks:
        if t in DEFAULT_DESCRIPTIONS:
            base_list = DEFAULT_DESCRIPTIONS[t]
            desc = list(base_list)
            while len(desc) < n_per_task:
                desc.extend(paraphrase_templates(base_list[0], n_per_task - len(desc)))
            bank[t] = desc[:n_per_task]
        elif sni_definitions and t in sni_definitions:
            base = sni_definitions[t]
            bank[t] = paraphrase_templates(base, n_per_task)
        else:
            bank[t] = paraphrase_templates(f"Perform the task named {t}.", n_per_task)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bank, f, indent=2)
    return bank


def load_description_bank(path: Path) -> Dict[str, List[str]]:
    with open(path) as f:
        return json.load(f)


#speculation and negation identification (SNI) task loading

def try_load_sni_task(task_slug: str, max_samples: int = 256) -> Optional[List[Dict]]:
    """Load input/output pairs for one SNI task. Returns None on failure."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    candidates = [
        f"Lots-of-LoRAs/{task_slug}",
        task_slug,
    ]
    for name in candidates:
        try:
            ds = load_dataset(name, split="train")
            rows = []
            for i, ex in enumerate(ds):
                if i >= max_samples:
                    break
                inp = ex.get("input") or ex.get("Input") or ex.get("question")
                out = ex.get("output") or ex.get("Output") or ex.get("answer")
                if isinstance(out, list):
                    out = out[0] if out else ""
                defn = ex.get("definition") or ex.get("Definition") or ""
                if inp is None or out is None:
                    continue
                rows.append({"input": str(inp), "output": str(out), "definition": str(defn)})
            if rows:
                return rows
        except Exception:
            continue
    return None


def _synthetic_sni(slug: str, n: int) -> List[Dict]:
    output_formats = ["true", "false", "A", "B", "0", "1", "positive", "negative"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "input": f"Example input {i} for {slug}",
                "output": output_formats[i % len(output_formats)],
                "definition": f"Classify or transform text for {slug.replace('_', ' ')}.",
            }
        )
    return rows


def load_sni_rows(data_dir: Path, slug: str) -> List[Dict]:
    with open(data_dir / f"{slug}.json") as f:
        return json.load(f)


class HyperSFTDataset(Dataset):
    """Samples (task, description, tokenized I/O) for end-to-end T2L SFT."""

    def __init__(
        self,
        task_slugs: List[str],
        data_dir: Path,
        desc_bank: Dict[str, List[str]],
        tokenizer,
        max_seq_len: int = 512,
    ):
        self.task_slugs = task_slugs
        self.task_to_id = {t: i for i, t in enumerate(task_slugs)}
        self.data_dir = data_dir
        self.desc_bank = desc_bank
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.index: List[Tuple[str, int]] = []
        self.cache: Dict[str, List[Dict]] = {}
        for t in task_slugs:
            rows = load_sni_rows(data_dir, t)
            self.cache[t] = rows
            for j in range(len(rows)):
                self.index.append((t, j))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task, j = self.index[idx]
        row = self.cache[task][j]
        desc = random.choice(self.desc_bank.get(task, [task]))
        definition = row.get("definition") or desc
        prompt = TEMPLATES["sni"].format(
            definition=definition, input=row["input"]
        )
        answer = str(row["output"])

        # Tokenize target separately so long inputs cannot truncate the answer.
        # Include EOS in the supervised target so generation learns to stop.
        target_ids = self.tokenizer(
            answer, add_special_tokens=False
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            target_ids.append(self.tokenizer.eos_token_id)
        target_ids = target_ids[: max(1, self.max_seq_len // 4)]
        prompt_budget = max(1, self.max_seq_len - len(target_ids))
        prompt_ids = self.tokenizer(
            prompt + " ",
            add_special_tokens=True,
            truncation=True,
            max_length=prompt_budget,
        )["input_ids"]

        input_ids = (prompt_ids + target_ids)[: self.max_seq_len]
        labels_list = ([-100] * len(prompt_ids) + target_ids)[: self.max_seq_len]
        attention = [1] * len(input_ids)
        pad_len = self.max_seq_len - len(input_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids.extend([pad_id] * pad_len)
        labels_list.extend([-100] * pad_len)
        attention.extend([0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "task": task,
            "task_id": self.task_to_id[task],
            "description": desc,
        }

#oracle LoRA weight packing

def peft_state_to_lora_dict(
    state: Dict[str, torch.Tensor],
    module_names: List[str],
    n_layers: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convert PEFT LoRA state_dict keys into format:
    {module: {"A": (L,r,in), "B": (L,out,r)}}.
    """
    result: Dict[str, Dict[str, torch.Tensor]] = {}
    for mod in module_names:
        As, Bs = {}, {}
        for k, v in state.items():
            if mod not in k:
                continue
            # find layer index
            m = re.search(r"layers\.(\d+)\.", k)
            if not m:
                continue
            li = int(m.group(1))
            if "lora_A" in k:
                As[li] = v.detach().float().cpu()
            elif "lora_B" in k:
                Bs[li] = v.detach().float().cpu()
        if not As:
            continue
        r, in_f = As[0].shape
        out_f = Bs[0].shape[0]
        A = torch.zeros(n_layers, r, in_f)
        B = torch.zeros(n_layers, out_f, r)
        for li, t in As.items():
            A[li] = t
        for li, t in Bs.items():
            B[li] = t
        result[mod] = {"A": A, "B": B}
    return result


def save_lora_dict(path: Path, lora: Dict[str, Dict[str, torch.Tensor]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora, path)

def load_lora_dict(path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    return torch.load(path, map_location="cpu", weights_only=False)

def compute_zscore_stats(
    library: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
) -> Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Per-element mean/std across tasks for each module's A and B.
    library: task -> module -> {A,B}
    """
    tasks = list(library.keys())
    mods = list(next(iter(library.values())).keys())
    stats = {}
    for mod in mods:
        A_stack = torch.stack([library[t][mod]["A"] for t in tasks])
        B_stack = torch.stack([library[t][mod]["B"] for t in tasks])
        stats[mod] = {
            "A": (A_stack.mean(0), A_stack.std(0).clamp_min(1e-6)),
            "B": (B_stack.mean(0), B_stack.std(0).clamp_min(1e-6)),
        }
    return stats


def zscore_lora(lora, stats):
    out = {}
    for mod, mats in lora.items():
        Am, As = stats[mod]["A"]
        Bm, Bs = stats[mod]["B"]
        Am = Am.to(device=mats["A"].device, dtype=mats["A"].dtype)
        As = As.to(device=mats["A"].device, dtype=mats["A"].dtype)
        Bm = Bm.to(device=mats["B"].device, dtype=mats["B"].dtype)
        Bs = Bs.to(device=mats["B"].device, dtype=mats["B"].dtype)
        out[mod] = {
            "A": (mats["A"] - Am) / As,
            "B": (mats["B"] - Bm) / Bs,
        }
    return out


def dezscore_lora(lora, stats):
    out = {}
    for mod, mats in lora.items():
        Am, As = stats[mod]["A"]
        Bm, Bs = stats[mod]["B"]
        Am = Am.to(device=mats["A"].device, dtype=mats["A"].dtype)
        As = As.to(device=mats["A"].device, dtype=mats["A"].dtype)
        Bm = Bm.to(device=mats["B"].device, dtype=mats["B"].dtype)
        Bs = Bs.to(device=mats["B"].device, dtype=mats["B"].dtype)
        out[mod] = {
            "A": mats["A"] * As + Am,
            "B": mats["B"] * Bs + Bm,
        }
    return out
