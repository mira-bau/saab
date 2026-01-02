"""Shared constants for tokenization and encoding."""

# Special token names
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
NONE_TOKEN = "[NONE]"

# Standard special token sets
VALUE_SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN]
TAG_SPECIAL_TOKENS = [PAD_TOKEN, NONE_TOKEN, UNK_TOKEN]

# Standard special token indices (for consistency)
PAD_IDX = 0
UNK_IDX = 1
NONE_IDX = 1  # For tag vocabularies
MASK_IDX = 2
CLS_IDX = 3
SEP_IDX = 4
