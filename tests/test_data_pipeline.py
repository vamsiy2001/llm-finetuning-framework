"""
Tests for the data pipeline (data/download_and_clean.py).
Runs locally without GPU or internet access — uses synthetic data.
"""

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.download_and_clean import (
    SYSTEM_PROMPT,
    clean_dataset,
    format_for_training,
)


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe mirroring the Bitext schema."""
    return pd.DataFrame({
        "instruction": [
            "I was charged twice for my order.",
            "How do I return a product?",
            "Can I change my shipping address?",
            "Hi",                                   # too short — should be dropped
            "What is your return policy?",
            "I want to cancel my subscription.",
            "I was charged twice for my order.",    # duplicate
        ],
        "response": [
            "I'm sorry to hear that. Let me look into the duplicate charge for you.",
            "You can return products within 30 days of purchase.",
            "Yes, you can update your shipping address before the order ships.",
            "OK",                                   # too short — should be dropped
            "Our return policy allows returns within 30 days.",
            "I can help you cancel your subscription right away.",
            "I'm sorry to hear that. Let me look into the duplicate charge for you.",  # dup
        ],
        "intent": [
            "billing_issue", "return_request", "change_address",
            "general", "return_request", "cancel_subscription", "billing_issue",
        ],
        "category": [
            "billing", "returns", "shipping",
            "general", "returns", "subscription", "billing",
        ],
        "flags": ["B", "Z", "", "BIK", "Q", "P", "B"],
    })


# ── cleaning tests ─────────────────────────────────────────────────────────

class TestCleanDataset:
    def test_drops_short_instructions(self, sample_df):
        cleaned = clean_dataset(sample_df.copy())
        assert all(cleaned["instruction"].str.len() >= 5)

    def test_drops_short_responses(self, sample_df):
        cleaned = clean_dataset(sample_df.copy())
        assert all(cleaned["response"].str.len() >= 20)

    def test_removes_duplicates(self, sample_df):
        cleaned = clean_dataset(sample_df.copy())
        pairs = cleaned[["instruction", "response"]].drop_duplicates()
        assert len(pairs) == len(cleaned), "Duplicate (instruction, response) pairs remain"

    def test_quality_flag_column_added(self, sample_df):
        cleaned = clean_dataset(sample_df.copy())
        assert "has_quality_flag" in cleaned.columns

    def test_quality_flag_detects_BIK(self, sample_df):
        cleaned = clean_dataset(sample_df.copy())
        # rows with B/I/K flags should be marked True
        flagged = cleaned[cleaned["has_quality_flag"] == True]
        # at least the "B" flagged rows should survive cleaning and be marked
        assert len(flagged) > 0

    def test_whitespace_normalised(self):
        df = pd.DataFrame({
            "instruction": ["  Hello   world  "],
            "response": ["  This is   a   response  with extra spaces.  "],
            "intent": ["test"],
            "category": ["test"],
            "flags": [""],
        })
        cleaned = clean_dataset(df.copy())
        assert cleaned.iloc[0]["instruction"] == "Hello world"
        assert "  " not in cleaned.iloc[0]["response"]

    def test_null_rows_dropped(self):
        df = pd.DataFrame({
            "instruction": ["Valid question", None, "Another question"],
            "response": ["Valid response here.", "A response here.", None],
            "intent": ["a", "b", "c"],
            "category": ["x", "y", "z"],
            "flags": ["", "", ""],
        })
        cleaned = clean_dataset(df.copy())
        assert cleaned["instruction"].isna().sum() == 0
        assert cleaned["response"].isna().sum() == 0

    def test_index_reset_after_cleaning(self, sample_df):
        cleaned = clean_dataset(sample_df.copy())
        assert list(cleaned.index) == list(range(len(cleaned)))


# ── formatting tests ───────────────────────────────────────────────────────

class TestFormatForTraining:
    @pytest.fixture
    def sample_row(self):
        return {
            "instruction": "I was charged twice for my order.",
            "response": "I'm sorry to hear that. Let me check your billing.",
            "intent": "billing_issue",
            "category": "billing",
            "has_quality_flag": False,
        }

    def test_messages_structure(self, sample_row):
        result = format_for_training(sample_row)
        assert "messages" in result
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_system_prompt_content(self, sample_row):
        result = format_for_training(sample_row)
        system_msg = result["messages"][0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == SYSTEM_PROMPT

    def test_user_message_is_instruction(self, sample_row):
        result = format_for_training(sample_row)
        user_msg = result["messages"][1]
        assert user_msg["content"] == sample_row["instruction"]

    def test_assistant_message_is_response(self, sample_row):
        result = format_for_training(sample_row)
        asst_msg = result["messages"][2]
        assert asst_msg["content"] == sample_row["response"]

    def test_metadata_preserved(self, sample_row):
        result = format_for_training(sample_row)
        assert result["intent"] == sample_row["intent"]
        assert result["category"] == sample_row["category"]
        assert result["has_quality_flag"] == sample_row["has_quality_flag"]

    def test_original_fields_preserved(self, sample_row):
        result = format_for_training(sample_row)
        assert result["instruction"] == sample_row["instruction"]
        assert result["response"] == sample_row["response"]


# ── system prompt tests ────────────────────────────────────────────────────

class TestSystemPrompt:
    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_customer_support(self):
        assert "customer support" in SYSTEM_PROMPT.lower()
