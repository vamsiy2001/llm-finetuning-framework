"""
Tests for evaluation utilities (evaluation/automated_eval.py, evaluation/llm_judge.py).
Uses synthetic data — no model or GPU required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.automated_eval import (
    compute_bleu,
    compute_response_length_stats,
    compute_rouge,
)
from evaluation.llm_judge import aggregate_scores


# ── ROUGE tests ────────────────────────────────────────────────────────────

class TestComputeRouge:
    def test_perfect_match(self):
        preds = ["The quick brown fox"]
        refs = ["The quick brown fox"]
        scores = compute_rouge(preds, refs)
        assert scores["rougeL"] == 1.0

    def test_no_overlap(self):
        preds = ["apple orange banana"]
        refs = ["dog cat bird"]
        scores = compute_rouge(preds, refs)
        assert scores["rougeL"] == 0.0

    def test_returns_rougeL(self):
        preds = ["I can help with your billing issue."]
        refs = ["I will help you with the billing problem."]
        scores = compute_rouge(preds, refs)
        assert "rougeL" in scores

    def test_multiple_samples(self):
        preds = ["hello world", "foo bar baz"]
        refs = ["hello world", "foo bar baz"]
        scores = compute_rouge(preds, refs)
        assert scores["rougeL"] == 1.0

    def test_scores_are_rounded(self):
        preds = ["hello"]
        refs = ["hello world"]
        scores = compute_rouge(preds, refs)
        for v in scores.values():
            assert isinstance(v, float)
            assert len(str(v).split(".")[-1]) <= 4


# ── BLEU tests ─────────────────────────────────────────────────────────────

class TestComputeBleu:
    def test_perfect_match(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = compute_bleu(preds, refs)
        assert score > 0.9

    def test_no_overlap_returns_zero(self):
        preds = ["apple orange"]
        refs = ["dog cat"]
        score = compute_bleu(preds, refs)
        assert score == 0.0

    def test_returns_float(self):
        preds = ["I can help you with your order."]
        refs = ["I will assist you with the order."]
        score = compute_bleu(preds, refs)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ── Response length stats tests ────────────────────────────────────────────

class TestResponseLengthStats:
    def test_equal_lengths(self):
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "foo bar"]
        stats = compute_response_length_stats(preds, refs)
        assert stats["length_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_pred_shorter_than_ref(self):
        preds = ["hi"]
        refs = ["this is a much longer reference response"]
        stats = compute_response_length_stats(preds, refs)
        assert stats["length_ratio"] < 1.0

    def test_pred_longer_than_ref(self):
        preds = ["this is a much longer predicted response than the reference"]
        refs = ["short"]
        stats = compute_response_length_stats(preds, refs)
        assert stats["length_ratio"] > 1.0

    def test_output_keys(self):
        stats = compute_response_length_stats(["hello world"], ["hi there"])
        assert "avg_pred_length" in stats
        assert "avg_ref_length" in stats
        assert "length_ratio" in stats

    def test_values_rounded(self):
        stats = compute_response_length_stats(["a b c"], ["a b c d e"])
        for v in stats.values():
            assert isinstance(v, float)


# ── LLM Judge aggregate tests ──────────────────────────────────────────────

class TestAggregateScores:
    def test_basic_average(self):
        scores = [
            {"helpfulness": 4, "accuracy": 5, "professionalism": 3},
            {"helpfulness": 2, "accuracy": 3, "professionalism": 5},
        ]
        result = aggregate_scores(scores)
        assert result["avg_helpfulness"] == pytest.approx(3.0)
        assert result["avg_accuracy"] == pytest.approx(4.0)
        assert result["avg_professionalism"] == pytest.approx(4.0)

    def test_composite_is_mean_of_three(self):
        scores = [{"helpfulness": 3, "accuracy": 3, "professionalism": 3}]
        result = aggregate_scores(scores)
        assert result["avg_composite"] == pytest.approx(3.0)

    def test_skips_none_scores(self):
        scores = [
            {"helpfulness": 5, "accuracy": 5, "professionalism": 5},
            {"helpfulness": None, "accuracy": None, "professionalism": None, "reasoning": "error"},
        ]
        result = aggregate_scores(scores)
        assert result["n_valid"] == 1
        assert result["n_failed"] == 1
        assert result["avg_helpfulness"] == 5.0

    def test_empty_list_returns_empty(self):
        result = aggregate_scores([])
        assert result == {}

    def test_all_failed_returns_empty(self):
        scores = [
            {"helpfulness": None, "accuracy": None, "professionalism": None},
        ]
        result = aggregate_scores(scores)
        assert result == {}

    def test_output_keys(self):
        scores = [{"helpfulness": 4, "accuracy": 4, "professionalism": 4}]
        result = aggregate_scores(scores)
        expected_keys = {
            "avg_helpfulness", "avg_accuracy", "avg_professionalism",
            "avg_composite", "n_valid", "n_failed",
        }
        assert expected_keys.issubset(result.keys())


# ── Agent tools tests ──────────────────────────────────────────────────────

class TestOrderTools:
    def test_lookup_order_valid(self):
        from agent.tools.order_tools import lookup_order
        result = lookup_order("ORD-123456")
        assert "order_id" in result
        assert result["order_id"] == "ORD-123456"
        assert "status" in result

    def test_lookup_order_invalid_format(self):
        from agent.tools.order_tools import lookup_order
        result = lookup_order("INVALID")
        assert "error" in result

    def test_cancel_order_already_cancelled(self):
        from agent.tools.order_tools import cancel_order, lookup_order
        with patch("agent.tools.order_tools.lookup_order") as mock_lookup:
            mock_lookup.return_value = {
                "order_id": "ORD-999999",
                "status": "cancelled",
                "shipping_address": "123 Main St",
                "total": 49.99,
                "carrier": "FedEx",
                "tracking_number": "FE123456789",
            }
            result = cancel_order("ORD-999999")
        assert result["success"] is False

    def test_update_address_shipped_order_fails(self):
        from agent.tools.order_tools import update_shipping_address
        with patch("agent.tools.order_tools.lookup_order") as mock_lookup:
            mock_lookup.return_value = {
                "order_id": "ORD-111111",
                "status": "shipped",
                "shipping_address": "Old address",
                "carrier": "UPS",
                "tracking_number": "UP123",
            }
            result = update_shipping_address("ORD-111111", "New address")
        assert result["success"] is False
        assert "carrier" in result


class TestAccountTools:
    def test_get_account_info_valid_email(self):
        from agent.tools.account_tools import get_account_info
        result = get_account_info("test@example.com")
        assert "email" in result
        assert "account_status" in result
        assert "subscription_plan" in result

    def test_get_account_info_invalid_email(self):
        from agent.tools.account_tools import get_account_info
        result = get_account_info("notanemail")
        assert "error" in result

    def test_process_refund_over_limit(self):
        from agent.tools.account_tools import process_refund
        result = process_refund("user@example.com", "ORD-123", 999.99, "damaged item")
        assert result["success"] is False
        assert result.get("escalated") is True

    def test_process_refund_valid(self):
        from agent.tools.account_tools import process_refund
        result = process_refund("user@example.com", "ORD-123", 49.99, "wrong item")
        assert result["success"] is True
        assert "refund_id" in result

    def test_update_subscription_invalid_plan(self):
        from agent.tools.account_tools import update_subscription
        result = update_subscription("user@example.com", "UltraPro")
        assert "error" in result
