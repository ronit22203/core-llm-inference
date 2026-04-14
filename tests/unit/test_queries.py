"""Unit tests for src/benchmark/queries.py"""

import json
import tempfile
from pathlib import Path

import pytest
from src.benchmark.queries import filter_by_category, load_queries, sample_queries


@pytest.fixture
def tmp_query_file(tmp_path: Path) -> Path:
    data = {
        "medical_queries": [
            {"id": i, "category": cat, "query": f"Query {i}"}
            for i, cat in enumerate(
                ["Cardiology", "Oncology", "Cardiology", "Neurology", "Oncology"], 1
            )
        ]
    }
    p = tmp_path / "query.json"
    p.write_text(json.dumps(data))
    return p


class TestLoadQueries:
    def test_loads_all(self, tmp_query_file):
        queries = load_queries(tmp_query_file)
        assert len(queries) == 5

    def test_keys_present(self, tmp_query_file):
        q = load_queries(tmp_query_file)[0]
        assert {"id", "category", "query"} <= q.keys()


class TestFilterByCategory:
    def test_no_filter_returns_all(self, tmp_query_file):
        qs = load_queries(tmp_query_file)
        assert filter_by_category(qs, []) == qs

    def test_filter_single(self, tmp_query_file):
        qs = load_queries(tmp_query_file)
        cardio = filter_by_category(qs, ["Cardiology"])
        assert all(q["category"] == "Cardiology" for q in cardio)
        assert len(cardio) == 2

    def test_filter_multiple(self, tmp_query_file):
        qs = load_queries(tmp_query_file)
        result = filter_by_category(qs, ["Cardiology", "Oncology"])
        assert len(result) == 4


class TestSampleQueries:
    def test_sample_n(self, tmp_query_file):
        qs = load_queries(tmp_query_file)
        sampled = sample_queries(qs, 3, seed=0)
        assert len(sampled) == 3

    def test_all_queries(self, tmp_query_file):
        qs = load_queries(tmp_query_file)
        all_q = sample_queries(qs, 3, all_queries=True)
        assert len(all_q) == 5

    def test_sorted_by_id(self, tmp_query_file):
        qs = load_queries(tmp_query_file)
        sampled = sample_queries(qs, 5, seed=99)
        ids = [q["id"] for q in sampled]
        assert ids == sorted(ids)
