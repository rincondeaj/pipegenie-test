import pytest
import numpy as np
from collections import defaultdict

from pipegenie.logging._logging import Logbook

def test_logbook():
    lb = Logbook(headers=["a", "b"], chapter_headers={"chap1": ["x", "y"]})

    assert lb.headers == ["a", "b"]
    assert "chap1" in lb.chapters
    assert lb.chapters["chap1"].headers == ["x", "y"]

def test_logbook_record():
    lb = Logbook(headers=["a", "b"])
    lb.record(a=1, b=2)

    assert len(lb) == 1
    assert lb[0]["a"] == 1
    assert lb[0]["b"] == 2

def test_logbook_record_missing_keys():
    lb = Logbook(headers=["a", "b", "c"])
    lb.record(a=1)

    assert "b" in lb[0] and np.isnan(lb[0]["b"])
    assert "c" in lb[0] and np.isnan(lb[0]["c"])

def test_logbook_record_chapter():
    lb = Logbook(headers=["gen"], chapter_headers={"stats": ["hp", "mp"]})
    lb.record(gen=0, stats={"hp": 10})
    
    assert lb[0]["gen"] == 0

    chapter_record = lb.chapters["stats"][0]
    assert chapter_record["hp"] == 10
    assert "mp" in chapter_record and np.isnan(chapter_record["mp"])

def test_logbook_apply_to_all_in_chapters():
    lb = Logbook(headers=["gen"], chapter_headers={"stats": ["hp", "mp"]})
    lb.record(gen=0, stats={"hp": 10})
    chapter_record = lb.chapters["stats"][0]

    assert chapter_record["gen"] == 0

def test_logbook_stream_output():
    lb = Logbook(headers=["a", "b"], chapter_headers={"chap": ["x", "y"]})
    lb.record(a=1, b=2, chap={"x": 5})
    out = lb.stream
    
    assert isinstance(out, str)
    assert "1" in out
    assert "2" in out
    
    chapter_record = lb.chapters["chap"][0]
    assert chapter_record["x"] == 5
    assert "y" in chapter_record and np.isnan(chapter_record["y"])

def test_logbook_multiple_records():
    lb = Logbook(headers=["a", "b"])
    lb.record(a=1, b=2)
    lb.record(a=3)
    
    assert len(lb) == 2
    assert lb[1]["a"] == 3
    assert np.isnan(lb[1]["b"])

def test_logbook_record_new_header():
    lb = Logbook(headers=["a", "b"], chapter_headers={"chap": ["x", "y"]})
    
    lb.record(a=1, b=2, c=3)
    
    assert lb[0]["a"] == 1
    assert lb[0]["b"] == 2
    assert lb[0]["c"] == 3

def test_logbook_add_headers():
    lb = Logbook(headers=["gen", "stats"], chapter_headers={"stats": ["a", "b"]})

    lb.record(gen=0, stats={"a": 10, "b": 20})
    lb.record(gen=1, stats={"a": 15, "b": 25})

    _ = str(lb)

    chapter = lb.chapters["stats"]
    assert len(chapter) == 2
    assert chapter[0]["a"] == 10
    assert chapter[1]["b"] == 25