import json
import pytest
import tiktoken
from retrieval.windows import pack_page_windows

def test_token_limit_splits_whole_pages_and_oversize_context_is_explicit():
    enc = tiktoken.get_encoding("o200k_base")
    text = {p: "hello " * 60 for p in range(1, 4)}
    limit = len(enc.encode(json.dumps({1: text[1]}, ensure_ascii=False)))
    windows = list(pack_page_windows([(p, [p]) for p in text], text, enc, max_tokens=limit))
    assert [w.targets for w in windows] == [(1,), (2,), (3,)]
    assert all(w.text_tokens <= limit and w.pages[next(iter(w.pages))] == text[1] for w in windows)
    with pytest.raises(ValueError, match="not truncated or checked"):
        list(pack_page_windows([(2, [1, 2, 3])], text, enc, max_tokens=limit))
