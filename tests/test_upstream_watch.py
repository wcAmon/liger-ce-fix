"""Pure-logic tests for tools/upstream_watch.py (no network): diff stat, keyword hits, the notify decision, the body."""
from tools.upstream_watch import diff_stat, keyword_hits, should_notify, render_body


def test_diff_stat_counts_added_and_removed_lines():
    base = "a\nb\nc\n"
    new = "a\nB\nc\nd\n"
    stat = diff_stat("f.py", base, new)
    assert stat["added"] == 2 and stat["removed"] == 1
    assert "+B" in stat["diff"] and "-b" in stat["diff"] and "+d" in stat["diff"]


def test_diff_stat_identical_files_is_empty():
    stat = diff_stat("f.py", "same\n", "same\n")
    assert stat["added"] == 0 and stat["removed"] == 0 and stat["diff"] == ""


def test_keyword_hits_are_case_insensitive_and_report_the_source():
    hits = keyword_hits({"release notes": "Fix Softmax Collapse in CE", "commit abc": "refactor"}, ["softmax collapse", "2610.31642"])
    assert hits == [("softmax collapse", "release notes")]


def test_should_notify_only_when_files_changed_or_keywords_hit():
    unchanged = [{"path": "f.py", "added": 0, "removed": 0, "diff": ""}]
    changed = [{"path": "f.py", "added": 1, "removed": 0, "diff": "+x"}]
    assert should_notify(unchanged, []) is False
    assert should_notify(changed, []) is True
    assert should_notify(unchanged, [("2610.31642", "release notes")]) is True


def test_render_body_lists_stat_commits_hits_and_a_collapsed_diff():
    stats = [{"path": "src/a.py", "added": 3, "removed": 1, "diff": "--- a\n+++ b\n+x\n"}]
    commits = {"src/a.py": [("deadbee", "speed up CE")]}
    body = render_body("linkedin/Liger-Kernel", "v0.8.0", "v0.8.5", stats, commits, [("softmax collapse", "release notes")])
    assert "v0.8.0" in body and "v0.8.5" in body
    assert "src/a.py" in body and "+3" in body and "-1" in body
    assert "deadbee" in body and "speed up CE" in body
    assert "softmax collapse" in body and "release notes" in body
    assert "<details>" in body and "+x" in body
