import sys
sys.path.insert(0, "bench")
from fused_ce_bench import parse_args, verdict


def test_verdict_thresholds():
    ours = dict(ms_per_iter=10.0, peak_bytes=1000)
    ok, why = verdict(ours, dict(ms_per_iter=9.6, peak_bytes=990))
    assert ok, why
    assert not verdict(ours, dict(ms_per_iter=9.0, peak_bytes=1000))[0]       # speed 0.90x < 0.95x
    assert not verdict(ours, dict(ms_per_iter=10.0, peak_bytes=950))[0]       # memory 1.053x > 1.02x


def test_defaults_are_the_circus_shape():
    a = parse_args([])
    assert (a.tokens, a.hidden, a.vocab, a.dtype, a.iters) == (8192, 2048, 64000, "bf16", 20)
