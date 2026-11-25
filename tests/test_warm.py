"""`myopenfund-mcp warm`: fake live source, fake indexer and a counting fake SEC (all offline)."""

import datetime as dt
import os

import pytest
from source_fakes import APPLE, MSFT, FakeLive
from test_filings_tools import APPLE_K24, MSFT_K23, MSFT_K24, MSFT_Q, FakeIndexer, status

from myopenfund_mcp import cli, warm
from myopenfund_mcp.companies.live import Warming
from myopenfund_mcp.companies.router import SourceRouter
from myopenfund_mcp.filings.indexer import StatusResult

SENTINEL = "sentinel-contact@example.invalid"
ARGS = ["warm", "--companies", "MSFT", "--forms", "10-K", "--since", "2022-01-01", "--poll-interval", "0.5"]


class CountingSec:
    """The SEC the fake indexer downloads documents from: every URL asked is recorded."""

    def __init__(self):
        self.urls: list[str] = []

    def get_bytes(self, url, *, max_age_s=None):
        self.urls.append(url)
        return b""


class StepIndexer(FakeIndexer):
    """Each `status` call moves a document one step: queued -> indexing -> ready (downloading it from
    `sec` on the way); `fail`ed accessions end `failed`. `ready` states persist across runs."""

    def __init__(self, sec, *, failing=(), **kw):
        super().__init__(**kw)
        self.sec, self.failing = sec, set(failing)

    def status(self, doc_ids=None):
        self._enter("status", doc_ids=doc_ids)
        for d in self.docs.values():
            if doc_ids is not None and d.doc_id not in doc_ids:
                continue
            if d.state == "queued":
                d.state = "indexing"
            elif d.state == "indexing":
                self.sec.get_bytes(d.url)
                d.state, d.error = ("failed", "boom") if d.accession in self.failing else ("ready", None)
        return StatusResult([d for d in self.docs.values() if doc_ids is None or d.doc_id in doc_ids])


class Live(FakeLive):
    def __init__(self, *, warming=0, **kw):
        super().__init__(**kw)
        self.warming = warming

    def concept_rows(self, cik, as_of, **kw):
        if self.warming:
            self.warming -= 1
            raise Warming(1, 3)
        return super().concept_rows(cik, as_of, **kw)


@pytest.fixture
def clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith("MOF_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("MOF_SEC_CONTACT", SENTINEL)
    return monkeypatch


@pytest.fixture
def rig(clean_env, monkeypatch):
    """Patch the stack builder; returns the pieces and the sleeps recorded."""
    sec = CountingSec()
    live = Live(refs={MSFT: [MSFT_K24, MSFT_K23, MSFT_Q], APPLE: [APPLE_K24]})
    indexer = StepIndexer(sec)
    sleeps: list[float] = []
    closed: list[str] = []

    def build(settings):
        router = SourceRouter(None, live, "sec")
        return warm.Stack(router, indexer, closers=(lambda: closed.append("live"),))

    monkeypatch.setattr(warm, "build_stack", build)
    monkeypatch.setattr(warm.time, "sleep", sleeps.append)
    return type("Rig", (), {"sec": sec, "live": live, "indexer": indexer, "sleeps": sleeps, "closed": closed})


def test_all_ready_exits_0_with_progress_on_stderr(rig, capsys):
    assert cli.main(ARGS) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[warm] MSFT facts ok" in captured.err
    assert "[warm] MSFT filings 2" in captured.err  # 10-K only: the 10-Q is not asked for
    assert "[warm] 2/2 documents ready (0 indexing)" in captured.err
    assert "[warm] 0/2 documents ready (2 indexing)" in captured.err
    assert SENTINEL not in captured.err
    assert rig.indexer.closed
    assert sorted(rig.sec.urls) == sorted([MSFT_K24.url, MSFT_K23.url])


def test_facts_and_filings_use_the_window_and_company_names(rig):
    assert cli.main([*ARGS, "--until", "2024-12-31"]) == 0
    assert rig.live.periods == [(dt.date(2022, 1, 1), dt.date(2024, 12, 31))]
    [enqueue] = [kw for m, kw in rig.indexer.calls if m == "enqueue"]
    assert enqueue["names"] == {MSFT: "MICROSOFT CORP"}
    assert enqueue["tickers"] == {MSFT: ["MSFT"]}


def test_default_forms_come_from_the_settings(rig, clean_env):
    clean_env.setenv("MOF_FILINGS_DEFAULT_FORMS", "10-Q")
    argv = [a for a in ARGS if a not in ("--forms", "10-K")]
    assert cli.main(argv) == 0
    assert [kw["refs"][0].form for m, kw in rig.indexer.calls if m == "enqueue"] == ["10-Q"]


def test_a_failed_document_exits_1_and_is_listed(rig, capsys):
    rig.indexer.failing = {MSFT_K23.accession}
    assert cli.main(ARGS) == 1
    err = capsys.readouterr().err
    assert f"failed {MSFT_K23.doc_id}" in err and "boom" in err
    assert "1/2 documents ready" in err
    assert rig.indexer.closed


def test_a_facts_failure_exits_1_after_the_documents_are_done(rig, capsys):
    rig.live.fail = {"concept_rows": RuntimeError("x")}
    assert cli.main(ARGS) == 1
    err = capsys.readouterr().err
    assert "MSFT facts failed" in err and "2/2 documents ready" in err and "failed step: MSFT facts" in err


def test_warming_is_waited_for_at_the_poll_interval(rig, capsys):
    rig.live.warming = 2
    assert cli.main(ARGS) == 0
    assert rig.sleeps[:2] == [0.5, 0.5]
    assert "[warm] MSFT facts ok" in capsys.readouterr().err


def test_poll_interval_defaults_to_the_setting(rig, clean_env):
    clean_env.setenv("MOF_WARM_POLL_INTERVAL_S", "7")
    argv = [a for a in ARGS if a not in ("--poll-interval", "0.5")]
    assert cli.main(argv) == 0
    assert set(rig.sleeps) == {7.0}


def test_enqueue_is_batched_by_max_docs_not_refused(rig, clean_env):
    clean_env.setenv("MOF_FILINGS_MAX_DOCS", "1")
    assert cli.main(ARGS) == 0
    sizes = [len(kw["refs"]) for m, kw in rig.indexer.calls if m == "enqueue"]
    assert sizes == [1, 1]


def test_second_run_makes_no_sec_call_for_ready_documents(rig):
    assert cli.main(ARGS) == 0
    first = list(rig.sec.urls)
    assert len(first) == 2
    enqueues = len([1 for m, _ in rig.indexer.calls if m == "enqueue"])
    assert cli.main(ARGS) == 0
    assert rig.sec.urls == first  # nothing downloaded again
    assert len([1 for m, _ in rig.indexer.calls if m == "enqueue"]) == enqueues  # nor enqueued


def test_a_partly_ready_run_only_enqueues_the_rest(rig, capsys):
    rig.indexer.docs[MSFT_K24.doc_id] = status(MSFT_K24, "ready")
    assert cli.main(ARGS) == 0
    [enqueue] = [kw for m, kw in rig.indexer.calls if m == "enqueue"]
    assert [r.doc_id for r in enqueue["refs"]] == [MSFT_K23.doc_id]
    assert "1/2 documents already ready, skipped" in capsys.readouterr().err
    assert rig.sec.urls == [MSFT_K23.url]


def test_no_filing_in_the_window_exits_0(rig, capsys):
    rig.live.refs = {MSFT: []}
    assert cli.main(ARGS) == 0
    assert "no filing of these forms" in capsys.readouterr().err
    assert not [1 for m, _ in rig.indexer.calls if m == "enqueue"]


def test_ctrl_c_closes_the_indexer_and_exits_130(rig, capsys):
    def interrupt(seconds):
        raise KeyboardInterrupt

    rig.indexer.status = lambda doc_ids=None: (_ for _ in ()).throw(KeyboardInterrupt())
    assert cli.main(ARGS) == 130
    assert rig.indexer.closed
    assert "interrupted" in capsys.readouterr().err


def test_an_unknown_company_exits_2(rig, capsys):
    assert cli.main(["warm", "--companies", "NOSUCH", "--since", "2022-01-01"]) == 2
    assert "[unknown_company]" in capsys.readouterr().err
    assert rig.indexer.closed


@pytest.mark.parametrize(
    "argv",
    [
        ["warm", "--since", "2022-01-01"],  # --companies required
        ["warm", "--companies", "MSFT"],  # --since required
        ["warm", "--companies", "MSFT", "--since", "01/01/2022"],
        ["warm", "--companies", "MSFT", "--since", "2022-13-40"],
        ["warm", "--companies", "MSFT", "--since", "2022-01-01", "--until", "2021-01-01"],
        ["warm", "--companies", "MSFT", "--since", "2022-01-01", "--forms", "10 K"],
        ["warm", "--companies", "MSFT,,AAPL", "--since", "2022-01-01"],
        ["warm", "--companies", " ", "--since", "2022-01-01"],
        ["warm", "--companies", "MSFT", "--since", "2022-01-01", "--poll-interval", "0"],
    ],
)
def test_bad_arguments_exit_2_with_a_message_and_build_nothing(rig, capsys, monkeypatch, argv):
    monkeypatch.setattr(warm, "build_stack", lambda s: pytest.fail("built"))
    with pytest.raises(SystemExit) as exc:  # argparse's own refusals
        code = cli.main(argv)
        raise SystemExit(code)
    assert exc.value.code == 2
    assert capsys.readouterr().err.strip()


def test_missing_contact_exits_2_naming_the_variable(clean_env, capsys, monkeypatch):
    clean_env.delenv("MOF_SEC_CONTACT")
    monkeypatch.setattr(warm, "build_stack", lambda s: pytest.fail("built"))
    assert cli.main(ARGS) == 2
    assert "MOF_SEC_CONTACT" in capsys.readouterr().err


def test_a_configuration_error_exits_2(clean_env, capsys):
    clean_env.setenv("MOF_WARM_POLL_INTERVAL_S", "0")
    assert cli.main(ARGS) == 2
    assert "MOF_WARM_POLL_INTERVAL_S" in capsys.readouterr().err
