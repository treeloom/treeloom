"""Guards for the 2026 benchmark refresh: pricing and alias drift.

Two ways a paid cross-model run silently produces junk, both fixed here:

* an UNPRICED model reported no cost at all. `summarize_arm` only set
  `mean_cost_usd`/`cost_verified` when `compute_cost` returned a number, so a
  model missing from the price table dropped the fields entirely — no value, no
  warning. The run that costs money is the one you least want silent.
* a FLOATING model ID can be remapped by the vendor between runs (the
  2026-06-25 incident). The guard was a fixed name list, which by construction
  only covered IDs that existed when it was written.
"""

import pytest

from treeloom.adapters.benchmark.llm_client import is_floating_alias
from treeloom.domain.benchmark.agent_metrics import _arm_means
from treeloom.domain.benchmark.cost_model import PRICES, compute_cost, price_for

# Confirmed 2026-09-20 against vendor pricing pages. The earlier list named
# "deepseek-v4.1-flash" and "gpt-5.6" — NEITHER MODEL EXISTS. The vendors
# offer deepseek-flash / deepseek-v4-pro and gpt-5.6-{sol,luna,terra}. A run
# on the old keys would have 404'd at the vendor and reported cost_unpriced.
# The models the 2026 refresh will ACTUALLY run. gpt-5.6-sol is priced but
# excluded: it rejects function tools unless reasoning_effort='none', which
# would mean benchmarking a non-reasoning model at frontier prices (see
# cost_model.py). gpt-5.5 does native tool calls with reasoning intact.
REFRESH_MODELS = ["deepseek-flash", "claude-sonnet-5", "claude-opus-5", "gpt-5.5"]
RETIRED_GUESSES = ["deepseek-v4.1-flash", "gpt-5.6"]
USAGE = {"prompt_tokens": 20_000, "completion_tokens": 900,
         "cached_input_tokens": 0, "cache_write_tokens": 0}


class TestRefreshModelsArePriced:
    @pytest.mark.parametrize("model", REFRESH_MODELS)
    def test_has_a_price_entry(self, model):
        assert price_for(model) is not None, (
            f"{model} unpriced — its cost fields would vanish from the summary"
        )

    @pytest.mark.parametrize("model", REFRESH_MODELS)
    def test_cost_is_computable_and_positive(self, model):
        c = compute_cost(USAGE, model)
        assert c and c > 0

    @pytest.mark.parametrize("model", REFRESH_MODELS)
    def test_confirmed_prices_are_marked_verified(self, model):
        """Confirmed 2026-09-20 against each vendor's pricing page, so these
        are now publishable. Sonnet 5 was the one that mattered: the inherited
        $3/$15 was 33% high — its $2/$10 introductory rate became standard."""
        assert price_for(model)["verified"] is True

    @pytest.mark.parametrize("model", RETIRED_GUESSES)
    def test_nonexistent_model_ids_are_not_priced(self, model):
        """Pricing a model ID the vendor does not offer invites a run that
        404s after the budget is approved. Keep them absent, not guessed."""
        assert price_for(model) is None

    def test_established_models_stay_verified(self):
        """The refresh must not quietly downgrade confidence in the prices
        that WERE confirmed."""
        for m in ("deepseek-chat", "claude-sonnet-4-6", "claude-opus-4-8", "gpt-5.5"):
            assert PRICES[m]["verified"] is True


class TestUnpricedModelsAreLoud:
    def _summary(self, model):
        arm = {"judge": {"correctness": 4.0, "completeness": 4.0},
               "total_tokens": 20_900, "turns": 5, "tool_calls": 5,
               "recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0,
               "latency_s": 10.0, **USAGE}
        return _arm_means([{"arms": {"treeloom": arm}}], "treeloom", model)

    def test_unknown_model_is_named_not_omitted(self):
        out = self._summary("some-model-nobody-priced")
        assert out.get("cost_unpriced") == "some-model-nobody-priced"
        assert "mean_cost_usd" not in out

    def test_priced_model_reports_cost_and_no_unpriced_flag(self):
        out = self._summary("deepseek-chat")
        assert out.get("mean_cost_usd", 0) > 0
        assert out.get("cost_verified") is True
        assert "cost_unpriced" not in out

    def test_provisional_model_reports_cost_but_flags_unverified(self, monkeypatch):
        """Every table entry is verified as of 2026-09-20, so this injects a
        synthetic provisional entry: the guard is about the BEHAVIOUR (an
        unverified price still reports a cost, flagged) and must not depend on
        which models happen to be unconfirmed today."""
        monkeypatch.setitem(PRICES, "provisional-test-model",
                            {"input": 1.0, "cached_input": 0.1,
                             "cache_write": 1.25, "output": 5.0,
                             "verified": False})
        out = self._summary("provisional-test-model")
        assert out.get("mean_cost_usd", 0) > 0
        assert out.get("cost_verified") is False
        assert "cost_unpriced" not in out


class TestTheExistingAliasGuardAlreadyCoversTheRefresh:
    """No change was needed here, and that is worth pinning.

    The first attempt at this prep replaced the alias NAME LIST with a rule —
    "DeepSeek publishes no dated IDs, so any undated deepseek name is
    floating". `test_floating_alias.py` rejected it: `deepseek-v4-pro` is
    asserted to PASS. The codebase's position is that a VERSIONED DeepSeek
    name is specific enough to pin, and only the bare rolling aliases float.
    That is right, and the rule would have flagged every new model as drift —
    training people to pass --allow-floating-model reflexively, which is worse
    than no guard.
    """

    @pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
    def test_bare_rolling_aliases_are_still_flagged(self, model):
        assert is_floating_alias(model)

    @pytest.mark.parametrize("model", [
        "deepseek-v4-pro",        # pinned by the existing suite
        "deepseek-v4.1-flash",    # the refresh model — versioned, so specific
    ])
    def test_versioned_deepseek_ids_pass(self, model):
        assert is_floating_alias(model) is None, (
            f"{model} names a specific model version; flagging it would make "
            "the guard noise"
        )

    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5", "gpt-5.6"])
    def test_refresh_models_from_other_vendors_pass(self, model):
        assert is_floating_alias(model) is None

    def test_the_known_floating_names_still_fire(self):
        assert is_floating_alias("gpt-4o")
        assert is_floating_alias("anything-latest")
