import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from reward_gap.memory import GapMemory, MemoryContext, MemoryError


CONTEXT = MemoryContext("proxy", "revision-1", "last_non_pad_final_hidden_l2_v1", "calibration-1")


def build(k=2):
    return GapMemory.build(["b", "a"], torch.tensor([[0., 2.], [3., 0.]]),
                           proxy_scores=[0., 4.], judge_scores=[2., 1.],
                           context=CONTEXT, k=k, temperature=1.)


def test_known_cosine_weighted_signed_gaps():
    prediction = build().predict(torch.tensor([[1., 0.]]), context=CONTEXT)
    expected_weights = torch.softmax(torch.tensor([1., 0.], dtype=torch.float64), dim=0)
    assert prediction.neighbor_ids == (("a", "b"),)
    assert prediction.similarities.tolist() == [[1., 0.]]
    torch.testing.assert_close(prediction.weights[0], expected_weights)
    assert prediction.gaps.item() == pytest.approx((3 * expected_weights[0] - 2 * expected_weights[1]).item())
    assert build(k=1).predict(torch.tensor([[0., 1.]]), context=CONTEXT).gaps.item() == -2


def test_ties_independent_of_input_order_and_query_scale():
    results = []
    for ids, gaps in [(["b", "a"], [2., 5.]), (["a", "b"], [5., 2.])]:
        memory = GapMemory(ids, torch.tensor([[1., 0.], [1., 0.]]), gaps, context=CONTEXT, k=1)
        results.append(memory.predict(torch.tensor([[8., 0.], [1., 0.]]), context=CONTEXT))
    for result in results:
        assert result.neighbor_ids == (("a",), ("a",))
        assert result.gaps.tolist() == [5., 5.]


def test_append_preserves_original_and_rejects_duplicate():
    initial = build(k=1)
    refreshed = initial.append(["c"], torch.tensor([[-1., 0.]]), proxy_scores=[8.],
                               judge_scores=[1.], context=CONTEXT)
    assert initial.size == 2
    assert refreshed.size == 3
    assert refreshed.predict(torch.tensor([[-1., 0.]]), context=CONTEXT).gaps.item() == 7
    with pytest.raises(MemoryError, match="unique"):
        initial.append(["a"], torch.tensor([[1., 0.]]), proxy_scores=[0.], judge_scores=[0.], context=CONTEXT)


@pytest.mark.parametrize("field", ["encoder_id", "encoder_revision", "pooling", "calibration_id"])
def test_context_mismatch_rejected(field, tmp_path):
    memory = build()
    wrong = replace(CONTEXT, **{field: "different"})
    with pytest.raises(MemoryError, match="does not match"):
        memory.predict(torch.ones(1, 2), context=wrong)
    with pytest.raises(MemoryError, match="does not match"):
        memory.append(["c"], torch.ones(1, 2), proxy_scores=[0.], judge_scores=[0.], context=wrong)
    path = memory.save(tmp_path / "memory.json")
    with pytest.raises(MemoryError, match="does not match"):
        GapMemory.load(path, context=wrong)


def test_round_trip_and_no_overwrite(tmp_path):
    memory = build()
    path = memory.save(tmp_path / "M0.json")
    original = path.read_bytes()
    restored = GapMemory.load(path, context=CONTEXT)
    query = torch.tensor([[1., 0.], [0., 1.]])
    before = memory.predict(query, context=CONTEXT)
    after = restored.predict(query, context=CONTEXT)
    assert after.neighbor_ids == before.neighbor_ids
    torch.testing.assert_close(after.gaps, before.gaps)
    with pytest.raises(FileExistsError):
        memory.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_detached_and_independent_of_caller_mutation():
    embeddings = torch.tensor([[1., 0.]], requires_grad=True)
    memory = GapMemory(["a"], embeddings, [2.], context=CONTEXT, k=1)
    with torch.no_grad():
        embeddings.zero_()
    prediction = memory.predict(torch.tensor([[1., 0.]], requires_grad=True), context=CONTEXT)
    assert prediction.gaps.item() == 2
    assert not prediction.gaps.requires_grad
    assert prediction.similarities.item() == 1


@pytest.mark.parametrize("vectors", [torch.zeros(1, 2), torch.tensor([[float("nan"), 1.]]),
                                     torch.tensor([[float("inf"), 1.]]), torch.ones(1, 3),
                                     torch.empty(0, 2), torch.ones(2), torch.ones(1, 2, dtype=torch.long)])
def test_invalid_query_vectors(vectors):
    with pytest.raises(MemoryError):
        build().predict(vectors, context=CONTEXT)


@pytest.mark.parametrize("settings", [{"k": 3}, {"k": True}, {"k": 0}, {"temperature": 0},
                                      {"temperature": float("nan")}, {"temperature": True}])
def test_invalid_settings(settings):
    with pytest.raises(MemoryError):
        GapMemory(["a", "b"], torch.eye(2), [1., 2.], context=CONTEXT, **settings)


@pytest.mark.parametrize("proxy,judge", [([1.], [2., 3.]), ([float("nan"), 2.], [1., 2.]),
                                         ([True, 2.], [1., 2.])])
def test_invalid_scores(proxy, judge):
    with pytest.raises(MemoryError):
        GapMemory.build(["a", "b"], torch.eye(2), proxy_scores=proxy, judge_scores=judge,
                        context=CONTEXT, k=1)


def test_malformed_snapshot(tmp_path):
    path = tmp_path / "bad.json"
    for contents in ["not json", "[]", json.dumps({"schema_version": 7})]:
        path.write_text(contents, encoding="utf-8")
        with pytest.raises(MemoryError):
            GapMemory.load(path, context=CONTEXT)


def test_tiny_temperature_stays_finite():
    memory = GapMemory(["a", "b"], torch.eye(2), [3., -2.], context=CONTEXT,
                       k=2, temperature=1e-320)
    assert memory.predict(torch.tensor([[1., 0.]]), context=CONTEXT).gaps.item() == 3
