import torch

from tensorrt_wan.conditioning.manager import ConditioningManager
from tensorrt_wan.conditioning.source import ConditioningSource
from tensorrt_wan.conditioning.types import ConditioningKind, ConditioningTensor


class _FakeTextSource(ConditioningSource):
    kind = ConditioningKind.TEXT

    def encode(self, inputs: str) -> ConditioningTensor:
        return ConditioningTensor(kind=self.kind, embedding=torch.full((1, 4), float(len(inputs))))


class _FakeLoRASource(ConditioningSource):
    kind = ConditioningKind.LORA

    def encode(self, inputs) -> ConditioningTensor:
        return ConditioningTensor(
            kind=self.kind, metadata={"path": inputs, "state_dict": {"w": torch.ones(2)}}
        )


def test_combine_routes_embedding_by_kind():
    manager = ConditioningManager()
    manager.register(_FakeTextSource())

    unified = manager.combine({ConditioningKind.TEXT: "hello"})

    assert "text" in unified.embeddings
    assert torch.equal(unified.embeddings["text"], torch.full((1, 4), 5.0))


def test_combine_raises_for_unregistered_kind():
    manager = ConditioningManager()
    try:
        manager.combine({ConditioningKind.CONTROL: object()})
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_combine_routes_lora_into_lora_weights_not_embeddings():
    manager = ConditioningManager()
    manager.register(_FakeLoRASource())

    unified = manager.combine({ConditioningKind.LORA: "my_lora.safetensors"})

    assert "lora" not in unified.embeddings
    assert "my_lora.safetensors" in unified.lora_weights
    assert torch.equal(unified.lora_weights["my_lora.safetensors"]["w"], torch.ones(2))


def test_combine_encoded_merges_without_a_registered_source():
    manager = ConditioningManager()
    tensor = ConditioningTensor(kind=ConditioningKind.IMAGE, embedding=torch.zeros(1, 3))

    unified = manager.combine_encoded([tensor])

    assert torch.equal(unified.embeddings["image"], torch.zeros(1, 3))
