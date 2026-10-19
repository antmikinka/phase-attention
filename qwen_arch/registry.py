"""Registries for layer types. Add a new type with
    @register("attention", "my_attn")   class MyAttn(Qwen3Attention): ...
    @register("position", "my_pos")     class MyPos(RotaryPositions): ...
    @register("extra", "my_layer")      class MyLayer(ExtraLayer): ...
and select it per layer through LayerSpec / a JSON spec."""
ATTENTION_TYPES: dict = {}
POSITION_TYPES: dict = {}
EXTRA_LAYER_TYPES: dict = {}
_KIND = {"attention": ATTENTION_TYPES, "position": POSITION_TYPES, "extra": EXTRA_LAYER_TYPES}


def register(kind: str, name: str):
    def deco(cls):
        _KIND[kind][name] = cls
        cls.type_name = name
        return cls
    return deco
