"""GuardTX (the transformer guard of eyeclean-r57): EyeGuard's interface, initialisation from a restorer, and the bundle
loader's guard_arch switch (CPU only, tiny configurations)."""
import json

import pytest
import torch
from safetensors.torch import save_file

from eyeclean.models.bundle import Bundle
from eyeclean.models.eyeguard import GUARD_MASKS
from eyeclean.models.eyerestore_tx import EyeRestoreTX
from eyeclean.models.guard_tx import GuardTX, build_guard_tx

TINY = {"dim": 8, "blocks": (1, 1, 1, 1), "dec_blocks": (1, 1, 1), "refine": 1}


def test_guard_tx_has_the_eyeguard_outputs():
    g = build_guard_tx(TINY).eval()
    with torch.no_grad():
        o = g(torch.rand(2, 4, 64, 64))
    assert o["masks"].shape == (2, len(GUARD_MASKS), 64, 64)
    assert o["present"].shape == (2,) and o["refl_present"].shape == (2,)


def test_init_from_restorer_copies_the_whole_backbone():
    r = EyeRestoreTX(**TINY, heads=(2, 4, 8, 16), checkpoint=False)
    g = GuardTX(**TINY)
    left = g.init_from_restorer(r.state_dict())
    assert left == []
    rs, gs = r.state_dict(), g.net.state_dict()
    assert all(torch.equal(v, rs[k]) for k, v in gs.items() if k != "intro.weight")
    # the input conv keeps the restorer's filters for the shared channels: normalised crop (0-2) and exposure token (9)
    assert torch.equal(gs["intro.weight"], rs["intro.weight"][:, [0, 1, 2, 9]])


def test_bundle_loader_refuses_an_unknown_guard_arch(tmp_path):
    p = tmp_path / "b.safetensors"
    man = {"model": {"guard_arch": "rnn", "crop_size": 256}, "components": {"guard": {}, "restore_single": {}}}
    save_file({"guard.x": torch.zeros(1), "restore_single.x": torch.zeros(1)}, str(p), metadata={"eyeclean_manifest": json.dumps(man)})
    with pytest.raises(ValueError, match="unknown guard_arch"):
        Bundle(p, torch.device("cpu"))
