"""ys_lf3_init <src.pt> <dst.pt> (7-39, 3-a): rope14 with the actor's first-layer weights from the twist vx slot zeroed, so the rope's
mass that the LF3 env writes there (ys_rope.add_mass_input) changes nothing at the start. rope14 still reads that slot as a walking
speed (its walking ancestry): +-0.087 there (15 / 25 g) dropped handle_sep 0.56 / 0.69 -> 0.30 / 0.33 and widened its sd 29 -> 82 mm."""
import sys, torch
from mjlab_microduck.tasks.ys_rope import MASS_IN_SLOT
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
w = sd["actor_state_dict"]["mlp.0.weight"]
print(f"slot {MASS_IN_SLOT}: column |w| mean {float(w[:, MASS_IN_SLOT].abs().mean()):.4f} (all columns {float(w.abs().mean()):.4f}) -> 0")
w[:, MASS_IN_SLOT] = 0.0
torch.save(sd, sys.argv[2]); print("saved", sys.argv[2])
