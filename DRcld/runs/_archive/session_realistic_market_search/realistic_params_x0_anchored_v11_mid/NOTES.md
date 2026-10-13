# cross_density/realistic_params_x0_anchored_v11_mid

**Test**: v7 of the fixed-x0/parameter-diversity training axis -- "x0-anchored" perturbation
instead of v5/v6's "level+width". Each node's D-box and G-box are rebuilt AROUND the one shared
fixed x0 by rescaling the base market's own distance from x0 to each edge by two INDEPENDENT
positive factors in [0.4,2.5] -- x0 is mathematically guaranteed inside every box, by
construction, for every draw. D*/G_res keep their own independent level factor (+/-20%),
D* re-clipped into the new box. Same topology (density=1.0), same realistic base market as v1-v6.

**Why**: v5/v6 built the box from D*'s level+width and patched x0 back in afterward when it drifted
out (~93% of instances needed a clip). v7 removes the need for that patch entirely -- and because the
two box-edge distances scale independently, x0's relative position inside the box still varies
freely (the "how far is x0 from a good spot" signal that correlated with speedup, r=0.428 in v5),
without the v1-v4 box-collapse risk (the box can shrink toward x0 but never invert past it).

**Dataset**: 75 FIXED training instances (slot 0 = unperturbed original), 30 FIXED test
instances (slot 0 = original). x0 needed adjusting in 0/75 train and
0/30 test instances (sanity check on the by-construction guarantee).

**Results**: mean 1.14x, median 1.12x,
min 0.97x, max 1.38x, win-rate
93%
(28/30).

**Question answered**: does anchoring the box construction to x0 (guaranteeing it's always inside,
while still letting its relative position vary via independent below/above scale factors) improve
on v5's 1.25x/100% (previous best of this training axis)?
