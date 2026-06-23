#!/usr/bin/env python3
# Copyright (c) 2026 Peter Fors
# SPDX-License-Identifier: MIT
#
# Full per-category scoreboard for a both-decoder bench CSV: overall geomean
# ratio (ffpng / image-png) and every category sorted worst-to-best. Filenames
# may contain commas, so the six trailing fields are split off the right.
#
# Usage: analyze.py <csv>   (default: /tmp/both.csv)
import csv, math, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/both.csv"
rows = defaultdict(dict)
cat_ratios = defaultdict(list)
with open(path) as f:
	r = csv.reader(f)
	next(r)
	for row in r:
		p = ','.join(row[:-6])
		dec = row[-6]
		mp = float(row[-1])
		rows[p][dec] = mp

ffpng_g = []
ip_g = []
for p, d in rows.items():
	if 'ffpng' in d and 'image-png' in d:
		ffpng_g.append(d['ffpng'])
		ip_g.append(d['image-png'])
		cat_ratios[p.split('/')[-2]].append((d['ffpng'], d['image-png']))

def geo(xs):
	return math.exp(sum(math.log(x) for x in xs) / len(xs))

print("OVERALL ffpng=%.1f image-png=%.1f ratio=%.4f  N=%d" % (geo(ffpng_g), geo(ip_g), geo(ffpng_g) / geo(ip_g), len(ffpng_g)))
print()
print("%-22s %8s %8s %6s %5s" % ("category", "ffpng", "imgpng", "ratio", "N"))
tot = []
for c in cat_ratios:
	o = [x[0] for x in cat_ratios[c]]
	i = [x[1] for x in cat_ratios[c]]
	go, gi = geo(o), geo(i)
	tot.append((c, go, gi, go / gi, len(o)))
for c, go, gi, rt, n in sorted(tot, key=lambda x: x[3]):
	print("%-22s %8.1f %8.1f %6.3f %5d" % (c, go, gi, rt, n))
