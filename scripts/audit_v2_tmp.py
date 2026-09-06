from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path


path = Path('paper/figures/doubao-img-V2.drawio')
root = ET.parse(path).getroot()
cells = [c for c in root.findall('.//mxCell') if c.get('id')]


def style(s: str | None) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for tok in (s or '').split(';'):
        if '=' in tok:
            k, v = tok.split('=', 1)
            out[k] = v
        elif tok:
            out[tok] = None
    return out


def txt(s: str | None) -> str:
    s = html.unescape(s or '')
    s = re.sub(r'<br\s*/?>', '\n', s, flags=re.I)
    s = re.sub(r'<[^>]*>', '', s)
    return s.replace('\n', ' / ').replace('&nbsp;', ' ')


vs = [c for c in cells if c.get('vertex') == '1']
es = [c for c in cells if c.get('edge') == '1']
print(f'cells={len(cells)} vertices={len(vs)} edges={len(es)} labelled_edges={sum(bool(c.get("value")) for c in es)}')
print('\nVERTICES')
for c in vs:
    g = c.find('mxGeometry')
    st = style(c.get('style'))
    print(f"{c.get('id')}\t{g.attrib if g is not None else {}}\tfill={st.get('fillColor')} stroke={st.get('strokeColor')} font={st.get('fontColor')} fs={st.get('fontSize')}\t{txt(c.get('value'))}")
print('\nEDGES')
for c in es:
    g = c.find('mxGeometry')
    st = style(c.get('style'))
    print(f"{c.get('id')}\t{c.get('source')} -> {c.get('target')}\t{g.attrib if g is not None else {}}\tstroke={st.get('strokeColor')} font={st.get('fontColor')} fs={st.get('fontSize')} bg={st.get('labelBackgroundColor')} dashed={st.get('dashed')} arrow={st.get('endArrow')}\t{txt(c.get('value'))}")

print('\nFONT FRAGMENTS')
for c in vs:
    val = html.unescape(c.get('value') or '')
    frags = re.findall(r'(?is)<font[^>]*style=["\']([^"\']*)["\'][^>]*>(.*?)</font>', val)
    if frags:
        print(c.get('id'), frags)

print('\nNONASCII')
for ident in ['semantic_retrieval', 'relation_checks', 'agree_signal', 'accepted_entities', 'detail_stack', 'truth_panel']:
    c = next((x for x in vs if x.get('id') == ident), None)
    if c is not None:
        s = html.unescape(c.get('value') or '')
        print(ident, repr(s), [hex(ord(ch)) for ch in s if ord(ch) > 127][:60])

print('\nCOLOR AUDIT')
for c in cells:
    st = style(c.get('style'))
    v = html.unescape(c.get('value') or '')
    colors = re.findall(r'(?i)(?:fontColor|color|fillColor|strokeColor)\s*[:=]\s*["\']?(#[0-9a-f]{3,8})', (c.get('style') or '') + ' ' + v)
    pale = [x for x in colors if x.lower() in {'#ffffff','#fff','#5b6573','#4f5965','#9aa4b2','#6b7280','#7f8995'}]
    if pale:
        print(c.get('id'), 'pale=', pale, 'style=', st.get('fontColor'), 'value=', txt(v)[:100])

print('\nFONT STYLE COUNTS')
from collections import Counter
print('vertex style fs', Counter(style(c.get('style')).get('fontSize') for c in vs))
print('edge style fs', Counter(style(c.get('style')).get('fontSize') for c in es))
print('edge label styles', Counter((style(c.get('style')).get('fontSize'), style(c.get('style')).get('fontColor'), style(c.get('style')).get('labelBackgroundColor'), style(c.get('style')).get('labelBorderColor')) for c in es if c.get('value')))

print('\nSVG TEXT IMAGE BOUNDS')
svg_path = Path('paper/figures/doubao-img-V2.svg')
if svg_path.exists():
    sr = ET.parse(svg_path).getroot()
    ns = {'s': 'http://www.w3.org/2000/svg'}
    for grp in sr.findall('.//s:g[@data-cell-id]', ns):
        ident = grp.get('data-cell-id')
        if ident not in {c.get('id') for c in vs if c.get('value')}: continue
        ims = grp.findall('.//s:image', ns)
        fos = grp.findall('.//s:foreignObject', ns)
        if not ims: continue
        # First image is fallback text raster; list dimensions and position only.
        im = ims[0]
        print(ident, 'img=',im.get('x'),im.get('y'),im.get('width'),im.get('height'),'fo=',fos[0].get('width') if fos else None, fos[0].get('height') if fos else None)

    # compare raster text footprint against geometry for all text vertices
    geom_by_id = {}
    for c in vs:
        g = c.find('mxGeometry')
        if g is not None and c.get('value'):
            try:
                geom_by_id[c.get('id')] = tuple(float(g.get(k, '0')) for k in ('width', 'height'))
            except ValueError:
                pass
    ratios = []
    for grp in sr.findall('.//s:g[@data-cell-id]', ns):
        ident = grp.get('data-cell-id')
        if ident not in geom_by_id: continue
        ims = grp.findall('.//s:image', ns)
        if not ims: continue
        im = ims[0]
        try:
            iw, ih = float(im.get('width')), float(im.get('height'))
            gw, gh = geom_by_id[ident]
            ratios.append((max(iw/gw, ih/gh), iw/gw, ih/gh, ident))
        except (TypeError, ValueError):
            pass
    print('TOP FOOTPRINT RATIOS (max,width,height,id)')
    for item in sorted(ratios, reverse=True)[:20]: print(item)
    print('HEIGHT-RATIO >= .9')
    for item in sorted((r for r in ratios if r[2] >= .9), reverse=True): print(item)

    print('EDGE LABEL FOOTPRINTS')
    edge_ids = {c.get('id') for c in es if c.get('value')}
    for grp in sr.findall('.//s:g[@data-cell-id]', ns):
        ident = grp.get('data-cell-id')
        if ident not in edge_ids: continue
        ims = grp.findall('.//s:image', ns)
        if not ims: continue
        im = ims[0]
        print(ident, txt(next(c for c in es if c.get('id') == ident).get('value')), 'img=', im.get('width'), im.get('height'), 'at=',im.get('x'),im.get('y'))

print('\nEDGE ROUTE POINTS / POTENTIAL PARALLEL LANES')
for c in es:
    g = c.find('mxGeometry')
    arr = g.find('Array') if g is not None else None
    pts = []
    if arr is not None:
        pts = [(float(q.get('x')), float(q.get('y'))) for q in arr.findall('mxPoint')]
    if pts:
        print(c.get('id'), c.get('source'), '->', c.get('target'), pts)

# collinear overlap among explicit horizontal/vertical route segments
routes = {}
for c in es:
    g = c.find('mxGeometry'); arr = g.find('Array') if g is not None else None
    if arr is None: continue
    pts = [(float(q.get('x')), float(q.get('y'))) for q in arr.findall('mxPoint')]
    if not pts: continue
    routes[c.get('id')] = pts
def segments(pts):
    return list(zip(pts, pts[1:]))
print('COLLINEAR ROUTE OVERLAPS (>=10px)')
for i, (a, pa) in enumerate(routes.items()):
    for b, pb in list(routes.items())[i+1:]:
        for (x1,y1),(x2,y2) in segments(pa):
            for (u1,v1),(u2,v2) in segments(pb):
                if y1 == y2 == v1 == v2:
                    lo, hi = max(min(x1,x2), min(u1,u2)), min(max(x1,x2), max(u1,u2))
                    if hi-lo >= 10: print(a,b,'horizontal',y1,lo,hi)
                if x1 == x2 == u1 == u2:
                    lo, hi = max(min(y1,y2), min(v1,v2)), min(max(y1,y2), max(v1,v2))
                    if hi-lo >= 10: print(a,b,'vertical',x1,lo,hi)

print('CROSSING ROUTES (explicit segments, excluding shared endpoints)')
def orient(p,q): return 'h' if p[1] == q[1] else 'v' if p[0] == q[0] else 'o'
for i,(a,pa) in enumerate(routes.items()):
    for b,pb in list(routes.items())[i+1:]:
        for p,q in segments(pa):
            for u,v in segments(pb):
                op, oq, ou, ov = orient(p,q), orient(u,v), orient(u,v), orient(p,q)
                if {op, oq} != {'h','v'}: continue
                h = (p,q) if op=='h' else (u,v)
                vv = (u,v) if op=='h' else (p,q)
                y=h[0][1]; x=vv[0][0]
                if min(h[0][0],h[1][0]) < x < max(h[0][0],h[1][0]) and min(vv[0][1],vv[1][1]) < y < max(vv[0][1],vv[1][1]):
                    print(a,b,'cross',x,y)
