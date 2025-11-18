# palletizer.py
import streamlit as st
import copy
import math
import os
import datetime
import tempfile
import pandas as pd
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# -------------------- Page --------------------
st.set_page_config(page_title="Palletizer", layout="wide")
st.markdown("<h2 style='text-align:center; color:#008080;'>JK Fenner Palletizer Dashboard — MaxRects Edition</h2>", unsafe_allow_html=True)

# -------------------- Defaults --------------------
DEFAULT_PALLET = {'L': 48.0, 'W': 40.0, 'H': 36.0}
DEFAULT_BOXES = {
    'AZ17': [40.0, 24.0, 9.0],
    'AZ13': [40.0, 16.0, 9.0],
    'AZ6':  [40.0, 11.3, 9.0],
    'AZ16': [24.0, 20.0, 9.0],
    'AZ4':  [24.0, 10.0, 9.0],
    'AZ3':  [24.0, 8.0, 9.0],
    'AZ15': [22.9, 19.3, 9.0],
    'AZ11': [22.7, 13.0, 9.0],
    'AZ14': [22.375, 18.75, 9.0],
    'AZ10': [22.375, 12.75, 9.0],
    'AZ12': [20.0, 16.0, 9.0],
    'AZ8':  [18.375, 11.6, 9.0],
    'AZ5':  [18.375, 11.0, 9.0],
    'AZ7':  [12.0, 11.375, 9.0],
    'AZ2':  [12.0, 8.0, 9.0],
    'AZ18': [48.0, 40.0, 18.0]
}
MOQ = 10  # units per box

def normalize_box_name(name: str) -> str:
    return ''.join(ch for ch in str(name).upper().strip() if ch.isalnum())

# Normalize DEFAULT_BOXES keys
DEFAULT_BOXES = {normalize_box_name(k): v for k, v in DEFAULT_BOXES.items()}

# -------------------- Sidebar Inputs --------------------
with st.sidebar:
    if os.path.exists(os.path.join(os.path.dirname(__file__), "jk_fenner_logo.png")):
        st.image("jk_fenner_logo.png", width=150)
    st.markdown("### Pallet settings & Inputs")
    pallet_L = st.number_input("Pallet length (in)", value=float(DEFAULT_PALLET['L']))
    pallet_W = st.number_input("Pallet width (in)", value=float(DEFAULT_PALLET['W']))
    pallet_H = st.number_input("Pallet height (in)", value=float(DEFAULT_PALLET['H']))
    scale = st.number_input("Scale factor (visual)", value=8, min_value=1)
    st.markdown("---")
    st.markdown("Paste orders (Part, Qty) — one per line")
    st.markdown("Example: E71531,870")
    paste_text = st.text_area("Paste Orders (one per line, comma/tab/space separated)", height=240)
    st.markdown("---")
    show_pdf = st.checkbox("Enable PDF download buttons", value=True)
    enable_reserve = st.checkbox("Enable reserve small remainders (legacy behavior)", value=False)

# -------------------- Load mapping --------------------
mapping_file = os.path.join(os.path.dirname(__file__), "MASTER PART.xlsx")
try:
    df_map = pd.read_excel(mapping_file)
    df_map.columns = [c.strip().upper() for c in df_map.columns]
    part_col = [c for c in df_map.columns if "PART" in c.upper()][0]
    box_col = [c for c in df_map.columns if "BOX" in c.upper()][0]
    part_series = df_map[part_col].astype(str).apply(lambda x: x.strip().upper())
    box_series = df_map[box_col].astype(str).apply(normalize_box_name)
    PART_TO_BOX = dict(zip(part_series, box_series))
    BOX_TO_PARTS = {}
    for p, b in PART_TO_BOX.items():
        BOX_TO_PARTS.setdefault(b, []).append(p)
    ALL_PARTS = sorted(PART_TO_BOX.keys())
except Exception as e:
    st.warning(f"MASTER PART.xlsx not found or failed to load: {e}")
    PART_TO_BOX, BOX_TO_PARTS, ALL_PARTS = {}, {}, []

# -------------------- Colors --------------------
if 'colors' not in st.session_state:
    st.session_state.colors = {}
for key in list(ALL_PARTS) + list(DEFAULT_BOXES.keys()):
    if key not in st.session_state.colors:
        h = abs(hash(key)) % (256**3)
        r = (h >> 16) & 0xFF
        g = (h >> 8) & 0xFF
        b = h & 0xFF
        r = 80 + (r % 160)
        g = 80 + (g % 160)
        b = 80 + (b % 160)
        st.session_state.colors[key] = f"rgba({r},{g},{b},0.85)"
st.session_state.colors.setdefault("", "rgba(200,200,200,0.6)")

# -------------------- MaxRects Implementation (2D) --------------------
# Based on common MaxRects variations (best short side fit as default).
class Rect:
    def __init__(self, x, y, w, h, name=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.name = name

    def area(self):
        return self.w * self.h

    def __repr__(self):
        return f"Rect(x={self.x},y={self.y},w={self.w},h={self.h},name={self.name})"

class MaxRectsBin:
    def __init__(self, width, height, allow_rotate=True):
        self.width = width
        self.height = height
        self.allow_rotate = allow_rotate
        self.free_rects = [Rect(0, 0, width, height)]
        self.used_rects = []

    def _find_position_for_new_node_best_short_side_fit(self, w, h):
        best_rect = None
        best_short_side = None
        best_long_side = None
        best_x = best_y = 0
        best_rotated = False

        for fr in self.free_rects:
            # try without rotation
            if w <= fr.w + 1e-9 and h <= fr.h + 1e-9:
                leftover_h = abs(fr.h - h)
                leftover_w = abs(fr.w - w)
                short_side = min(leftover_h, leftover_w)
                long_side = max(leftover_h, leftover_w)
                if best_rect is None or (short_side < best_short_side) or (short_side == best_short_side and long_side < best_long_side):
                    best_rect = fr
                    best_short_side = short_side
                    best_long_side = long_side
                    best_x, best_y = fr.x, fr.y
                    best_rotated = False
            # try with rotation if allowed
            if self.allow_rotate and h <= fr.w + 1e-9 and w <= fr.h + 1e-9:
                leftover_h = abs(fr.h - w)
                leftover_w = abs(fr.w - h)
                short_side = min(leftover_h, leftover_w)
                long_side = max(leftover_h, leftover_w)
                if best_rect is None or (short_side < best_short_side) or (short_side == best_short_side and long_side < best_long_side):
                    best_rect = fr
                    best_short_side = short_side
                    best_long_side = long_side
                    best_x, best_y = fr.x, fr.y
                    best_rotated = True
        if best_rect is None:
            return None
        return (best_x, best_y, w if not best_rotated else h, h if not best_rotated else w, best_rotated, best_rect)

    def _split_free_rect(self, free_rect, used):
        new_rects = []
        # used and free rect overlap check
        if used.x >= free_rect.x + free_rect.w or used.x + used.w <= free_rect.x or used.y >= free_rect.y + free_rect.h or used.y + used.h <= free_rect.y:
            # no overlap
            return [free_rect]
        # split horizontally
        if used.x > free_rect.x and used.x < free_rect.x + free_rect.w:
            new_rects.append(Rect(free_rect.x, free_rect.y, used.x - free_rect.x, free_rect.h))
        if used.x + used.w < free_rect.x + free_rect.w:
            new_rects.append(Rect(used.x + used.w, free_rect.y, (free_rect.x + free_rect.w) - (used.x + used.w), free_rect.h))
        # split vertically
        if used.y > free_rect.y and used.y < free_rect.y + free_rect.h:
            new_rects.append(Rect(free_rect.x, free_rect.y, free_rect.w, used.y - free_rect.y))
        if used.y + used.h < free_rect.y + free_rect.h:
            new_rects.append(Rect(free_rect.x, used.y + used.h, free_rect.w, (free_rect.y + free_rect.h) - (used.y + used.h)))
        return new_rects

    def _prune_free_list(self):
        pruned = []
        for i, r in enumerate(self.free_rects):
            overlapped = False
            for j, r2 in enumerate(self.free_rects):
                if i != j and r.x >= r2.x - 1e-9 and r.y >= r2.y - 1e-9 and (r.x + r.w) <= (r2.x + r2.w) + 1e-9 and (r.y + r.h) <= (r2.y + r2.h) + 1e-9:
                    overlapped = True
                    break
            if not overlapped:
                pruned.append(r)
        self.free_rects = pruned

    def insert(self, name, w, h):
        pos = self._find_position_for_new_node_best_short_side_fit(w, h)
        if pos is None:
            return None
        x, y, pw, ph, rotated, used_fr = pos
        used = Rect(x, y, pw, ph, name)
        self.used_rects.append(used)
        # split free rects
        new_free = []
        for fr in self.free_rects:
            splitted = self._split_free_rect(fr, used)
            new_free.extend(splitted)
        self.free_rects = new_free
        self._prune_free_list()
        return {'name': name, 'x': used.x, 'y': used.y, 'L': used.w, 'W': used.h, 'rotated': rotated}

# -------------------- Packer: pack one layer using MaxRects --------------------
def pack_one_layer_maxrects(pallet_L, pallet_W, boxes_info, order_left):
    """
    boxes_info: dict of code -> {'L','W','H','Area'}
    order_left: dict code -> count (mutated)
    returns placed_list (list of placements) and updated order_left
    """
    # build list of items to try to place sorted by area desc
    items = []
    for code, cnt in order_left.items():
        for i in range(cnt):
            items.append((code, boxes_info[code]['L'], boxes_info[code]['W'], boxes_info[code]['H']))
    if not items:
        return [], order_left

    # sort largest area first improves packing
    items.sort(key=lambda t: t[1]*t[2], reverse=True)

    bin = MaxRectsBin(pallet_L, pallet_W, allow_rotate=True)
    placed = []
    used_count = {}
    for code, Lb, Wb, Hb in items:
        res = bin.insert(code, Lb, Wb)
        if res is None:
            # try rotated (MaxRects already considers rotation in insert), so skip
            continue
        # placed successfully
        placed.append({'name': code, 'x': res['x'], 'y': res['y'], 'L': res['L'], 'W': res['W'], 'H': boxes_info[code]['H'], 'rotated': res['rotated']})
        used_count[code] = used_count.get(code, 0) + 1

    # subtract used_count from order_left
    for k, v in used_count.items():
        order_left[k] -= v
        if order_left[k] < 0:
            order_left[k] = 0

    return placed, order_left

# -------------------- High-level: pack all pallets by layering --------------------
def build_info(boxes):
    info = {}
    for nm, dims in boxes.items():
        L, W, H = dims
        info[nm] = {'L': float(L), 'W': float(W), 'H': float(H), 'Area': float(L)*float(W)}
    return info

def pack_all_pallets_maxrects(pallet, boxes, order_counts, reserve_small=False):
    """
    pallet: dict L,W,H
    boxes: dict code -> [L,W,H]
    order_counts: dict code->count
    reserve_small: if True, reserve small remainders (legacy)
    Returns: list of pallets where each pallet is list of layers, each layer is list of placed boxes
    """
    order_left = copy.deepcopy(order_counts)
    info = build_info(boxes)
    pallets_layers = []

    # Optionally compute and reserve remainders (legacy). Default false for better packing.
    forced_remainders = {}
    if reserve_small:
        for code, tot in list(order_left.items()):
            if tot <= 0:
                continue
            # estimate per-layer count by fitting box on pallet once
            per_layer_est = max(1, int((pallet['L'] // info[code]['L']) * (pallet['W'] // info[code]['W'])))
            if per_layer_est == 0:
                per_layer_est = 1
            rem = tot % per_layer_est
            if rem and (rem <= 5 or rem <= math.ceil(0.10 * tot)):
                forced_remainders[code] = rem
                order_left[code] -= rem

    # Now pack until nothing left
    while sum(order_left.values()) > 0:
        layers_for_this_pallet = []
        current_stack_height = 0.0
        while current_stack_height < pallet['H'] and sum(order_left.values()) > 0:
            # attempt to pack a layer
            placed, order_left = pack_one_layer_maxrects(pallet['L'], pallet['W'], info, order_left)
            if not placed:
                # cannot place any more boxes in this pallet layer stack (maybe some tall box etc.)
                break
            tallest = max([b['H'] for b in placed]) if placed else 0
            # if adding this layer exceeds height, revert the placements (put back to order_left) and break
            if current_stack_height + tallest > pallet['H'] + 1e-9:
                # return those placed to order_left
                for b in placed:
                    order_left[b['name']] = order_left.get(b['name'], 0) + 1
                break
            layers_for_this_pallet.append(placed)
            current_stack_height += tallest
        if not layers_for_this_pallet:
            # can't put any layer -> break to avoid infinite loop
            break
        pallets_layers.append(layers_for_this_pallet)

    # After main packing, add dedicated pallets for forced remainders
    for code, rem in forced_remainders.items():
        dims = boxes[code]
        per_layer = max(1, int((pallet['L'] // dims[0]) * (pallet['W'] // dims[1])))
        if per_layer == 0:
            per_layer = 1
        while rem > 0:
            take = min(per_layer, rem)
            layer = []
            # naive stacked placement with x/y zeros (visualization will scale)
            for _ in range(take):
                layer.append({'name': code, 'x': 0.0, 'y': 0.0, 'L': dims[0], 'W': dims[1], 'H': dims[2], 'rotated': False})
            pallets_layers.append([layer])
            rem -= take

    return pallets_layers

# -------------------- Parse pasted orders --------------------
order_counts = {}
order_part_queue = {}
for line in paste_text.splitlines():
    s = line.strip()
    if not s:
        continue
    if ',' in s:
        a, b = [t.strip() for t in s.split(',', 1)]
    else:
        parts = s.split()
        if len(parts) >= 2:
            a, b = parts[0].strip(), parts[1].strip()
        else:
            continue
    a_up = a.upper()
    if a_up in DEFAULT_BOXES:
        st.warning(f"Detected box code '{a_up}' in pasted input. Please paste PART numbers only. This line was skipped.")
        continue
    if a_up not in PART_TO_BOX:
        st.warning(f"Unknown part '{a_up}' skipped.")
        continue
    try:
        qty_units = int(float(b))
    except:
        st.warning(f"Bad qty on line '{line}' — skipped.")
        continue
    boxes_needed = math.ceil(qty_units / MOQ)
    box_code = normalize_box_name(PART_TO_BOX[a_up])
    order_counts[box_code] = order_counts.get(box_code, 0) + boxes_needed
    order_part_queue.setdefault(box_code, []).extend([a_up] * boxes_needed)

# ensure default boxes keys exist
for b in DEFAULT_BOXES.keys():
    order_counts.setdefault(b, 0)
    order_part_queue.setdefault(b, [])

if not any(v > 0 for v in order_counts.values()):
    st.info("No valid parts found from pasted orders. Paste PART numbers & quantities to pack.")
    st.stop()

# -------------------- Prepare boxes_for_packing (allow rotation automatically handled by packer) --------------------
boxes_for_packing = copy.deepcopy(DEFAULT_BOXES)

# -------------------- Run packer --------------------
pallet = {'L': float(pallet_L), 'W': float(pallet_W), 'H': float(pallet_H)}
reserve_flag = enable_reserve
pallet_layers = pack_all_pallets_maxrects(pallet, boxes_for_packing, order_counts, reserve_small=reserve_flag)
total_pallets = len(pallet_layers)

# -------------------- Assign parts to placed boxes for summary & visuals --------------------
local_queues_for_summary = {k: v.copy() for k, v in order_part_queue.items()}
summary_per_pallet = []
grand_total = {}

assigned_layers_per_pallet = []
for pal_layers in pallet_layers:
    assigned_layers_per_pallet.append([])
    pal_layer_dicts = []
    pal_totals = {}
    for layer in pal_layers:
        layer_counts = {}
        # The layer is list of boxes with 'name' code; assign part from queue if available
        for b in layer:
            box_code = b['name']
            qlist = local_queues_for_summary.get(box_code, [])
            if qlist:
                part_assigned = qlist.pop(0)
            else:
                part_assigned = BOX_TO_PARTS.get(box_code, [''])[0] or ''
            b['part'] = part_assigned
            key = f"{part_assigned} ({box_code})" if part_assigned else box_code
            layer_counts[key] = layer_counts.get(key, 0) + 1
            pal_totals[key] = pal_totals.get(key, 0) + 1
            grand_total[key] = grand_total.get(key, 0) + 1
        pal_layer_dicts.append(layer_counts)
        assigned_layers_per_pallet[-1].append(layer)
    summary_obj = {'_layers': pal_layer_dicts, '_pallet_totals': {'total_boxes': sum(pal_totals.values())}}
    for k, v in pal_totals.items():
        summary_obj[k] = v
    summary_per_pallet.append(summary_obj)

# -------------------- Visual: top view with Plotly --------------------
def plot_layer_topview(pallet, layer_data, colors_map, scale=8, title=None, show_labels=True):
    L_px = pallet['L'] * scale
    W_px = pallet['W'] * scale
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, y0=0, x1=L_px, y1=W_px, line=dict(width=2, color='black'), fillcolor="rgba(0,0,0,0)")
    for d in layer_data:
        x0 = d['x'] * scale
        y0 = d['y'] * scale
        x1 = x0 + d['L'] * scale
        y1 = y0 + d['W'] * scale
        part = d.get('part','')
        color_key = part if part else d['name']
        color = colors_map.get(color_key, colors_map.get(d['name'], "rgba(200,200,200,0.7)"))
        fig.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color='black', width=1), fillcolor=color)
        if show_labels:
            label = f"{part} ({d['name']})" if part else d['name']
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label, showarrow=False,
                               font=dict(size=9), xanchor='center', yanchor='middle', font_color="white")
    fig.update_xaxes(showticklabels=False, range=[0, L_px])
    fig.update_yaxes(showticklabels=False, range=[0, W_px], scaleanchor='x', scaleratio=1)
    fig.update_layout(width=600, height=int(600 * (pallet['W'] / pallet['L'] + 0.02)), margin=dict(l=10, r=10, t=30, b=10))
    if title:
        fig.update_layout(title=title)
    return fig

# -------------------- PDF generation --------------------
def ddmmyyyy_hhmmss_now():
    return datetime.datetime.now().strftime("%d%m%Y_%H%M%S")

def create_summary_pdf_text(summary_per_pallet, grand_total, pallet):
    ts = ddmmyyyy_hhmmss_now()
    path = tempfile.NamedTemporaryFile(delete=False, suffix=f"_summary_{ts}.pdf").name
    c = canvas.Canvas(path, pagesize=A4)
    pw, ph = A4
    margin = 36
    y = ph - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Pallet Packing - Summary Report")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')}")
    y -= 18
    total_pallets = len(summary_per_pallet)
    c.drawString(margin, y, f"Total Pallets: {total_pallets}")
    y -= 20

    for p_idx, pal_sum in enumerate(summary_per_pallet, start=1):
        if y < 120:
            c.showPage(); y = ph - 60
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"Pallet {p_idx}")
        y -= 14
        c.setFont("Helvetica", 10)
        layers = pal_sum.get('_layers', [])
        if layers:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin+6, y, "Layer")
            c.drawString(margin+120, y, "Box Type")
            c.drawString(margin+260, y, "Boxes")
            y -= 12
            c.setFont("Helvetica", 10)
            for lnum, layer_dict in enumerate(layers, start=1):
                for boxk, cnt in layer_dict.items():
                    if y < 80:
                        c.showPage(); y = ph - 60
                    c.drawString(margin+6, y, f"{lnum}")
                    c.drawString(margin+120, y, f"{boxk}")
                    c.drawString(margin+260, y, f"{cnt}")
                    y -= 12
                y -= 6
        pal_tot = pal_sum.get('_pallet_totals', {})
        if y < 80:
            c.showPage(); y = ph - 60
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin+6, y, "Pallet Totals:")
        y -= 12
        c.setFont("Helvetica", 10)
        total_boxes = pal_tot.get('total_boxes', sum([v for k,v in pal_sum.items() if not k.startswith('_')]))
        total_material = total_boxes * MOQ
        c.drawString(margin+20, y, f"Total boxes: {total_boxes}")
        y -= 12
        c.drawString(margin+20, y, f"Total material (units): {total_material}")
        y -= 18

    if y < 120:
        c.showPage(); y = ph - 60
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Grand Totals")
    y -= 14
    c.setFont("Helvetica", 10)
    total_boxes_all = sum(v for k,v in grand_total.items())
    total_material_all = total_boxes_all * MOQ
    c.drawString(margin+10, y, f"Total boxes across all pallets: {total_boxes_all}")
    y -= 12
    c.drawString(margin+10, y, f"Total material across all pallets (units): {total_material_all}")
    y -= 12

    c.save()
    return path

def create_layout_pdf_visuals(assigned_layer_details, pallet):
    ts = ddmmyyyy_hhmmss_now()
    path = tempfile.NamedTemporaryFile(delete=False, suffix=f"_layout_{ts}.pdf").name
    c = canvas.Canvas(path, pagesize=A4)
    pw, ph = A4
    margin = 20
    cols = 2
    rows = 4
    thumb_w = (pw - margin*2 - (cols-1)*6) / cols
    thumb_h = (ph - margin*2 - 60 - (rows-1)*6) / rows

    for p_idx, pal_layers in enumerate(assigned_layer_details, start=1):
        i = 0
        while i < len(pal_layers):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, ph - margin - 10, f"Pallet {p_idx} - Layout (Generated {datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')})")
            for r in range(rows):
                for co in range(cols):
                    idx = i
                    x = margin + co*(thumb_w+6)
                    y_top = ph - margin - 40 - r*(thumb_h+6)
                    c.rect(x, y_top - thumb_h, thumb_w, thumb_h, stroke=1, fill=0)
                    if idx < len(pal_layers):
                        layer = pal_layers[idx]
                        if layer:
                            min_x = min(b['x'] for b in layer)
                            min_y = min(b['y'] for b in layer)
                            max_x = max(b['x'] + b['L'] for b in layer)
                            max_y = max(b['y'] + b['W'] for b in layer)
                            lw = max_x - min_x
                            lh = max_y - min_y
                            if lw <= 0: lw = 1.0
                            if lh <= 0: lh = 1.0
                            sx = (thumb_w - 8) / lw
                            sy = (thumb_h - 8) / lh
                            s = min(sx, sy)
                        else:
                            s = 1.0
                            min_x = min_y = 0.0
                        for b in layer:
                            bx = x + 4 + (b['x'] - min_x) * s
                            by_top = y_top - 4 - (b['y'] - min_y) * s
                            bw = b['L'] * s
                            bh = b['W'] * s
                            c.setFillColorRGB(0.85, 0.9, 1)
                            c.rect(bx, by_top - bh, bw, bh, stroke=1, fill=1)
                            lab = (b.get('part','') + ' (' + b['name'] + ')') if b.get('part','') else b['name']
                            c.setFont('Helvetica', 6)
                            c.drawString(bx+2, by_top - bh/2 - 3, lab[:40])
                        c.setFont('Helvetica-Bold', 9)
                        c.drawString(x+2, y_top - thumb_h - 10, f"Layer {idx+1}")
                    i += 1
                    if i >= len(pal_layers):
                        break
            c.showPage()
    c.save()
    return path

# -------------------- Top bar & PDF buttons --------------------
st.markdown("---")
top_cols = st.columns([2, 4, 1.2, 1.2])
with top_cols[0]:
    st.markdown(f"<div style='display:flex; align-items:center; gap:8px;'><h4 style='margin:0;'>✅ Total pallets used: {total_pallets}</h4></div>", unsafe_allow_html=True)
with top_cols[1]:
    st.write("")
with top_cols[2]:
    if show_pdf:
        try:
            layout_pdf_path = create_layout_pdf_visuals(assigned_layers_per_pallet, pallet)
            with open(layout_pdf_path, 'rb') as f:
                st.download_button("🖼 Download Layout PDF", f, file_name=os.path.basename(layout_pdf_path), mime="application/pdf")
        except Exception as e:
            st.error(f"Error preparing Layout PDF: {e}")
with top_cols[3]:
    if show_pdf:
        try:
            summary_pdf_path = create_summary_pdf_text(summary_per_pallet, grand_total, pallet)
            with open(summary_pdf_path, 'rb') as f:
                st.download_button("📝 Download Summary PDF", f, file_name=os.path.basename(summary_pdf_path), mime="application/pdf")
        except Exception as e:
            st.error(f"Error preparing Summary PDF: {e}")
st.markdown("---")

# -------------------- Show Pallets & Layers --------------------
local_queues = {k: v.copy() for k, v in order_part_queue.items()}
for p_idx, layers in enumerate(pallet_layers, start=1):
    st.markdown(f"## 🟫 Pallet {p_idx}")
    for l_idx, layer in enumerate(layers, start=1):
        fig = plot_layer_topview(pallet, layer, st.session_state.colors, scale=scale, title=f"Pallet {p_idx} — Layer {l_idx}", show_labels=True)
        st.plotly_chart(fig, key=f"final_view_p{p_idx}_l{l_idx}", use_container_width=True)

st.caption("Use the Download buttons above to get the Layout and Summary PDFs.")
