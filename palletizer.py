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

# -------------------- Dashboard title --------------------
st.markdown("<h2 style='text-align:center; color:#008080;'>JK Fenner Palletizer Dashboard</h2>", unsafe_allow_html=True)

# -------------------- Constants / Defaults --------------------
DEFAULT_PALLET = {'L': 42.0, 'W': 42.0, 'H': 90.0}
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

# -------------------- Helpers --------------------
def normalize_box_name(name: str) -> str:
    return ''.join(ch for ch in str(name).upper().strip() if ch.isalnum())

# Normalize keys of DEFAULT_BOXES
DEFAULT_BOXES = {normalize_box_name(k): v for k, v in DEFAULT_BOXES.items()}

# -------------------- Sidebar: logo + inputs --------------------
with st.sidebar:
    # logo above inputs (the image file must exist in same folder)
    if os.path.exists(os.path.join(os.path.dirname(__file__), "jk_fenner_logo.png")):
        st.image("jk_fenner_logo.png", width=150)
    st.markdown("### Pallet settings & Inputs")
    pallet_L = st.number_input("Pallet length (in)", value=float(DEFAULT_PALLET['L']))
    pallet_W = st.number_input("Pallet width (in)", value=float(DEFAULT_PALLET['W']))
    pallet_H = st.number_input("Pallet height (in)", value=float(DEFAULT_PALLET['H']))
    scale = st.number_input("Scale factor (visual)", value=8, min_value=1)
    st.markdown("---")
    st.markdown("Paste orders (Part, Qty)")
    st.markdown("Example: E71531,870")
    paste_text = st.text_area("Paste Orders (one per line, comma/tab/space separated)", height=240)
    st.markdown("---")
    show_pdf = st.checkbox("Enable PDF download buttons", value=True)

# -------------------- Load mapping from Excel (silent on success) --------------------
mapping_file = os.path.join(os.path.dirname(__file__), "MASTER PART.xlsx")
try:
    df_map = pd.read_excel(mapping_file)
    df_map.columns = [c.strip().upper() for c in df_map.columns]
    part_col = [c for c in df_map.columns if "PART" in c][0]
    box_col = [c for c in df_map.columns if "BOX" in c][0]

    part_series = df_map[part_col].astype(str).apply(lambda x: x.strip().upper())
    box_series = df_map[box_col].astype(str).apply(normalize_box_name)

    PART_TO_BOX = dict(zip(part_series, box_series))
    BOX_TO_PARTS = {}
    for p, b in PART_TO_BOX.items():
        BOX_TO_PARTS.setdefault(b, []).append(p)

    ALL_PARTS = sorted(PART_TO_BOX.keys())
except Exception as e:
    st.error(f"Failed to load 'MASTER PART.xlsx': {e}")
    PART_TO_BOX, BOX_TO_PARTS, ALL_PARTS = {}, {}, []

# -------------------- Color palette (deterministic) --------------------
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

# -------------------- Fit utility --------------------
def fit_boxes_on_pallet(pallet_L, pallet_W, box_L, box_W):
    nL1, nW1 = int(pallet_L // box_L), int(pallet_W // box_W)
    c1 = nL1 * nW1
    nL2, nW2 = int(pallet_L // box_W), int(pallet_W // box_L)
    c2 = nL2 * nW2
    if c2 > c1:
        return c2, True, nL2, nW2
    else:
        return c1, False, nL1, nW1

# -------------------- Packer utilities --------------------
def build_info(boxes):
    info = {}
    for nm, dims in boxes.items():
        L, W, H = dims
        info[nm] = {'L': float(L), 'W': float(W), 'H': float(H), 'Area': float(L) * float(W)}
    return info

def try_place(free_rects, dims):
    best = None
    L_box, W_box = dims
    for i, fr in enumerate(free_rects):
        fx, fy, fL, fW = fr
        fr_area = fL * fW
        for l, w, rotated in [(L_box, W_box, False), (W_box, L_box, True)]:
            if l <= fL + 1e-9 and w <= fW + 1e-9:
                leftover = fr_area - (l * w)
                new = free_rects.copy()
                new.pop(i)
                if fL - l > 1e-9:
                    new.append([fx + l, fy, fL - l, w])
                if fW - w > 1e-9:
                    new.append([fx, fy + w, fL, fW - w])
                placement = [fx, fy, l, w, rotated]
                if best is None or leftover < best[0]:
                    best = (leftover, new, placement)
    if best is None:
        return False, free_rects, [0,0,0,0,False]
    return True, best[1], best[2]

def pack_one_layer(pallet, boxes, order_left):
    placed = []
    free_rects = [[0.0, 0.0, pallet['L'], pallet['W']]]
    names = [n for n in order_left.keys() if order_left[n] > 0]
    if not names:
        return placed, order_left
    names.sort(key=lambda n: boxes[n]['Area'], reverse=True)
    for nm in names:
        while order_left[nm] > 0:
            dims = [boxes[nm]['L'], boxes[nm]['W']]
            ok, new_free, pos = try_place(free_rects, dims)
            if not ok:
                break
            fx, fy, L_used, W_used, rotated = pos
            free_rects = new_free
            order_left[nm] -= 1
            placed.append({'name': nm, 'x': fx, 'y': fy, 'L': L_used, 'W': W_used,
                           'H': boxes[nm]['H'], 'rotated': rotated})
    # fill leftover fragments with smaller boxes
    rem_names = [n for n in names if order_left[n] > 0]
    rem_names.sort(key=lambda n: boxes[n]['Area'])
    r = 0
    while r < len(free_rects):
        fr = free_rects[r]
        filled = False
        for nm in rem_names:
            if order_left[nm] <= 0:
                continue
            ok, new_free, pos = try_place([fr], [boxes[nm]['L'], boxes[nm]['W']])
            if ok:
                fx, fy, L_used, W_used, rotated = pos
                order_left[nm] -= 1
                placed.append({'name': nm, 'x': fx, 'y': fy, 'L': L_used, 'W': W_used,
                               'H': boxes[nm]['H'], 'rotated': rotated})
                free_rects.pop(r)
                free_rects.extend(new_free)
                filled = True
                break
        if not filled:
            r += 1
    return placed, order_left

def pack_all_pallets(pallet, boxes, order):
    pallets = []
    pallet_layers = []
    order_left = copy.deepcopy(order)
    info = build_info(boxes)
    max_height = pallet['H']
    while sum(order_left.values()) > 0:
        layer_list = []
        current_height = 0.0
        while current_height < max_height:
            remaining = [n for n, q in order_left.items() if q > 0]
            if not remaining:
                break
            placed, order_left = pack_one_layer(pallet, info, order_left)
            if not placed:
                break
            layer_list.append(placed)
            tallest = max([b['H'] for b in placed]) if placed else 0
            current_height += tallest
            if current_height >= max_height:
                break
        if not layer_list:
            break
        pallet_layers.append(layer_list)
        pallets.append({})
    return pallets, pallet_layers

# -------------------- Visual helper --------------------
def plot_layer_topview(pallet, layer_data, part_assignments, colors_map, scale=8, title=None, show_labels=True):
    L_px = pallet['L'] * scale
    W_px = pallet['W'] * scale
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, y0=0, x1=L_px, y1=W_px, line=dict(width=2, color='black'), fillcolor="rgba(0,0,0,0)")
    for i, d in enumerate(layer_data):
        x0 = d['x'] * scale
        y0 = d['y'] * scale
        x1 = x0 + d['L'] * scale
        y1 = y0 + d['W'] * scale
        part = part_assignments[i] if i < len(part_assignments) else ''
        color_key = part if part else d['name']
        color = colors_map.get(color_key, "rgba(200,200,200,0.7)")
        fig.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color='black', width=1), fillcolor=color)
        if show_labels:
            label = f"{part} ({d['name']})" if part else d['name']
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label, showarrow=False,
                               font=dict(size=10), xanchor='center', yanchor='middle', font_color="white")
    fig.update_xaxes(showticklabels=False, range=[0, L_px])
    fig.update_yaxes(showticklabels=False, range=[0, W_px], scaleanchor='x', scaleratio=1)
    fig.update_layout(width=600, height=int(600 * (pallet['W'] / pallet['L'] + 0.02)), margin=dict(l=10, r=10, t=30, b=10))
    if title:
        fig.update_layout(title=title)
    return fig

# -------------------- PDF generators --------------------
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
            c.drawString(margin+80, y, "Box Type")
            c.drawString(margin+200, y, "Boxes")
            y -= 12
            c.setFont("Helvetica", 10)
            for lnum, layer_dict in enumerate(layers, start=1):
                for boxk, cnt in layer_dict.items():
                    if y < 80:
                        c.showPage(); y = ph - 60
                    c.drawString(margin+6, y, f"{lnum}")
                    c.drawString(margin+80, y, f"{boxk}")
                    c.drawString(margin+200, y, f"{cnt}")
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

def create_layout_pdf_visuals(assigned_layer_details, pallet, scale_for_pdf=4):
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

# if no orders -> stop
if not any(v > 0 for v in order_counts.values()):
    st.info("No valid parts found from pasted orders. Paste PART numbers & quantities to pack.")
    st.stop()

# -------------------- Smart per-box orientation --------------------
pallet = {'L': float(pallet_L), 'W': float(pallet_W), 'H': float(pallet_H)}
boxes_for_packing = copy.deepcopy(DEFAULT_BOXES)
orientation_info = {}
for bcode, dims in DEFAULT_BOXES.items():
    best_count, rotated, nL, nW = fit_boxes_on_pallet(pallet['L'], pallet['W'], dims[0], dims[1])
    orientation_info[bcode] = {'rotated': rotated, 'per_layer_est': best_count, 'layout': (nL, nW)}
    if rotated:
        boxes_for_packing[bcode] = [dims[1], dims[0], dims[2]]

pallets, pallet_layers = pack_all_pallets(pallet, boxes_for_packing, order_counts)
total_pallets = len(pallet_layers)

# -------------------- Prepare PDFs --------------------
assigned_layers_per_pallet = []
summary_per_pallet = []
grand_total = {}
local_queues_for_summary = {k: v.copy() for k, v in order_part_queue.items()}

for p_idx, layers in enumerate(pallet_layers, start=1):
    assigned_layers_per_pallet.append([])
    pal_layer_dicts = []
    pal_totals = {}
    for l_idx, layer in enumerate(layers, start=1):
        layer_counts = {}
        for b in layer:
            box_code = b['name']
            qlist = local_queues_for_summary.get(box_code, [])
            if qlist:
                part_assigned = qlist.pop(0)
            else:
                part_assigned = BOX_TO_PARTS.get(box_code, [''])[0] or ''
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

# -------------------- Top bar: Total pallets + PDF buttons --------------------
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

# -------------------- Display pallets & layers --------------------
local_queues = {k: v.copy() for k, v in order_part_queue.items()}
for p_idx, layers in enumerate(pallet_layers, start=1):
    st.markdown(f"## 🟫 Pallet {p_idx}")
    for l_idx, layer in enumerate(layers, start=1):
        part_assignments = []
        for b in layer:
            box_code = b['name']
            qlist = local_queues.get(box_code, [])
            if qlist:
                part_assigned = qlist.pop(0)
            else:
                part_assigned = BOX_TO_PARTS.get(box_code, [''])[0] or ''
            b['part'] = part_assigned
            part_assignments.append(part_assigned)
        fig = plot_layer_topview(pallet, layer, part_assignments, st.session_state.colors, scale=scale, title=f"Pallet {p_idx} — Layer {l_idx}", show_labels=True)
        st.plotly_chart(fig, key=f"final_view_p{p_idx}_l{l_idx}", use_container_width=True)

st.caption("Use the Download buttons above to get the Layout and Summary PDFs.")
