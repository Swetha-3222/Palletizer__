# palletizer.py
import streamlit as st
import copy
import math
import tempfile
import datetime
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
st.image("jk_fenner_logo.png", width=250)
st.markdown("<h2 style='text-align:center; color:#008080;'>JK Fenner Palletizer Dashboard</h2>", unsafe_allow_html=True)
st.markdown("---")
# -------------------- Configuration --------------------
st.set_page_config(page_title="Palletizer", layout="wide")
st.title("📦 Palletizer")

# -------------------- Defaults --------------------
DEFAULT_PALLET = {'L': 42.0, 'W': 42.0, 'H': 90.0}
DEFAULT_BOXES = {
    'AZ17': [40.0, 24.0, 9.0],
    'AZ13': [40.0, 16.0, 9.0],
    'AZ6': [40.0, 11.3, 9.0],
    'AZ16': [24.0, 20.0, 9.0],
    'AZ4': [24.0, 10.0, 9.0],
    'AZ3': [24.0, 8.0, 9.0],
    'AZ15': [22.9, 19.3, 9.0],
    'AZ11': [22.7, 13.0, 9.0],
    'AZ14': [22.375, 18.75, 9.0],
    'AZ10': [22.375, 12.75, 9.0],
    'AZ12': [20.0, 16.0, 9.0],
    'AZ8': [18.375, 11.6, 9.0],
    'AZ5': [18.375, 11.0, 9.0],
    'AZ7': [12.0, 11.375, 9.0],
    'AZ2': [12.0, 8.0, 9.0],
    'AZ18': [48.0, 40.0, 18.0]
}
MOQ = 10  # units per box

# -------------------- Full embedded Part->Box mapping --------------------
# (Updated / full mapping provided by user)
MAPPING_TEXT = """
A70620 AZ2
A70646 AZ2
A71692 AZ2
A71877 AZ2
B72025 AZ2
B86050 AZ2
B87618 AZ2
B87663 AZ2
C87646 AZ2
C88399 AZ2
C70399 AZ3
E72222 AZ3
E87775 AZ3
C71723 AZ4
D71325 AZ4
D72396 AZ4
E72003 AZ4
E72127 AZ4
E72142 AZ4
E72189 AZ4
E72284 AZ4
E87796 AZ4
B71442 AZ5
B71891 AZ5
C70368 AZ5
C70986 AZ5
C70988 AZ5
C71050 AZ5
C71794 AZ5
C72196 AZ5
C72402 AZ5
C87810 AZ5
D71396 AZ5
D71950 AZ5
D71967 AZ5
D72045 AZ5
D72227 AZ5
D86109 AZ5
D86501 AZ5
D87001 AZ5
D87658 AZ5
D87788 AZ5
D88357 AZ5
D88374 AZ5
E71990 AZ5
E72008 AZ5
E72069 AZ5
E72202 AZ5
E72403 AZ5
E86121 AZ5
E87614 AZ5
E88361 AZ5
D71906 AZ6
E71033 AZ6
E71420 AZ6
E71558 AZ6
E71881 AZ6
E71898 AZ6
E71911 AZ6
E71991 AZ6
E72049 AZ6
E72086 AZ6
E72126 AZ6
E72248 AZ6
E72419 AZ6
E87848 AZ6
A70637 AZ7
A70647 AZ7
A71621 AZ7
A71624 AZ7
A71651 AZ7
B70773 AZ7
B71383 AZ7
B71749 AZ7
B71750 AZ7
B71863 AZ7
B72022 AZ7
B72516 AZ7
B86800 AZ7
B86802 AZ7
B87625 AZ7
B88354 AZ7
C72165 AZ7
C87617 AZ7
C87621 AZ7
C87634 AZ7
C87671 AZ7
C87675 AZ7
C87725 AZ7
B72682 AZ10
C70443 AZ10
C71799 AZ10
C71850 AZ10
C71903 AZ10
D71321 AZ10
D71626 AZ10
D71660 AZ10
D71717 AZ10
D71862 AZ10
D71949 AZ10
D72220 AZ10
D72231 AZ10
E70735 AZ10
E71531 AZ10
E72044 AZ10
E72279 AZ10
E92493 AZ10
C71730 AZ11
C87864 AZ11
E72244 AZ11
D86801 AZ12
D71762 AZ13
E71040 AZ13
E71478 AZ13
E71713 AZ13
E71725 AZ13
E71962 AZ13
E72281 AZ13
E72595 AZ13
E88412 AZ13
D71823 AZ14
D71842 AZ14
E71200 AZ14
E71599 AZ14
E71714 AZ14
E71878 AZ14
E72051 AZ14
E72088 AZ14
E71988 AZ17
E72017 AZ17
"""

# parse mapping to dictionaries
def parse_mapping(text):
    part_to_box = {}
    box_to_parts = {}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        p, b = parts[0].strip().upper(), parts[-1].strip().upper()
        part_to_box[p] = b
        box_to_parts.setdefault(b, []).append(p)
    return part_to_box, box_to_parts

PART_TO_BOX, BOX_TO_PARTS = parse_mapping(MAPPING_TEXT)
ALL_PARTS = sorted(PART_TO_BOX.keys())

# deterministic colors
if 'colors' not in st.session_state:
    st.session_state.colors = {}
for p in ALL_PARTS:
    if p not in st.session_state.colors:
        h = abs(hash(p)) % (256**3)
        r = (h >> 16) & 0xFF
        g = (h >> 8) & 0xFF
        b = h & 0xFF
        r = 80 + (r % 160)
        g = 80 + (g % 160)
        b = 80 + (b % 160)
        st.session_state.colors[p] = f"rgba({r},{g},{b},0.85)"
st.session_state.colors.setdefault("", "rgba(200,200,200,0.6)")

# -------------------- Packing utilities --------------------
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
        # orientation 1
        if L_box <= fL + 1e-9 and W_box <= fW + 1e-9:
            leftover = fr_area - (L_box * W_box)
            new = free_rects.copy()
            new.pop(i)
            if fL - L_box > 1e-9:
                new.append([fx + L_box, fy, fL - L_box, W_box])
            if fW - W_box > 1e-9:
                new.append([fx, fy + W_box, fL, fW - W_box])
            placement = [fx, fy, L_box, W_box, False]
            if best is None or leftover < best[0]:
                best = (leftover, i, new, placement)
        # orientation 2 (rotated)
        if W_box <= fL + 1e-9 and L_box <= fW + 1e-9:
            leftover = fr_area - (L_box * W_box)
            new = free_rects.copy()
            new.pop(i)
            if fL - W_box > 1e-9:
                new.append([fx + W_box, fy, fL - W_box, L_box])
            if fW - L_box > 1e-9:
                new.append([fx, fy + L_box, fL, fW - L_box])
            placement = [fx, fy, W_box, L_box, True]
            if best is None or leftover < best[0]:
                best = (leftover, i, new, placement)
    if best is None:
        return False, free_rects, [0, 0, 0, 0, False]
    return True, best[2], best[3]


def pack_one_layer(pallet, boxes, order_left):
    placed_list = []
    free_rects = [[0.0, 0.0, pallet['L'], pallet['W']]]
    names = list(order_left.keys())
    valid = [n for n in names if order_left[n] > 0]
    if not valid:
        return placed_list, order_left
    valid.sort(key=lambda n: boxes[n]['Area'], reverse=True)
    for nm in valid:
        while order_left[nm] > 0:
            dims = [boxes[nm]['L'], boxes[nm]['W']]
            ok, new_free, pos = try_place(free_rects, dims)
            if not ok:
                break
            fx, fy, L_used, W_used, rotated = pos
            free_rects = new_free
            order_left[nm] -= 1
            placed_list.append({
                'name': nm, 'x': fx, 'y': fy, 'L': L_used, 'W': W_used,
                'H': boxes[nm]['H'], 'rotated': rotated
            })
    rem_names = [n for n in names if order_left[n] > 0]
    if rem_names:
        rem_names.sort(key=lambda n: boxes[n]['Area'])
    r = 0
    while r < len(free_rects):
        fr = free_rects[r]
        filled = False
        for nm in rem_names:
            if order_left[nm] <= 0:
                continue
            dims = [boxes[nm]['L'], boxes[nm]['W']]
            ok, new_free, pos = try_place([fr], dims)
            if ok:
                fx, fy, L_used, W_used, rotated = pos
                order_left[nm] -= 1
                placed_list.append({
                    'name': nm, 'x': fx, 'y': fy, 'L': L_used, 'W': W_used,
                    'H': boxes[nm]['H'], 'rotated': rotated
                })
                free_rects.pop(r)
                free_rects.extend(new_free)
                filled = True
                break
        if not filled:
            r += 1
    return placed_list, order_left


def pack_all_pallets(pallet, boxes, order):
    pallets = []
    pallet_layers = []
    order_left = copy.deepcopy(order)
    max_pallet_height = pallet['H']
    info = build_info(boxes)
    while sum(order_left.values()) > 0:
        layer_list = []
        current_height = 0.0
        while current_height < max_pallet_height:
            remaining_boxes = [n for n, q in order_left.items() if q > 0]
            if not remaining_boxes:
                break
            placed, order_left = pack_one_layer(pallet, info, order_left)
            if not placed:
                break
            layer_list.append(placed)
            tallest_in_layer = max([b['H'] for b in placed]) if placed else 0
            current_height += tallest_in_layer
            if tallest_in_layer <= 0:
                break
            if current_height >= max_pallet_height:
                break
        if not layer_list:
            break
        pallet_layers.append(layer_list)
        pallets.append({})  # placeholder
    return pallets, pallet_layers

# -------------------- Visualization --------------------
def plot_layer_topview(pallet, layer_data, part_assignments, colors_map, scale=8, title=None):
    L = pallet['L'] * scale
    W = pallet['W'] * scale
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, y0=0, x1=L, y1=W, line=dict(width=2))
    for i, d in enumerate(layer_data):
        x0 = d['x'] * scale
        y0 = d['y'] * scale
        x1 = x0 + d['L'] * scale
        y1 = y0 + d['W'] * scale
        part = part_assignments[i] if i < len(part_assignments) else ''
        color = colors_map.get(part, "rgba(200,200,200,0.7)")
        fig.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color='black', width=1), fillcolor=color, opacity=0.9)
        label = f"{part} ({d['name']})" if part else d['name']
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label, showarrow=False,
                           font=dict(color='black', size=10), xanchor='center', yanchor='middle')
    fig.update_xaxes(showticklabels=False, range=[0, L])
    fig.update_yaxes(showticklabels=False, range=[0, W], scaleanchor='x', scaleratio=1)
    fig.update_layout(width=400, height=max(300, int(400*(pallet['W']/pallet['L']))), margin=dict(l=5,r=5,t=20,b=5))
    if title:
        fig.update_layout(title=title)
    return fig

# -------------------- PDF generators --------------------
def ddmmyyyy_hhmmss_now():
    return datetime.datetime.now().strftime("%d%m%Y_%H%M%S")


def create_summary_pdf_text(summary_per_pallet, grand_total, pallet):
    # summary only text PDF
    ts = ddmmyyyy_hhmmss_now()
    path = tempfile.NamedTemporaryFile(delete=False, suffix=f"_summary_{ts}.pdf").name
    c = canvas.Canvas(path, pagesize=A4)
    pw, ph = A4
    margin = 40
    y = ph - 60
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Pallet Packing - Summary")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Pallet Size: {pallet['L']} x {pallet['W']} in   |   Pallet Height: {pallet['H']} in")
    y -= 14
    c.drawString(margin, y, f"Generated: {datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')}")
    y -= 20

    for p_idx, pal_sum in enumerate(summary_per_pallet, start=1):
        if y < 100:
            c.showPage(); y = ph - 60
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"Pallet {p_idx}")
        y -= 14
        c.setFont("Helvetica", 10)
        for k, v in sorted(pal_sum.items()):
            if y < 80:
                c.showPage(); y = ph - 60
            c.drawString(margin+10, y, f"{k}: {v}")
            y -= 12
        y -= 6

    # grand totals
    if y < 120:
        c.showPage(); y = ph - 60
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Grand Total")
    y -= 14
    c.setFont("Helvetica", 10)
    for k, v in sorted(grand_total.items()):
        if y < 80:
            c.showPage(); y = ph - 60
        c.drawString(margin+10, y, f"{k}: {v}")
        y -= 12

    c.save()
    return path


def create_layout_pdf_visuals(assigned_layer_details, pallet, scale_for_pdf=4):
    # layout PDF: reproduce web visuals in a 5x2 grid per pallet page
    ts = ddmmyyyy_hhmmss_now()
    path = tempfile.NamedTemporaryFile(delete=False, suffix=f"_layout_{ts}.pdf").name
    c = canvas.Canvas(path, pagesize=A4)
    pw, ph = A4
    margin = 20
    # grid params: 5 cols x 2 rows
    cols = 5
    rows = 2
    thumb_w = (pw - margin*2 - (cols-1)*6) / cols
    thumb_h = (ph - margin*2 - 60 - (rows-1)*6) / rows

    for p_idx, pallet_layers in enumerate(assigned_layer_details, start=1):
        # a page for each pallet; if more than 10 layers, continue to next page(s)
        layers = pallet_layers
        i = 0
        while i < len(layers):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(margin, ph - margin - 10, f"Pallet {p_idx} - Layout (Generated {datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')})")
            # draw grid cells
            for r in range(rows):
                for co in range(cols):
                    idx = i
                    x = margin + co*(thumb_w+6)
                    y_top = ph - margin - 40 - r*(thumb_h+6)
                    # border
                    c.rect(x, y_top - thumb_h, thumb_w, thumb_h, stroke=1, fill=0)
                    if idx < len(layers):
                        layer = layers[idx]
                        # compute layer extents
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
                        # draw boxes
                        for b in layer:
                            bx = x + 4 + (b['x'] - min_x) * s
                            by_top = y_top - 4 - (b['y'] - min_y) * s
                            bw = b['L'] * s
                            bh = b['W'] * s
                            c.setFillColorRGB(0.85, 0.9, 1)
                            c.rect(bx, by_top - bh, bw, bh, stroke=1, fill=1)
                            lab = (b.get('part','') + ' (' + b['name'] + ')') if b.get('part','') else b['name']
                            c.setFont('Helvetica', 6)
                            c.drawString(bx+2, by_top - bh/2 - 3, lab[:30])
                        # title under each thumb
                        c.setFont('Helvetica-Bold', 9)
                        c.drawString(x+2, y_top - thumb_h - 10, f"Layer {idx+1}")
                    i += 1
                    if i >= len(layers) and r==rows-1 and co==cols-1:
                        break
                # end cols
            c.showPage()
        # end while i
    c.save()
    return path

# -------------------- UI: Inputs (sidebar) --------------------
with st.sidebar:
    st.header("Pallet settings & Inputs")
    pallet_L = st.number_input("Pallet length (in)", value=float(DEFAULT_PALLET['L']))
    pallet_W = st.number_input("Pallet width (in)", value=float(DEFAULT_PALLET['W']))
    pallet_H = st.number_input("Pallet height (in)", value=float(DEFAULT_PALLET['H']))
    scale = st.number_input("Scale factor (visual)", value=8, min_value=1)
    st.markdown("---")
    st.markdown("Paste orders (Part, Qty). ONLY Part numbers accepted (Box codes like AZ10 will be skipped with a warning).\nExample: E71531,870  (870 units → boxes = ceil(870/10))")
    paste_text = st.text_area("Paste Orders (one per line, comma/tab/space separated)", height=220)
    st.markdown("---")
    st.markdown("Or select parts (MOQ=10 units per box):")
    sel_parts = st.multiselect("Select part codes", options=ALL_PARTS, default=[])
    if 'part_qtys' not in st.session_state:
        st.session_state.part_qtys = {}
    for p in sel_parts:
        st.session_state.part_qtys.setdefault(p, MOQ)
        cols = st.columns([3,1,1,2])
        cols[0].write(f"{p} → {PART_TO_BOX.get(p,'')}")
        if cols[1].button("+", key=f"plus_{p}"):
            st.session_state.part_qtys[p] += MOQ
        if cols[2].button("-", key=f"minus_{p}"):
            st.session_state.part_qtys[p] = max(MOQ, st.session_state.part_qtys[p] - MOQ)
        cols[3].write(f"Units: {st.session_state.part_qtys[p]}  → Boxes: {math.ceil(st.session_state.part_qtys[p]/MOQ)}")
    st.markdown("---")
    if st.button("Clear part selections"):
        st.session_state.part_qtys = {}

# -------------------- Parse inputs --------------------
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
        st.warning(f"⚠️ Detected box code '{a_up}' in pasted input. Please paste PART numbers only. This line was skipped.")
        continue
    if a_up not in PART_TO_BOX:
        st.warning(f"⚠️ Unknown part '{a_up}' skipped.")
        continue
    try:
        qty_units = int(float(b))
    except:
        st.warning(f"⚠️ Bad qty on line '{line}' — skipped.")
        continue
    boxes_needed = math.ceil(qty_units / MOQ)
    box_code = PART_TO_BOX[a_up]
    order_counts[box_code] = order_counts.get(box_code, 0) + boxes_needed
    order_part_queue.setdefault(box_code, []).extend([a_up] * boxes_needed)

# from multiselect
for p, units in st.session_state.part_qtys.items():
    if units <= 0:
        continue
    b = PART_TO_BOX.get(p)
    boxes_needed = math.ceil(units / MOQ)
    order_counts[b] = order_counts.get(b, 0) + boxes_needed
    order_part_queue.setdefault(b, []).extend([p] * boxes_needed)

# ensure all known boxes present
for b in DEFAULT_BOXES.keys():
    order_counts.setdefault(b, 0)
    order_part_queue.setdefault(b, [])

# -------------------- Packing --------------------
boxes_def = DEFAULT_BOXES
pallet = {'L': float(pallet_L), 'W': float(pallet_W), 'H': float(pallet_H)}
pallets, layerDetails = pack_all_pallets(pallet, boxes_def, order_counts)

# assign parts FIFO
local_queues = {k: v.copy() for k, v in order_part_queue.items()}
assigned_layer_details = []
summary_per_pallet = []
grand_total = {}

for p_idx, pallet_layers in enumerate(layerDetails):
    pallet_summary = {}
    assigned_pallet_layers = []
    for layer in pallet_layers:
        assigned_layer = []
        for b in layer:
            box_code = b['name']
            qlist = local_queues.get(box_code, [])
            if qlist:
                part_assigned = qlist.pop(0)
            else:
                part_assigned = BOX_TO_PARTS.get(box_code, [''])[0] or ''
            new_b = b.copy()
            new_b['part'] = part_assigned
            assigned_layer.append(new_b)
            key = f"{part_assigned} ({box_code})" if part_assigned else box_code
            pallet_summary[key] = pallet_summary.get(key, 0) + 1
            grand_total[key] = grand_total.get(key, 0) + 1
        assigned_pallet_layers.append(assigned_layer)
    assigned_layer_details.append(assigned_pallet_layers)
    summary_per_pallet.append(pallet_summary)

# -------------------- Quick status / download buttons --------------------
if not assigned_layer_details:
    st.info("No pallets produced (orders might be zero or boxes don't fit).")
else:
    st.success(f"Packed → Total pallets: {len(assigned_layer_details)}")

    # generate PDFs and present download buttons
    layout_pdf_path = create_layout_pdf_visuals(assigned_layer_details, pallet)
    summary_pdf_path = create_summary_pdf_text(summary_per_pallet, grand_total, pallet)
    with open(layout_pdf_path, 'rb') as f:
        st.download_button("🖼 Download Layout PDF", f, file_name=f"Layout_{ddmmyyyy_hhmmss_now()}.pdf", mime="application/pdf")
    with open(summary_pdf_path, 'rb') as f:
        st.download_button("📄 Download Summary PDF", f, file_name=f"Summary_{ddmmyyyy_hhmmss_now()}.pdf", mime="application/pdf")

# -------------------- Web visuals: display ALL pallets and layers in 5x2 grid --------------------
st.markdown("## Visuals (2D top view)")
if assigned_layer_details:
    for p_idx, pallet_layers in enumerate(assigned_layer_details, start=1):
        st.subheader(f"🟫 Pallet {p_idx}")
        # flatten into figs
        layer_figs = []
        for l_idx, layer_data in enumerate(pallet_layers, start=1):
            part_assignments = [b.get('part','') for b in layer_data]
            fig = plot_layer_topview(pallet, layer_data, part_assignments, st.session_state.colors, scale=scale, title=f"Layer {l_idx}")
            layer_figs.append((l_idx, fig))

        cols_per_row = 5
        # show up to 10 layers in grid (5 cols, 2 rows)
        for row_start in range(0, min(len(layer_figs), 10), cols_per_row):
            row_layers = layer_figs[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_layers))
            for col, (l_idx, fig) in zip(cols, row_layers):
                with col:
                    st.markdown(f"**Layer {l_idx}**")
                    st.plotly_chart(fig, use_container_width=True, key=f"p{p_idx}_l{l_idx}")

# -------------------- Footer/help --------------------
st.markdown("---")
st.caption("Paste only PART numbers in the orders box (first column). Quantities are units; 1 box = 10 units (MOQ=10). If you select parts using the selector they will be added too.")
