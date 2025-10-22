import streamlit as st
import plotly.graph_objects as go
import copy
from math import floor
import tempfile
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# -------------------- Default data --------------------
DEFAULT_PALLET = {'L': 42.0, 'W': 42.0, 'H': 90.0}
DEFAULT_BOX_HEIGHT = 9.0  # Standard layer height

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
    'AZ18': [48.0, 40.0, 18.0]  # AZ18 is double height (18)
}

DEFAULT_ORDER = {
    'AZ17': 47,
    'AZ13': 12,
    'AZ6': 15,
    'AZ16': 72,
    'AZ4': 1,
    'AZ3': 64,
    'AZ15': 65,
    'AZ11': 24,
    'AZ14': 3,
    'AZ10': 12,
    'AZ12': 24,
    'AZ8': 24,
    'AZ5': 47,
    'AZ7': 24,
    'AZ2': 95,
    'AZ18': 24  # AZ18 quantity
}

# -------------------- Packing utilities --------------------
def build_info(boxes):
    info = {}
    for nm, dims in boxes.items():
        L, W, H = dims
        info[nm] = {'L': L, 'W': W, 'H': H, 'Area': L * W}
    return info

def free_area(fr):
    """Return area of a free rect"""
    return fr[2] * fr[3]

def try_place(free_rects, dims):
    """
    Smart placement: for each free rect, evaluate both orientations (if they fit)
    and choose the placement that minimizes leftover free area (i.e., best fit).
    Returns (ok, new_free_rects, placement)
    placement = [x, y, L_used, W_used, rotated_flag]
    """
    best = None  # tuple: (leftover_area, index, new_free_list, placement)
    L_box, W_box = dims

    for i, fr in enumerate(free_rects):
        fx, fy, fL, fW = fr
        fr_area = fL * fW

        # Option 1: default orientation
        if L_box <= fL and W_box <= fW:
            leftover = fr_area - (L_box * W_box)
            # build new free rects after placing this orientation
            new = free_rects.copy()
            new.pop(i)
            # append resulting free rects (only if positive area)
            if fL - L_box > 1e-9:
                new.append([fx + L_box, fy, fL - L_box, W_box])
            if fW - W_box > 1e-9:
                new.append([fx, fy + W_box, fL, fW - W_box])
            placement = [fx, fy, L_box, W_box, False]
            if best is None or leftover < best[0]:
                best = (leftover, i, new, placement)

        # Option 2: rotated orientation
        if W_box <= fL and L_box <= fW:
            leftover = fr_area - (L_box * W_box)  # area same as above
            new = free_rects.copy()
            new.pop(i)
            if fL - W_box > 1e-9:
                new.append([fx + W_box, fy, fL - W_box, L_box])
            if fW - L_box > 1e-9:
                new.append([fx, fy + L_box, fL, fW - L_box])
            placement = [fx, fy, W_box, L_box, True]  # rotated True
            if best is None or leftover < best[0]:
                best = (leftover, i, new, placement)

    if best is None:
        return False, free_rects, [0, 0, 0, 0, False]
    # return best found placement
    return True, best[2], best[3]

# -------------------- Packing functions --------------------
def pack_one_layer(pallet, boxes, order_left):
    """
    Pack one 2D layer using best-fit with rotation consideration.
    Returns list of placed box dicts (with 'rotated' flag) and updated order_left.
    """
    placed_list = []
    free_rects = [[0.0, 0.0, pallet['L'], pallet['W']]]
    names = list(order_left.keys())
    valid = [n for n in names if order_left[n] > 0]
    if not valid:
        return placed_list, order_left

    # Sort by area descending for greedy placement
    valid.sort(key=lambda n: boxes[n]['Area'], reverse=True)
    for nm in valid:
        # try placing as many as possible of this box in the current layer
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

    # Try to fill remaining free rects with smaller boxes (ascending area)
    rem_names = [n for n in names if order_left[n] > 0]
    if rem_names:
        rem_names.sort(key=lambda n: boxes[n]['Area'])
    r = 0
    # iterate free rects and try to place small boxes
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
                # remove the free rect we used and extend free_rects with produced ones
                free_rects.pop(r)
                free_rects.extend(new_free)
                filled = True
                break
        if not filled:
            r += 1

    return placed_list, order_left


def pack_all_pallets(pallet, boxes, order):
    """
    Build pallets using layer stacking. Each layer packs using pack_one_layer.
    Height accounting: each layer increases the current height by the tallest box in that layer.
    Thus a box with height 18 will consume two 9-inch layers worth of height (if pallet height allows).
    """
    pallets = []
    pallet_layers = []
    order_left = copy.deepcopy(order)
    max_pallet_height = pallet['H']

    info = build_info(boxes)

    while sum(order_left.values()) > 0:
        layer_list = []
        current_height = 0.0
        # Build layers until pallet height reached
        while current_height < max_pallet_height:
            remaining_boxes = [n for n, q in order_left.items() if q > 0]
            if not remaining_boxes:
                break

            # Determine a layer pack - we allow any box heights in a layer,
            # but after packing we increment current_height by tallest in that layer.
            placed, order_left = pack_one_layer(pallet, info, order_left)
            if not placed:
                break  # can't place any more in this pallet
            layer_list.append(placed)
            tallest_in_layer = max([b['H'] for b in placed]) if placed else 0
            current_height += tallest_in_layer

            # safety: avoid infinite loops
            if tallest_in_layer <= 0:
                break

            # if next layer would exceed height, stop adding layers to this pallet
            if current_height >= max_pallet_height:
                break

        if not layer_list:
            break
        pallet_layers.append(layer_list)
        pallets.append({})  # placeholder keeps parity with original structure
    return pallets, pallet_layers

# -------------------- Visualization --------------------
def plot_layer(pallet, layer_data, scale=10, title=None):
    L = pallet['L'] * scale
    W = pallet['W'] * scale
    fig = go.Figure()
    # pallet boundary
    fig.add_shape(type='rect', x0=0, y0=0, x1=L, y1=W, line=dict(width=2))

    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b',
              '#e377c2','#7f7f7f','#bcbd22','#17becf']

    for i, d in enumerate(layer_data):
        x0 = d['x'] * scale
        y0 = d['y'] * scale
        x1 = x0 + d['L'] * scale
        y1 = y0 + d['W'] * scale
        # color scheme: rotated boxes get a distinct color tint
        if d.get('rotated', False):
            color = 'rgba(44,160,44,0.75)'  # green-ish for rotated
        else:
            color = colors[i % len(colors)]

        # highlight AZ18 with red border if present
        if d['name'] == 'AZ18':
            line = dict(color='red', width=2)
        else:
            line = dict(color='black', width=1)

        fig.add_shape(
            type='rect',
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=line,
            fillcolor=color, opacity=0.7
        )

        fig.add_annotation(
            x=(x0 + x1)/2, y=(y0 + y1)/2,
            text=d['name'],
            showarrow=False,
            font=dict(color='black', size=12),
            xanchor='center', yanchor='middle'
        )

    fig.update_xaxes(showticklabels=False, range=[0, L])
    fig.update_yaxes(showticklabels=False, range=[0, W], scaleanchor='x', scaleratio=1)
    fig.update_layout(
        width=600, height=600 * (pallet['W']/pallet['L']),
        margin=dict(l=10,r=10,t=30,b=10),
    )
    if title:
        fig.update_layout(title=title)
    return fig

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title='Pallet Packing Dashboard', layout='wide')
st.title('Pallet Packing Dashboard (Streamlit) — Smart Fit & PDF')

with st.sidebar:
    st.header('Inputs')
    pallet_L = st.number_input('Pallet length (in)', value=float(DEFAULT_PALLET['L']))
    pallet_W = st.number_input('Pallet width (in)', value=float(DEFAULT_PALLET['W']))
    pallet_H = st.number_input('Pallet height (in)', value=float(DEFAULT_PALLET['H']))
    scale = st.number_input('Scale factor (visual)', value=8)
    st.markdown('---')
    orders_text = st.text_area('Order (CSV: name,qty per line)',
                               value='\n'.join([f'{k},{v}' for k,v in DEFAULT_ORDER.items()]),
                               height=240)
    use_defaults = st.checkbox('Use default boxes definitions', value=True)

# parse orders
order = {}
for line in orders_text.splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(',') if p.strip()]
    if len(parts) >= 2:
        try:
            order[parts[0]] = int(float(parts[1]))
        except:
            order[parts[0]] = 0

# ensure all known boxes exist in order map
for k in DEFAULT_BOXES.keys():
    order.setdefault(k, 0)

boxes = DEFAULT_BOXES if use_defaults else DEFAULT_BOXES.copy()
pallet = {'L': pallet_L, 'W': pallet_W, 'H': pallet_H}

pallets, layerDetails = pack_all_pallets(pallet, boxes, order)
st.success(f'Total pallets used: {len(pallets)}')

# display and build summaries
summary_per_pallet = []  # per-pallet aggregated counts
grand_total = {}
for p_idx, layers in enumerate(layerDetails):
    st.subheader(f'Pallet {p_idx+1} — {len(layers)} layer(s)')
    nCols = 2
    cols = st.columns(nCols)
    pallet_summary = {}
    for l_idx, layer in enumerate(layers):
        fig = plot_layer(pallet, layer, scale=scale, title=f'Layer {l_idx+1}')
        col_idx = l_idx % nCols
        cols[col_idx].plotly_chart(fig, use_container_width=True, key=f'p{p_idx}_l{l_idx}')
        # accumulate counts for this layer into pallet_summary
        for b in layer:
            pallet_summary[b["name"]] = pallet_summary.get(b["name"], 0) + 1
    summary_per_pallet.append(pallet_summary)
    for k, v in pallet_summary.items():
        grand_total[k] = grand_total.get(k, 0) + v

# Verification block
st.markdown("### 🔍 Verification — Packed Totals per Box")
st.json(grand_total)

# -------------------- PDF Summary (layer-wise details) --------------------
def create_summary_pdf(layer_details, summary_per_pallet, grand_total, pallet, pallet_height):
    """
    layer_details: list of pallets -> each pallet is list of layers -> each layer is list of boxes(dict)
    summary_per_pallet: list of dicts summarizing counts per pallet (as currently built)
    """
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    c = canvas.Canvas(pdf_path, pagesize=A4)
    pw, ph = A4
    margin, y = 50, ph - 80
    now = datetime.datetime.now().strftime("%d-%b-%Y %I:%M %p")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Detailed Pallet Summary Report")
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Pallet Size: {pallet['L']} x {pallet['W']} in")
    y -= 15
    c.drawString(margin, y, f"Pallet Height: {pallet_height} in")
    y -= 15
    c.drawString(margin, y, f"Total Pallets: {len(summary_per_pallet)}")
    y -= 15
    c.drawString(margin, y, f"Generated on: {now}")
    y -= 25

    # For each pallet, list layer-wise box breakdown
    for p_idx, pallet_layers in enumerate(layer_details, start=1):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"Pallet {p_idx}")
        y -= 18
        for l_idx, layer in enumerate(pallet_layers, start=1):
            # count boxes in this layer
            layer_count = {}
            for b in layer:
                layer_count[b['name']] = layer_count.get(b['name'], 0) + 1
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin + 10, y, f"Layer {l_idx}")
            y -= 16
            c.setFont("Helvetica", 10)
            # print sorted parts
            for part, qty in sorted(layer_count.items()):
                c.drawString(margin + 25, y, f"{part}: {qty}")
                y -= 14
                if y < 100:
                    c.showPage()
                    y = ph - 80
            # layer totals
            c.setFont("Helvetica-Oblique", 10)
            c.drawString(margin + 25, y, f"Total boxes in layer: {sum(layer_count.values())}")
            y -= 18
            if y < 100:
                c.showPage()
                y = ph - 80
        # After listing layers, show pallet summary counts
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin + 10, y, "Pallet summary:")
        y -= 16
        c.setFont("Helvetica", 10)
        pallet_summary = summary_per_pallet[p_idx-1] if p_idx-1 < len(summary_per_pallet) else {}
        for part, qty in sorted(pallet_summary.items()):
            c.drawString(margin + 25, y, f"{part}: {qty}")
            y -= 14
            if y < 100:
                c.showPage()
                y = ph - 80
        c.drawString(margin + 25, y, f"Total boxes in pallet: {sum(pallet_summary.values())}")
        y -= 24
        if y < 100:
            c.showPage()
            y = ph - 80

    # Grand totals
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Grand Total Across All Pallets")
    y -= 18
    c.setFont("Helvetica", 10)
    for part, qty in sorted(grand_total.items()):
        c.drawString(margin + 20, y, f"{part}: {qty}")
        y -= 14
        if y < 100:
            c.showPage()
            y = ph - 80

    c.save()
    return pdf_path

# -------------------- PDF Download UI --------------------
st.markdown("---")
if st.button("📄 Generate Summary PDF"):
    if layerDetails:
        pdf_file = create_summary_pdf(layerDetails, summary_per_pallet, grand_total, pallet, pallet_H)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇ Download PDF Summary", f, file_name="Pallet_Summary.pdf", mime="application/pdf")
    else:
        st.warning("No pallets/layers to export to PDF.")

# -------------------- End of app --------------------
st.markdown("---")
st.caption("Smart-fit rotation: boxes are automatically rotated when the rotated orientation reduces leftover free area. Rotated boxes are shown in green on the layer visualizations.")
