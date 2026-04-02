import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(24, 11))
ax.set_xlim(0, 24)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('#F0F4F8')

# ── Palette ───────────────────────────────────────────────────────────────────
C_LLM    = '#2563EB'
C_LLM_L  = '#DBEAFE'
C_MEM    = '#059669'
C_MEM_L  = '#D1FAE5'
C_PROJ   = '#B45309'
C_PROJ_L = '#FEF3C7'
C_RWD    = '#DC2626'
C_RWD_L  = '#FEE2E2'
C_RL     = '#7C3AED'
C_RL_L   = '#EDE9FE'
C_DOC    = '#475569'
C_DOC_L  = '#E2E8F0'
C_TRAIN  = '#EFF6FF'
C_INF    = '#FFFBEB'

# ── Helpers ───────────────────────────────────────────────────────────────────
def shadow(ax, x, y, w, h, r=0.1):
    ax.add_patch(FancyBboxPatch(
        (x+0.06, y-0.06), w, h,
        boxstyle=f'round,pad=0.04,rounding_size={r}',
        fc='#94A3B8', ec='none', alpha=0.18, zorder=2))

def fbox(ax, x, y, w, h, fc, ec, lw=1.5, ls='solid', r=0.1, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.04,rounding_size={r}',
        fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=zorder))

def txt(ax, x, y, s, fs=9, bold=False, color='#1E293B',
        italic=False, ha='center', va='center', zorder=6):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, zorder=zorder,
            fontweight='bold' if bold else 'normal',
            style='italic' if italic else 'normal',
            multialignment='center')

def hdr_box(ax, x, y, w, h, title, fc_body, fc_head, ec, lw=2.0, r=0.12):
    """Box with solid colored header strip."""
    head_h = 0.52
    shadow(ax, x, y, w, h, r)
    fbox(ax, x, y, w, h, fc=fc_body, ec=ec, lw=lw, r=r, zorder=3)
    pad = 0.03
    ax.add_patch(plt.Rectangle(
        (x+pad, y+h-head_h), w-2*pad, head_h-pad,
        fc=fc_head, ec='none', zorder=4))
    txt(ax, x+w/2, y+h-head_h/2, title, fs=9.5, bold=True, color='white', zorder=5)

def simple_box(ax, x, y, w, h, title, sub=None,
               fc='white', ec='#334155', lw=1.6, fs=9.0, bold=True,
               tc='#1E293B', r=0.1):
    shadow(ax, x, y, w, h, r)
    fbox(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=r, zorder=3)
    cy = y + h/2 + (0.14 if sub else 0)
    txt(ax, x+w/2, cy, title, fs=fs, bold=bold, color=tc)
    if sub:
        txt(ax, x+w/2, y+h/2-0.22, sub, fs=fs-1.5, color=tc, italic=True)

def proj_box(ax, x, y, w, h, line1, line2=''):
    shadow(ax, x, y, w, h, r=0.08)
    fbox(ax, x, y, w, h, fc=C_PROJ_L, ec=C_PROJ, lw=1.8, r=0.08, zorder=3)
    if line2:
        txt(ax, x+w/2, y+h*0.67, line1, fs=7.5, bold=True,  color='#78350F')
        txt(ax, x+w/2, y+h*0.28, line2, fs=6.8, italic=True, color='#92400E')
    else:
        txt(ax, x+w/2, y+h/2,    line1, fs=7.5, bold=True,  color='#78350F')

def arr(ax, x1, y1, x2, y2, color='#334155', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'), zorder=7)

def line_arr(ax, xs, ys, color, lw=2.0):
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=6,
            solid_capstyle='round', solid_joinstyle='round')
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=7)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS
# ═══════════════════════════════════════════════════════════════════════════════
fbox(ax, 0.15, 0.15, 23.7, 10.65, fc='#F8FAFC', ec='#CBD5E1', lw=2, r=0.25, zorder=0)
fbox(ax, 0.30, 3.60, 23.4, 6.95,  fc=C_TRAIN,   ec='#93C5FD', lw=1.5, ls='dashed', r=0.2, zorder=1)
fbox(ax, 0.30, 0.25, 23.4, 3.15,  fc=C_INF,     ec='#FCD34D', lw=1.5, ls='dashed', r=0.2, zorder=1)

# Title
txt(ax, 12.0, 10.62,
    r'$\mathcal{R}$eMemorize: Adaptive Memory Management Framework',
    fs=14, bold=True, color='#0F172A', zorder=7)

# Numbered section badges
for cx, cy, num, label, color in [
    (0.77, 9.98, '1', 'Training Phase  —  Policy Learning', '#2563EB'),
    (0.77, 3.12, '2', 'Inference Phase  —  Policy Frozen',  '#D97706'),
]:
    ax.add_patch(plt.Circle((cx, cy), 0.22, color=color, zorder=6))
    txt(ax, cx, cy, num, fs=11, bold=True, color='white', zorder=7)
    txt(ax, cx+0.42, cy, label, fs=10.5, bold=True, color=color, ha='left', zorder=7)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING — FORWARD ROW  (boxes centered around y ≈ 7.2 – 9.6)
# ═══════════════════════════════════════════════════════════════════════════════
# 1. Clinical Document
simple_box(ax, 0.4, 7.3, 1.75, 2.0, 'Clinical\nDocument',
           sub='(Long & noisy)', fc='#E8F0FE', ec='#3B4FA8', fs=8.5)

# 2. Phase 1 LLM
simple_box(ax, 2.4, 7.3, 2.05, 2.0, 'Phase 1\nLLM',
           sub='token representations', fc=C_LLM_L, ec=C_LLM, lw=2, fs=8.5)

# 3. Key / Value Encoders
proj_box(ax, 4.70, 8.22, 1.48, 0.82, 'Key Encoder',   'token → key')
proj_box(ax, 4.70, 7.30, 1.48, 0.82, 'Value Encoder', 'token → value')

# 4. Memory Manager (header box + internal sub-boxes)
MM_X, MM_Y, MM_W, MM_H = 6.42, 6.70, 3.28, 2.90
hdr_box(ax, MM_X, MM_Y, MM_W, MM_H, 'Memory Manager', C_MEM_L, C_MEM, C_MEM, lw=2.2, r=0.14)

for label, yt in [
    ('Write Transform  —  encode value into memory cell', MM_Y + 2.05),
    ('Write Gate  —  how much to update memory',          MM_Y + 1.43),
]:
    fbox(ax, MM_X+0.16, yt, MM_W-0.32, 0.50, fc='#A7F3D0', ec=C_MEM, lw=1, r=0.07, zorder=4)
    txt(ax, MM_X+MM_W/2, yt+0.25, label, fs=7.2, color='#065F46', zorder=5)

fbox(ax, MM_X+0.16, MM_Y+0.56, MM_W-0.32, 0.74, fc='#A7F3D0', ec=C_MEM, lw=1, r=0.07, zorder=4)
txt(ax, MM_X+MM_W/2, MM_Y+0.98, 'GRU-style Update', fs=7.5, bold=True, color='#065F46', zorder=5)
txt(ax, MM_X+MM_W/2, MM_Y+0.65, 'interpolate old memory with new cell',
    fs=6.8, italic=True, color='#065F46', zorder=5)

# 5. Memory Decoder / Read Gate
proj_box(ax, 9.95, 8.22, 1.52, 0.82, 'Memory Decoder', 'memory → hidden')
proj_box(ax, 9.95, 7.30, 1.52, 0.82, 'Read Gate',      'how much to inject')

# 6. Phase 2 LLM
PH2_X, PH2_Y, PH2_W, PH2_H = 11.72, 6.85, 2.65, 2.55
hdr_box(ax, PH2_X, PH2_Y, PH2_W, PH2_H, 'Phase 2  LLM', C_LLM_L, C_LLM, C_LLM, r=0.12)
txt(ax, PH2_X+PH2_W/2, PH2_Y+1.82, 'Decoding + live memory',         fs=8.0, color='#1E3A5F', zorder=5)
txt(ax, PH2_X+PH2_W/2, PH2_Y+1.25, 'memory initialized from Phase 1', fs=7.2, italic=True, color='#059669', zorder=5)
txt(ax, PH2_X+PH2_W/2, PH2_Y+0.75, 'injected at each decoding step',  fs=7.0, italic=True, color='#334155', zorder=5)
txt(ax, PH2_X+PH2_W/2, PH2_Y+0.30, 'memory updated after each token', fs=6.8, italic=True, color='#475569', zorder=5)

# 7. Provisional Summary
simple_box(ax, 14.62, 7.3, 1.88, 1.95, 'Provisional\nSummary',
           sub='[Training only]', fc='#F0F9FF', ec='#0369A1', lw=1.8, fs=8.5, tc='#0C4A6E')

# 8. Reward Function
RWD_X, RWD_Y, RWD_W, RWD_H = 16.75, 6.70, 6.85, 2.90
hdr_box(ax, RWD_X, RWD_Y, RWD_W, RWD_H, 'Reward Function', C_RWD_L, C_RWD, C_RWD, lw=2, r=0.14)
for label, yy in [
    (r'$R_{\rm factual}$   —   NLI entailment  (DeBERTa-v3)',       8.83),
    (r'$R_{\rm coherence}$  —   sentence cosine similarity',          8.41),
    (r'$R_{\rm complete}$  —   soft recall over source sentences',    7.99),
    (r'$-R_{\rm halluc.}$   —   NLI contradiction probability',       7.57),
]:
    ax.text(RWD_X+0.24, yy, label, fontsize=8.2, color='#7F1D1D', va='center', zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING — FEEDBACK ROW  (y ≈ 4.2 – 5.2)
# ═══════════════════════════════════════════════════════════════════════════════
simple_box(ax, 11.72, 4.20, 2.65, 0.98, 'Dense Reward',
           sub='quality + memory efficiency',
           fc=C_RWD_L, ec=C_RWD, bold=False, fs=8.2)

simple_box(ax, 14.62, 4.20, 2.38, 0.98, 'RL Policy\nOptimizer',
           sub='GRPO + PPO,  KL δ=0.02',
           fc=C_RL_L, ec=C_RL, lw=2, bold=True, fs=8.2)

simple_box(ax, 17.25, 4.20, 2.70, 0.98, 'Training\nObjective',
           sub='summary loss + RL + aux',
           fc='#F5F3FF', ec=C_RL, bold=False, fs=8.2)

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ROW  (y ≈ 0.85 – 2.65)
# ═══════════════════════════════════════════════════════════════════════════════
simple_box(ax, 0.4,   0.9, 1.75, 1.65, 'Clinical\nDocument',
           fc='#E8F0FE', ec='#3B4FA8', bold=True, fs=8.5)

simple_box(ax, 2.4,   0.9, 2.05, 1.65, 'LLM\n(weights frozen)',
           fc=C_LLM_L, ec=C_LLM, lw=2, bold=True, fs=8.5)

proj_box(ax, 4.70, 0.9, 1.48, 1.65, 'Key Encoder', 'Value Encoder')

hdr_box(ax, 6.42, 0.80, 3.28, 1.85, 'Memory Manager',
        C_MEM_L, C_MEM, C_MEM, lw=2.0, r=0.12)
txt(ax, 6.42+1.64, 0.80+1.12, 'policy frozen', fs=7.5, italic=True, color='#065F46', zorder=5)
txt(ax, 6.42+1.64, 0.80+0.60, 'memory evolves with input', fs=7.0, italic=True, color='#065F46', zorder=5)

proj_box(ax, 9.95, 0.9, 1.52, 1.65, 'Memory Decoder', 'Read Gate')

simple_box(ax, 11.72, 0.9, 2.65, 1.65, 'LLM\n(single forward pass)',
           fc=C_LLM_L, ec=C_LLM, lw=2, bold=True, fs=8.5)

simple_box(ax, 14.62, 0.9, 2.55, 1.65, 'Final Clinical\nSummary',
           fc='#F0FDF4', ec='#15803D', lw=2.2, bold=True, fs=9.0, tc='#14532D')

# ═══════════════════════════════════════════════════════════════════════════════
# ARROWS — Training forward pass
# ═══════════════════════════════════════════════════════════════════════════════
arr(ax,  2.15,  8.30,  2.40,  8.30)          # Doc → Phase1
arr(ax,  4.45,  8.63,  4.70,  8.63)          # Phase1 → Key Enc
arr(ax,  4.45,  7.71,  4.70,  7.71)          # Phase1 → Val Enc
arr(ax,  6.18,  8.63,  6.42,  8.63)          # Key Enc → MM
arr(ax,  6.18,  7.71,  6.42,  7.71)          # Val Enc → MM
arr(ax,  9.70,  8.63,  9.95,  8.63)          # MM → Mem Dec
arr(ax,  9.70,  7.71,  9.95,  7.71)          # MM → Read Gate
arr(ax, 11.47,  8.63, 11.72,  8.63)          # Mem Dec → Phase2
arr(ax, 11.47,  7.71, 11.72,  7.71)          # Read Gate → Phase2
arr(ax, 14.37,  8.12, 14.62,  8.12)          # Phase2 → Prov Summary
arr(ax, 16.50,  8.12, 16.75,  8.12)          # Prov Sum → Reward

# ── Memory state handoff: Phase 1 builds → Phase 2 reads ─────────────────────
# Dashed horizontal line just above the Memory Manager + projection boxes
ARC_Y = 9.68
ax.plot([9.70, 11.58], [ARC_Y, ARC_Y],
        color='#047857', lw=1.8, linestyle='--', zorder=8,
        solid_capstyle='round')
ax.annotate('', xy=(11.72, ARC_Y), xytext=(11.58, ARC_Y),
            arrowprops=dict(arrowstyle='->', color='#047857', lw=1.8), zorder=8)
# small vertical connectors from boxes up to the line
ax.plot([9.70, 9.70], [9.60, ARC_Y], color='#047857', lw=1.4, linestyle=':', zorder=7)
ax.plot([11.72, 11.72], [ARC_Y, 9.40], color='#047857', lw=1.4, linestyle=':', zorder=7)
# label centered on the dashed line
fbox(ax, 9.95, ARC_Y - 0.14, 1.60, 0.28, fc='#ECFDF5', ec='#047857', lw=1.0, r=0.06, zorder=9)
txt(ax, 10.75, ARC_Y, 'memory state handoff',
    fs=7.2, bold=True, italic=False, color='#047857', zorder=10)

# Reward → Dense Reward (drop down, go left)
line_arr(ax, [20.17, 20.17, 13.045, 13.045],
             [6.70,   5.52,   5.52,   5.18], C_RWD)

# Dense Reward → RL Opt → Training Obj
arr(ax, 14.37, 4.69, 14.62, 4.69)
arr(ax, 17.00, 4.69, 17.25, 4.69)

# Feedback: RL Opt → Write Gate (GRPO)
line_arr(ax, [15.81, 15.81, 8.06, 8.06],
             [ 4.20,  3.80,  3.80, 6.70], C_RL)
txt(ax, 9.5, 3.91, 'Updates Write Gate  (GRPO)',
    fs=7.5, color=C_RL, italic=True, zorder=7)

# Feedback: Training Obj → all interface layers (AdamW)
line_arr(ax, [18.60, 18.60, 7.06, 7.06],
             [ 4.20,  3.64,  3.64, 6.70], C_MEM)
txt(ax, 14.5, 3.72,
    'Updates: Write Transform, Key/Value Encoders, Memory Decoder, Read Gate  (AdamW)',
    fs=7.0, color=C_MEM, italic=True, zorder=7)

# ═══════════════════════════════════════════════════════════════════════════════
# ARROWS — Inference
# ═══════════════════════════════════════════════════════════════════════════════
for x1, x2, y in [(2.15, 2.40, 1.725), (4.45, 4.70, 1.725),
                   (6.18, 6.42, 1.725), (9.70, 9.95, 1.725),
                   (11.47, 11.72, 1.725), (14.37, 14.62, 1.725)]:
    arr(ax, x1, y, x2, y)

# ═══════════════════════════════════════════════════════════════════════════════
# Legend
# ═══════════════════════════════════════════════════════════════════════════════
ax.legend(handles=[
    mpatches.Patch(fc=C_LLM_L,  ec=C_LLM,  label='LLM Backbone (frozen)'),
    mpatches.Patch(fc=C_MEM_L,  ec=C_MEM,  label='Memory Manager'),
    mpatches.Patch(fc=C_PROJ_L, ec=C_PROJ, label='Interface Layers'),
    mpatches.Patch(fc=C_RWD_L,  ec=C_RWD,  label='Reward Signal'),
    mpatches.Patch(fc=C_RL_L,   ec=C_RL,   label='RL Optimizer'),
], loc='lower right', bbox_to_anchor=(0.998, 0.002),
   fontsize=8.5, framealpha=0.95, ncol=5, handlelength=1.5,
   edgecolor='#CBD5E1')

plt.tight_layout(pad=0.2)
out = '/home/lizapiya/FLP_ReMemorizeProject/Images/Rememorize_figure1.pdf'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out,                          bbox_inches='tight', dpi=200)
plt.savefig(out.replace('.pdf', '.png'),  bbox_inches='tight', dpi=300)
plt.savefig(out.replace('.pdf', '.svg'),  bbox_inches='tight', format='svg')
print('Saved:', out)
print('Saved:', out.replace('.pdf', '.png'))
print('Saved:', out.replace('.pdf', '.svg'))
