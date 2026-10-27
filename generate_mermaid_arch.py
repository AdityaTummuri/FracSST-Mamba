import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_mermaid_architecture_diagram(output_paths):
    # Set up figure with standard printable aspect ratio (high resolution)
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 85)
    ax.axis('off')

    # Color palette - Mermaid Printable Academic Theme (Clean white/light pastel)
    subgraph_bg = '#f8fafc'
    subgraph_border = '#64748b'
    
    # Node styles (Mermaid pastel fills with crisp borders)
    c_input_bg = '#f0fdf4'      # soft green
    c_input_border = '#16a34a'
    
    c_dsp_bg = '#eff6ff'        # soft blue
    c_dsp_border = '#2563eb'
    
    c_dsp_sub_bg = '#fefce8'    # soft yellow/amber for alpha search
    c_dsp_sub_border = '#ca8a04'
    
    c_spec_bg = '#e0e7ff'       # indigo accent for spectrogram
    c_spec_border = '#4338ca'
    
    c_mamba_bg = '#faf5ff'      # soft purple
    c_mamba_border = '#7c3aed'
    
    c_head_bg = '#fdf2f8'       # soft pink
    c_head_border = '#db2777'
    
    c_out_bg = '#ecfdf5'        # light emerald
    c_out_border = '#059669'

    text_main = '#0f172a'
    text_sub = '#334155'
    arrow_color = '#1e293b'

    def draw_node(x, y, w, h, title, subtitle="", bg='#ffffff', border='#334155', rx=1.2, lw=1.4):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={rx}",
                                     facecolor=bg, edgecolor=border, linewidth=lw, zorder=3)
        ax.add_patch(box)
        
        cx = x + w / 2.0
        if subtitle:
            cy_t = y + h * 0.65
            cy_s = y + h * 0.32
            ax.text(cx, cy_t, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_main, zorder=4)
            ax.text(cx, cy_s, subtitle, ha='center', va='center', fontsize=8, color=text_sub, zorder=4, linespacing=1.2)
        else:
            cy = y + h / 2.0
            ax.text(cx, cy, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_main, zorder=4)
        return (x, y, w, h)

    def draw_subgraph(x, y, w, h, title):
        # Mermaid cluster box
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                     facecolor=subgraph_bg, edgecolor=subgraph_border, linestyle='--', linewidth=1.2, zorder=1)
        ax.add_patch(box)
        
        # Mermaid full-width cluster header
        pill = patches.FancyBboxPatch((x + 0.8, y + h - 3.8), w - 1.6, 3.2, boxstyle="round,pad=0.1,rounding_size=0.5",
                                      facecolor='#e2e8f0', edgecolor=subgraph_border, linewidth=0.8, zorder=2)
        ax.add_patch(pill)
        ax.text(x + w/2.0, y + h - 2.2, title, ha='center', va='center', fontsize=9, fontweight='bold', color='#1e293b', zorder=3)

    def draw_arrow(x1, y1, x2, y2, label="", label_pos=0.5, side='top'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=1.5, mutation_scale=12),
                    zorder=2)
        if label:
            lx = x1 + (x2 - x1) * label_pos
            ly = y1 + (y2 - y1) * label_pos
            offset_y = 1.6 if side == 'top' else -1.6
            ax.text(lx, ly + offset_y, label, ha='center', va='center', fontsize=7.5,
                    fontweight='semibold', color='#1e293b',
                    bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.3", fc="#ffffff", ec="#94a3b8", lw=0.8),
                    zorder=5)

    # -------------------------------------------------------------
    # DIAGRAM TITLE
    # -------------------------------------------------------------
    ax.text(75, 81.8, "FracSST-Mamba: End-to-End System Architecture",
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0f172a')
    ax.text(75, 79.0, "Discrete Fractional Synchrosqueezing (DSP Front-End) + 4× Mamba Selective State-Space Classifier",
            ha='center', va='center', fontsize=9, color='#475569', style='italic')

    # -------------------------------------------------------------
    # SECTION 1: RAW RADAR INPUT
    # -------------------------------------------------------------
    draw_subgraph(3, 12, 23, 63, "Input Signal")
    
    draw_node(5.5, 48, 18, 13, "CW Radar Rx", "5.8 GHz Carrier\nBaseband Sampling", bg=c_input_bg, border=c_input_border)
    draw_node(5.5, 22, 18, 15, "Raw I/Q Signal", "In-phase & Quad.\n4000 samples @ 4 kHz\ns(t) = I(t) + j·Q(t)", bg=c_input_bg, border=c_input_border)
    draw_arrow(14.5, 48, 14.5, 37, label="demod")

    # Arrow from Input to DSP
    draw_arrow(23.5, 29.5, 29.5, 29.5, label="s(t)")

    # -------------------------------------------------------------
    # SECTION 2: DSP FRONT-END (FracSST)
    # -------------------------------------------------------------
    draw_subgraph(29.5, 12, 50, 63, "DSP Front-End: FracSST Pipeline")

    # 1. Optimal Alpha Search (Top branch)
    draw_node(32.5, 52, 26, 12, "Rényi Entropy Search", "Scan 31 candidate α ∈ [0, 1.5]\nOrder-3 Rényi Entropy\nSelect α* = argmin H_α", bg=c_dsp_sub_bg, border=c_dsp_sub_border)
    
    # 2. FrFT (Main path)
    draw_node(32.5, 23.5, 26, 12, "Discrete FrFT (α*)", "Ozaktas Fast Chirp Algorithm\n2× sinc-interp + chirp conv", bg=c_dsp_bg, border=c_dsp_border)

    # Input branching into alpha search & FrFT
    ax.plot([26.5, 26.5], [29.5, 58], color=arrow_color, lw=1.5, zorder=2)
    draw_arrow(26.5, 58, 32.5, 58, label="")
    draw_arrow(58.5, 58, 45.5, 35.5, label="optimal α*", label_pos=0.5, side='top')

    # 3. SST Reassignment
    draw_node(62, 23.5, 15.5, 12, "1D SST", "Phase reassignment\nEnergy squeezing\nInstantaneous Freq.", bg=c_dsp_bg, border=c_dsp_border)
    draw_arrow(58.5, 29.5, 62, 29.5, label="rotated")

    # 4. Sharpened Representation Output
    draw_node(40, 14.5, 35, 7.5, "Sharpened Micro-Doppler", "Time-Frequency Spectrogram [256 × 256]", bg=c_spec_bg, border=c_spec_border)
    draw_arrow(69.75, 23.5, 69.75, 22, label="")

    # Arrow from DSP to Mamba
    draw_arrow(75, 18.25, 86, 18.25, label="Spectrogram [256, 256]")

    # -------------------------------------------------------------
    # SECTION 3: DEEP LEARNING BACK-END (MambaClassifier)
    # -------------------------------------------------------------
    draw_subgraph(83, 12, 44, 63, "Mamba Sequence Classifier")

    # Linear Projection
    draw_node(86, 14.5, 38, 7.5, "Linear Projection", "Maps Frequency: 256 → 128 dimensions", bg=c_mamba_bg, border=c_mamba_border)
    draw_arrow(105, 22, 105, 25.5, label="[T=256, D=128]")

    # 4x Mamba Blocks
    draw_node(86, 25.5, 38, 17.5, "4× Stacked Mamba Blocks", "• LayerNorm (128 dims)\n• Selective SSM Scan (Linear O(T) complexity)\n• Discretization: Δ, A, B, C parameters\n• Dropout (0.2) + Residual Skip Connection", bg=c_mamba_bg, border=c_mamba_border)
    draw_arrow(105, 43, 105, 46.5, label="[T=256, D=128]")

    # Global Average Pooling
    draw_node(86, 46.5, 38, 7.5, "Global Average Pooling", "Temporal Average: [256, 128] → [128]", bg=c_mamba_bg, border=c_mamba_border)
    draw_arrow(105, 54, 105, 57.5, label="[128]")

    # Classification MLP Head
    draw_node(86, 57.5, 38, 9.5, "MLP Classifier Head", "Linear(128→64) → SiLU → Dropout(0.2) → Linear(64→3)", bg=c_head_bg, border=c_head_border)

    # Arrow from Classifier to Output
    draw_arrow(124, 62.25, 131, 62.25, label="Logits")

    # -------------------------------------------------------------
    # SECTION 4: OUTPUT PREDICTIONS
    # -------------------------------------------------------------
    draw_subgraph(131, 12, 16, 63, "Predictions")
    
    draw_node(133, 53, 12, 11, "Walking", "F1: 0.95\n38/38 Correct\n(Continuous)", bg=c_out_bg, border=c_out_border)
    draw_node(133, 37, 12, 11, "Sitting", "F1: 0.70\n15/23 Correct\n(Deceleration)", bg=c_out_bg, border=c_out_border)
    draw_node(133, 21, 12, 11, "Falling", "F1: 0.77\n13/21 Correct\n(Impulsive)", bg=c_out_bg, border=c_out_border)

    # Router line to 3 outputs
    ax.plot([127.5, 127.5], [62.25, 26.5], color=arrow_color, lw=1.5, zorder=2)
    draw_arrow(127.5, 58.5, 133, 58.5, label="")
    draw_arrow(127.5, 42.5, 133, 42.5, label="")
    draw_arrow(127.5, 26.5, 133, 26.5, label="")

    # -------------------------------------------------------------
    # FOOTER SUMMARY / METRICS BADGE (Printable)
    # -------------------------------------------------------------
    badge_box = patches.FancyBboxPatch((3, 2.5), 144, 7.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                                       facecolor='#ffffff', edgecolor='#64748b', linewidth=1.0, zorder=2)
    ax.add_patch(badge_box)
    
    ax.text(6, 6.25, "Evaluation Summary:", fontsize=8.5, fontweight='bold', color='#0f172a', va='center')
    ax.text(28, 6.25, "Test Accuracy: 80.49%", fontsize=8.5, fontweight='bold', color='#047857', va='center')
    ax.text(49, 6.25, "|", fontsize=8.5, color='#cbd5e1', va='center')
    ax.text(52, 6.25, "Macro F1: 0.8045", fontsize=8.5, fontweight='bold', color='#047857', va='center')
    ax.text(70, 6.25, "|", fontsize=8.5, color='#cbd5e1', va='center')
    ax.text(73, 6.25, "Total Parameters: ~422K", fontsize=8.5, color='#334155', va='center')
    ax.text(94, 6.25, "|", fontsize=8.5, color='#cbd5e1', va='center')
    ax.text(97, 6.25, "Evaluation Protocol: Leave-Subjects-Out (9 Unseen Subjects)", fontsize=8.5, color='#334155', va='center')

    plt.tight_layout()
    for path in output_paths:
        plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()

if __name__ == '__main__':
    paths = [
        '/home/nuker/Documents/projects/FracSST-Mamba/scratch_arch_test.png',
        '/home/nuker/Documents/projects/FracSST-Mamba/assets/architecture.png',
        '/home/nuker/.gemini/antigravity-ide/brain/b41227f3-f243-4e86-bb13-6f20433f779d/architecture.png'
    ]
    create_mermaid_architecture_diagram(paths)
