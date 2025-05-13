# Export the final pipeline diagram with stalls and arrows as a high-resolution PNG image
fig, ax = plt.subplots(figsize=(24, 12), dpi=300)

# Draw pipeline stages
for i, (label, stages) in enumerate(zip(instructions, pipeline_cycles)):
    y = len(instructions) - i
    stage_idx = 0
    for cycle in stages:
        if cycle == '●':
            ax.add_patch(mpatches.Circle((stages[stage_idx+1], y), 0.25, color=stage_colors['●']))
        else:
            if stage_idx < 5:
                stage = stage_labels[stage_idx]
                ax.add_patch(plt.Rectangle((cycle - 0.5, y - 0.3), 1, 0.6, color=stage_colors[stage], ec='black'))
                ax.text(cycle, y, stage, va='center', ha='center', fontsize=7, fontweight='bold')
            stage_idx += 1
    ax.text(0, y, label, ha='right', va='center', fontsize=8, fontweight='bold')

# Draw bold forwarding arrows
arrow_style = dict(arrowstyle='-|>', lw=1.5, color='blue', shrinkA=2, shrinkB=2)
arrow_labels = [
    "x1→x2", "x4→x4", "x1→x2", "x4→x4", "x1→x4", "x6→x6", "x1→x2", "x4→x4"
]
for (x1, y1, x2, y2), label in zip(arrows, arrow_labels):
    if 0 <= y1 <= len(instructions) and 0 <= y2 <= len(instructions):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_style)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.3, label, color='blue', fontsize=6, ha='center', va='bottom')

# Final layout
ax.set_xlim(0, num_cycles)
ax.set_ylim(0, len(instructions) + 1)
ax.set_xticks(range(num_cycles))
ax.set_yticks([])
ax.set_xlabel("Clock Cycles", fontsize=10)
ax.set_title("Pipeline Diagram with Stalls (●) and Forwarding Arrows", fontsize=12, fontweight='bold')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the figure
image_path = "/mnt/data/pipeline_diagram_with_arrows.png"
plt.savefig(image_path)
plt.close()

image_path
