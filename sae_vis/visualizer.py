"""
Visualizer: create image-based visualizations of SAE feature analysis.
FIXED VERSION with comprehensive text sanitization to prevent LaTeX parsing errors.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import re
from typing import Dict, List, Optional
from pathlib import Path
import seaborn as sns
from real_activations import FeatureStats

# Color scheme constants
REAL_COLOR = 'lightcoral'      # Red for real data
SYNTHETIC_COLOR = 'lightblue'   # Blue for synthetic data


def sanitize_text_for_matplotlib(text: str) -> str:
    """
    Sanitize text to prevent matplotlib from interpreting it as LaTeX.
    
    Args:
        text: Raw text that might contain special characters
        
    Returns:
        Sanitized text safe for matplotlib display
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Replace problematic characters that trigger LaTeX parsing
    text = text.replace('$', '\\$')  # Escape dollar signs
    text = text.replace('\\', '\\\\')  # Escape backslashes (but do this after $ replacement)
    text = text.replace('_', '\\_')   # Escape underscores (subscript in LaTeX)
    text = text.replace('^', '\\^')   # Escape carets (superscript in LaTeX)
    text = text.replace('{', '\\{')   # Escape braces
    text = text.replace('}', '\\}')   # Escape braces
    text = text.replace('#', '\\#')   # Escape hashes
    text = text.replace('&', '\\&')   # Escape ampersands
    text = text.replace('%', '\\%')   # Escape percent signs
    
    # Handle newlines and special whitespace
    text = text.replace('\n', '\\n')
    text = text.replace('\t', '\\t')
    text = text.replace('\r', '\\r')
    
    # Truncate very long strings
    if len(text) > 50:
        text = text[:47] + "..."
    
    return text


def safe_matplotlib_text(ax, x, y, text, **kwargs):
    """
    Safely add text to matplotlib axis with automatic sanitization.
    
    Args:
        ax: matplotlib axis
        x, y: coordinates
        text: text to display
        **kwargs: additional arguments for ax.text()
    """
    # Sanitize the text
    safe_text = sanitize_text_for_matplotlib(text)
    
    return ax.text(x, y, safe_text, **kwargs)


def configure_matplotlib_safe():
    """Configure matplotlib to be safe from LaTeX parsing errors"""
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'


def visualize_features(feature_stats: Dict[int, FeatureStats], output_dir: str = "visualizations",
                      analysis_type: str = "analysis", max_features_per_plot: int = 6) -> List[str]:
    """
    Create image visualizations for feature analysis.
    
    Args:
        feature_stats: Dict mapping feature_idx -> FeatureStats
        output_dir: Directory to save images
        analysis_type: "real" or "synthetic" for labeling
        max_features_per_plot: Maximum features per summary plot
        
    Returns:
        List of generated image paths
    """
    configure_matplotlib_safe()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Create individual feature plots
    for feature_idx, stats in feature_stats.items():
        print(f"Creating visualization for feature {feature_idx}...")
        
        fig_path = output_path / f"feature_{feature_idx}_{analysis_type}.png"
        _create_single_feature_plot(stats, fig_path, analysis_type)
        generated_files.append(str(fig_path))
    
    # Create summary plots
    feature_indices = list(feature_stats.keys())
    for i in range(0, len(feature_indices), max_features_per_plot):
        batch_indices = feature_indices[i:i+max_features_per_plot]
        batch_stats = {idx: feature_stats[idx] for idx in batch_indices}
        
        summary_path = output_path / f"summary_{i//max_features_per_plot}_{analysis_type}.png"
        _create_summary_plot(batch_stats, summary_path, analysis_type)
        generated_files.append(str(summary_path))
    
    # Create overall statistics plot
    stats_path = output_path / f"statistics_{analysis_type}.png"
    _create_statistics_plot(feature_stats, stats_path, analysis_type)
    generated_files.append(str(stats_path))
    
    print(f"Generated {len(generated_files)} visualization images in {output_dir}")
    return generated_files


def compare_real_vs_synthetic(real_stats: Dict[int, FeatureStats], 
                            synthetic_stats: Dict[int, FeatureStats],
                            output_dir: str = "comparison") -> List[str]:
    """Create combined real+synthetic visualizations for comparison"""
    configure_matplotlib_safe()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Find common features
    common_features = set(real_stats.keys()) & set(synthetic_stats.keys())
    
    # Create combined feature plots
    for feature_idx in common_features:
        print(f"Creating combined visualization for feature {feature_idx}...")
        
        fig_path = output_path / f"feature_{feature_idx}_combined.png"
        _create_combined_feature_plot(real_stats[feature_idx], synthetic_stats[feature_idx], fig_path)
        generated_files.append(str(fig_path))
    
    print(f"Generated {len(generated_files)} combined visualization images in {output_dir}")
    return generated_files


def _create_single_feature_plot(stats: FeatureStats, output_path: Path, analysis_type: str):
    """Create detailed plot for a single feature"""
    configure_matplotlib_safe()
    
    fig = plt.figure(figsize=(16, 10))
    
    # Choose color based on analysis type
    color = REAL_COLOR if analysis_type == "real" else SYNTHETIC_COLOR
    
    # Main title
    title_text = f"Feature {stats.feature_idx} - {analysis_type.title()} Analysis"
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    
    # Create more compact 3x2 grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1, 1.5], width_ratios=[2, 1])
    
    # Feature statistics (top row, spanning both columns)
    ax_stats = fig.add_subplot(gs[0, :])
    _plot_feature_stats_list(ax_stats, stats)
    
    # Activation distribution (middle left)
    ax_dist = fig.add_subplot(gs[1, 0])
    _plot_activation_histogram(ax_dist, stats, title="Activation Distribution", color=color)
    
    # Logit effects (middle right)
    ax_logits = fig.add_subplot(gs[1, 1])
    _plot_logit_effects(ax_logits, stats)
    
    # Top firing tokens (bottom row, spanning both columns) - now large detailed view
    ax_tokens = fig.add_subplot(gs[2, :])
    _plot_top_tokens(ax_tokens, stats, color=color)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _create_combined_feature_plot(real_stats: FeatureStats, synthetic_stats: FeatureStats, output_path: Path):
    """Create combined real+synthetic plot for a single feature"""
    configure_matplotlib_safe()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Main title
    title_text = f"Feature {real_stats.feature_idx} - Real vs Synthetic Analysis"
    fig.suptitle(title_text, fontsize=18, fontweight='bold')
    
    # Create more compact grid layout - moved everything up
    gs = fig.add_gridspec(4, 2, height_ratios=[0.25, 0.8, 0.8, 1.5], width_ratios=[2, 1])
    
    # === REAL ANALYSIS SECTION (RED) ===
    ax_real_title = fig.add_subplot(gs[0, :])
    safe_matplotlib_text(ax_real_title, 0.5, 0.5, "REAL DATA ANALYSIS", ha='center', va='center', 
                        fontsize=14, fontweight='bold', 
                        bbox=dict(boxstyle='round', facecolor=REAL_COLOR, alpha=0.7))
    ax_real_title.axis('off')
    
    ax_real_dist = fig.add_subplot(gs[1, 0])
    _plot_activation_histogram(ax_real_dist, real_stats, title="Real Activation Distribution", color=REAL_COLOR)
    
    ax_real_stats = fig.add_subplot(gs[1, 1])
    _plot_feature_stats_list(ax_real_stats, real_stats)
    
    # === SYNTHETIC ANALYSIS SECTION (BLUE) ===
    ax_synth_dist = fig.add_subplot(gs[2, 0])
    _plot_activation_histogram(ax_synth_dist, synthetic_stats, title="Synthetic Activation Distribution", color=SYNTHETIC_COLOR)
    
    ax_synth_logits = fig.add_subplot(gs[2, 1])
    _plot_logit_effects(ax_synth_logits, synthetic_stats)
    
    # Large token info section (spans full width)
    ax_tokens = fig.add_subplot(gs[3, :])
    _plot_large_token_comparison(ax_tokens, real_stats, synthetic_stats)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _plot_large_token_comparison(ax, real_stats: FeatureStats, synthetic_stats: FeatureStats):
    """Plot top 20 tokens as clean lists with sub-word token awareness"""
    ax.set_title("Top 20 Firing Tokens: Real vs Synthetic", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add explanation text
    safe_matplotlib_text(ax, 0.5, 0.98, "Note: Each entry represents one token position. Sub-word tokens are shown in blue.", 
                        ha='center', va='top', fontsize=9, style='italic',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Collect tokens from both analyses
    real_tokens = []
    synthetic_tokens = []
    
    # Real tokens (include token IDs)
    for example in real_stats.examples:
        for i, act in enumerate(example.activations):
            if act > 0 and i < len(example.tokens) and i < len(example.token_ids):
                real_tokens.append((act, example.tokens[i], example.token_ids[i]))
    
    # Synthetic tokens (include token IDs)
    for example in synthetic_stats.examples:
        for i, act in enumerate(example.activations):
            if act > 0 and i < len(example.tokens) and i < len(example.token_ids):
                synthetic_tokens.append((act, example.tokens[i], example.token_ids[i]))
    
    # Sort and take top 20 tokens from each
    real_tokens.sort(key=lambda x: x[0], reverse=True)
    synthetic_tokens.sort(key=lambda x: x[0], reverse=True)
    
    top_real = real_tokens[:20]
    top_synthetic = synthetic_tokens[:20]
    
    # === LEFT SIDE: REAL DATA (RED) ===
    real_x = 0.02
    real_width = 0.46
    
    # Real data background box
    real_rect = patches.Rectangle((real_x, 0.05), real_width, 0.87, 
                                linewidth=2, edgecolor='black', 
                                facecolor=REAL_COLOR, alpha=0.1)
    ax.add_patch(real_rect)
    
    # Real data title
    safe_matplotlib_text(ax, real_x + real_width/2, 0.88, "Real Data Tokens", 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=REAL_COLOR, alpha=0.8))
    
    # List real tokens
    _draw_token_list(ax, top_real, real_x + 0.02, 0.82, real_width - 0.04)
    
    # === RIGHT SIDE: SYNTHETIC DATA (BLUE) ===
    synth_x = 0.52
    synth_width = 0.46
    
    # Synthetic data background box
    synth_rect = patches.Rectangle((synth_x, 0.05), synth_width, 0.87, 
                                 linewidth=2, edgecolor='black', 
                                 facecolor=SYNTHETIC_COLOR, alpha=0.1)
    ax.add_patch(synth_rect)
    
    # Synthetic data title
    safe_matplotlib_text(ax, synth_x + synth_width/2, 0.88, "Synthetic Tokens", 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=SYNTHETIC_COLOR, alpha=0.8))
    
    # List synthetic tokens
    _draw_token_list(ax, top_synthetic, synth_x + 0.02, 0.82, synth_width - 0.04)


def _draw_token_list(ax, token_list, start_x, start_y, width):
    """Draw a clean list of tokens with context when possible"""
    if not token_list:
        safe_matplotlib_text(ax, start_x + width/2, start_y - 0.1, "No tokens found", 
                            ha='center', va='center', fontsize=10)
        return
    
    line_height = 0.04  # Space between lines
    
    for i, (activation, token, token_id) in enumerate(token_list):
        y = start_y - (i * line_height)
        
        # Clean and sanitize token display
        token_display = sanitize_text_for_matplotlib(str(token).strip())
        if len(token_display) > 15:
            token_display = token_display[:15] + "..."
        
        # Check if this looks like a sub-word token (check original token before sanitization)
        original_token = str(token).strip()
        is_subword = (original_token.startswith(('Ä ', 'â–', '##')) or 
                     len(original_token.strip()) <= 2 or
                     not original_token.strip().replace('\\', '').replace('$', '').isalnum())
        
        # Format the display based on whether it's likely a sub-word
        rank = i + 1
        if is_subword:
            # Show it's a sub-word token
            text = f"{rank:2d}. \"{token_display}\" (subword ID: {token_id}) - {activation:.3f}"
        else:
            # Regular token
            text = f"{rank:2d}. \"{token_display}\" (ID: {token_id}) - {activation:.3f}"
        
        # Color based on rank
        if rank <= 5:
            fontweight = 'bold'
            fontsize = 9
        elif rank <= 10:
            fontweight = 'normal'
            fontsize = 8.5
        else:
            fontweight = 'normal'
            fontsize = 8
        
        # Color coding for sub-word vs full tokens
        if is_subword:
            color = 'darkblue'  # Darker for sub-word tokens
        else:
            color = 'black'     # Black for complete tokens
        
        safe_matplotlib_text(ax, start_x, y, text, ha='left', va='center', 
                           fontsize=fontsize, fontweight=fontweight,
                           family='monospace', color=color)


def _plot_top_tokens(ax, stats: FeatureStats, color: str = 'lightblue'):
    """Plot top 20 firing tokens as a clean list with sub-word awareness"""
    ax.set_title("Top 20 Firing Tokens", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    if not stats.examples:
        safe_matplotlib_text(ax, 0.5, 0.5, "No token data available", ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
        return
    
    # Add explanation
    safe_matplotlib_text(ax, 0.5, 0.95, "Each entry = one token position. Sub-word tokens in blue.", 
                        ha='center', va='center', fontsize=9, style='italic',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Collect all activations and corresponding tokens with IDs
    activation_tokens = []
    for example in stats.examples:
        for i, act in enumerate(example.activations):
            if act > 0 and i < len(example.tokens) and i < len(example.token_ids):
                activation_tokens.append((act, example.tokens[i], example.token_ids[i]))
    
    if not activation_tokens:
        safe_matplotlib_text(ax, 0.5, 0.5, "No positive activations found", ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
        return
    
    # Sort by activation value and get top 20 tokens
    activation_tokens.sort(key=lambda x: x[0], reverse=True)
    top_tokens = activation_tokens[:20]
    
    # Background box
    rect = patches.Rectangle((0.05, 0.05), 0.9, 0.85, 
                           linewidth=2, edgecolor='black', 
                           facecolor=color, alpha=0.1)
    ax.add_patch(rect)
    
    # Draw the token list
    _draw_token_list(ax, top_tokens, 0.08, 0.85, 0.84)


def _plot_feature_stats_list(ax, stats: FeatureStats):
    """Plot feature statistics as a clean list with red box"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title above the box
    safe_matplotlib_text(ax, 0.5, 0.95, "Feature Statistics", ha='center', va='center', 
                        fontsize=12, fontweight='bold')
    
    # Red background box
    rect = patches.Rectangle((0.05, 0.05), 0.9, 0.85, 
                           linewidth=2, edgecolor='black', 
                           facecolor=REAL_COLOR, alpha=0.3)
    ax.add_patch(rect)
    
    # Statistics list
    stats_list = [
        f"Sparsity: {stats.sparsity:.3f}",
        f"Mean Activation: {stats.mean_activation:.3f}",
        f"Max Activation: {stats.max_activation:.3f}",
        f"Decoder Norm: {stats.decoder_norm:.3f}",
        f"Examples Found: {len(stats.examples)}"
    ]
    
    # Draw each line
    for i, stat_line in enumerate(stats_list):
        y_pos = 0.75 - (i * 0.12)
        safe_matplotlib_text(ax, 0.1, y_pos, stat_line, ha='left', va='center', 
                            fontsize=10, fontweight='normal')


def _plot_activation_histogram(ax, stats: FeatureStats, title: str = "Activation Distribution", color: str = 'skyblue'):
    """Plot distribution of activations"""
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    if not stats.examples:
        safe_matplotlib_text(ax, 0.5, 0.5, "No activation data", ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
        return
    
    # Collect all positive activations
    all_activations = []
    for example in stats.examples:
        all_activations.extend([act for act in example.activations if act > 0])
    
    if not all_activations:
        safe_matplotlib_text(ax, 0.5, 0.5, "No positive activations", ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
        return
    
    # Create histogram
    ax.hist(all_activations, bins=20, alpha=0.7, color=color, edgecolor='black')
    ax.set_xlabel("Activation Value", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xlim(0, None)
    
    # Set y-axis (counts) to show only whole numbers
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Add more ticks on x-axis (activation values) for better granularity
    max_activation = max(all_activations)
    if max_activation <= 1:
        # For small ranges, use 0.1 intervals
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    elif max_activation <= 5:
        # For medium ranges, use 0.5 intervals
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    elif max_activation <= 10:
        # For larger ranges, use 1.0 intervals
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    else:
        # For very large ranges, use adaptive spacing
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        ax.xaxis.set_minor_locator(ticker.MaxNLocator(nbins=20))
    
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = np.mean(all_activations)
    max_val = np.max(all_activations)
    std_val = np.std(all_activations)
    
    stats_text = f"Mean: {mean_val:.3f} | Max: {max_val:.3f} | Std: {std_val:.3f} | Count: {len(all_activations)}"
    safe_matplotlib_text(ax, 0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))


def _plot_logit_effects(ax, stats: FeatureStats):
    """Plot top boosted and suppressed tokens in blue theme"""
    ax.set_title("Logit Effects", fontsize=12, fontweight='bold')
    
    # Combine boosted and suppressed tokens
    all_tokens = []
    all_effects = []
    colors = []
    
    # Add boosted tokens (positive effects) - now blue
    for token, effect in stats.top_boosted_tokens[:5]:
        # Sanitize token for display
        safe_token = sanitize_text_for_matplotlib(str(token))
        all_tokens.append(f"+{safe_token}")
        all_effects.append(effect)
        colors.append('steelblue')  # Blue for positive
    
    # Add suppressed tokens (negative effects) - also blue but darker
    for token, effect in stats.top_suppressed_tokens[:5]:
        # Sanitize token for display
        safe_token = sanitize_text_for_matplotlib(str(token))
        all_tokens.append(f"-{safe_token}")
        all_effects.append(effect)
        colors.append('darkblue')  # Darker blue for negative
    
    if not all_tokens:
        safe_matplotlib_text(ax, 0.5, 0.5, "No logit effects computed", ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        return
    
    # Create horizontal bar plot
    y_pos = np.arange(len(all_tokens))
    bars = ax.barh(y_pos, all_effects, color=colors, alpha=0.7)
    
    # Set y-tick labels with sanitized text
    sanitized_labels = []
    for token in all_tokens:
        # Apply additional sanitization for y-tick labels
        sanitized_token = sanitize_text_for_matplotlib(token)
        sanitized_labels.append(sanitized_token)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sanitized_labels, fontsize=8)
    ax.set_xlabel("Logit Effect", fontsize=10)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, effect in zip(bars, all_effects):
        width = bar.get_width()
        label_text = f'{effect:.3f}'
        ax.text(width + (0.01 * max(abs(e) for e in all_effects) if all_effects else 0.01), 
               bar.get_y() + bar.get_height()/2, label_text,
               ha='left' if width >= 0 else 'right', va='center', fontsize=7)


def _create_summary_plot(feature_stats: Dict[int, FeatureStats], output_path: Path, analysis_type: str):
    """Create summary plot for multiple features"""
    configure_matplotlib_safe()
    
    n_features = len(feature_stats)
    color = REAL_COLOR if analysis_type == "real" else SYNTHETIC_COLOR
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Feature Summary - {analysis_type.title()} Analysis", fontsize=16, fontweight='bold')
    
    feature_indices = list(feature_stats.keys())
    
    # Sparsity comparison
    ax = axes[0, 0]
    sparsities = [feature_stats[idx].sparsity for idx in feature_indices]
    ax.bar(range(len(feature_indices)), sparsities, color=color, alpha=0.7)
    ax.set_title("Sparsity by Feature")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Sparsity")
    ax.set_xticks(range(len(feature_indices)))
    ax.set_xticklabels(feature_indices, rotation=45)
    
    # Max activation comparison
    ax = axes[0, 1]
    max_acts = [feature_stats[idx].max_activation for idx in feature_indices]
    ax.bar(range(len(feature_indices)), max_acts, color=color, alpha=0.7)
    ax.set_title("Max Activation by Feature")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Max Activation")
    ax.set_xticks(range(len(feature_indices)))
    ax.set_xticklabels(feature_indices, rotation=45)
    
    # Decoder norm comparison
    ax = axes[0, 2]
    dec_norms = [feature_stats[idx].decoder_norm for idx in feature_indices]
    ax.bar(range(len(feature_indices)), dec_norms, color=color, alpha=0.7)
    ax.set_title("Decoder Norm by Feature")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Decoder Norm")
    ax.set_xticks(range(len(feature_indices)))
    ax.set_xticklabels(feature_indices, rotation=45)
    
    # Example count comparison
    ax = axes[1, 0]
    example_counts = [len(feature_stats[idx].examples) for idx in feature_indices]
    ax.bar(range(len(feature_indices)), example_counts, color=color, alpha=0.7)
    ax.set_title("Number of Examples Found")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Example Count")
    ax.set_xticks(range(len(feature_indices)))
    ax.set_xticklabels(feature_indices, rotation=45)
    
    # Sparsity vs Max Activation scatter
    ax = axes[1, 1]
    ax.scatter(sparsities, max_acts, alpha=0.7, s=60, color=color)
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Max Activation")
    ax.set_title("Sparsity vs Max Activation")
    for i, idx in enumerate(feature_indices):
        ax.annotate(str(idx), (sparsities[i], max_acts[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    # Remove unused subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _create_statistics_plot(feature_stats: Dict[int, FeatureStats], output_path: Path, analysis_type: str):
    """Create overall statistics visualization"""
    configure_matplotlib_safe()
    
    color = REAL_COLOR if analysis_type == "real" else SYNTHETIC_COLOR
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Overall Statistics - {analysis_type.title()} Analysis", fontsize=16, fontweight='bold')
    
    # Extract all statistics
    sparsities = [stats.sparsity for stats in feature_stats.values()]
    max_activations = [stats.max_activation for stats in feature_stats.values()]
    mean_activations = [stats.mean_activation for stats in feature_stats.values()]
    decoder_norms = [stats.decoder_norm for stats in feature_stats.values()]
    
    # Sparsity distribution
    axes[0, 0].hist(sparsities, bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[0, 0].set_title("Sparsity Distribution")
    axes[0, 0].set_xlabel("Sparsity")
    axes[0, 0].set_ylabel("Frequency")
    
    # Max activation distribution
    axes[0, 1].hist(max_activations, bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[0, 1].set_title("Max Activation Distribution")
    axes[0, 1].set_xlabel("Max Activation")
    axes[0, 1].set_ylabel("Frequency")
    
    # Mean activation distribution
    axes[1, 0].hist(mean_activations, bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[1, 0].set_title("Mean Activation Distribution")
    axes[1, 0].set_xlabel("Mean Activation")
    axes[1, 0].set_ylabel("Frequency")
    
    # Decoder norm distribution
    axes[1, 1].hist(decoder_norms, bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[1, 1].set_title("Decoder Norm Distribution")
    axes[1, 1].set_xlabel("Decoder Norm")
    axes[1, 1].set_ylabel("Frequency")
    
    # Add statistics text
    stats_text = f"""
    Features analyzed: {len(feature_stats)}
    Avg sparsity: {np.mean(sparsities):.3f}
    Avg max activation: {np.mean(max_activations):.3f}
    Avg decoder norm: {np.mean(decoder_norms):.3f}
    Analysis type: {analysis_type.title()}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# Legacy function name for compatibility
def visualize_features_combined(real_stats: Dict[int, FeatureStats], 
                              synthetic_stats: Dict[int, FeatureStats],
                              output_dir: str = "visualizations") -> List[str]:
    """Legacy wrapper for compare_real_vs_synthetic"""
    return compare_real_vs_synthetic(real_stats, synthetic_stats, output_dir)