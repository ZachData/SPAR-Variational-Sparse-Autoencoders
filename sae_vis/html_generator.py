"""
Generate clean, interactive HTML visualizations for SAE features.
Minimal dependencies, maximum clarity.
Now includes beautiful histogram visualizations using Plotly.
"""
import json
import html
from typing import Dict, List, Any
from pathlib import Path
from .feature_analyzer import FeatureAnalysis


class HTMLGenerator:
    """Generate clean HTML visualizations for SAE features with interactive histograms"""
    
    def __init__(self):
        pass
    
    def create_visualization(
        self,
        feature_analyses: Dict[int, FeatureAnalysis],
        output_path: str,
        title: str = "SAE Feature Visualization"
    ):
        """
        Create a complete HTML visualization file.
        
        Args:
            feature_analyses: Dictionary of feature_idx -> FeatureAnalysis
            output_path: Where to save the HTML file
            title: Title for the visualization
        """
        html_content = self._generate_html(feature_analyses, title)
        
        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Visualization saved to {output_path}")
    
    def _generate_html(self, feature_analyses: Dict[int, FeatureAnalysis], title: str) -> str:
        """Generate the complete HTML document"""
        
        # Convert data to JSON for JavaScript
        feature_data = {}
        for feature_idx, analysis in feature_analyses.items():
            feature_data[feature_idx] = self._serialize_analysis(analysis)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{html.escape(title)}</h1>
            <div class="controls">
                <label for="feature-select">Feature:</label>
                <select id="feature-select">
                    {self._generate_feature_options(feature_analyses)}
                </select>
                <span id="feature-count">{len(feature_analyses)} features total</span>
            </div>
        </header>
        
        <main id="feature-content">
            <div class="loading">Loading feature data...</div>
        </main>
    </div>
    
    <script>
        const FEATURE_DATA = {json.dumps(feature_data, indent=2)};
        {self._get_javascript()}
    </script>
</body>
</html>
"""
        return html_template
    
    def _serialize_analysis(self, analysis: FeatureAnalysis) -> Dict[str, Any]:
        """Convert FeatureAnalysis to JSON-serializable format"""
        return {
            'feature_idx': analysis.feature_idx,
            'sparsity': analysis.sparsity,
            'mean_activation': analysis.mean_activation,
            'max_activation': analysis.max_activation,
            'decoder_norm': analysis.decoder_norm,
            'top_examples': [
                {
                    'text': example.text,
                    'tokens': example.tokens,
                    'activations': example.activations,
                    'max_activation': example.max_activation,
                    'seq_pos': example.seq_pos
                }
                for example in analysis.top_examples
            ],
            'top_boosted_tokens': analysis.top_boosted_tokens,
            'top_suppressed_tokens': analysis.top_suppressed_tokens,
            'activation_histogram': {
                'bin_edges': analysis.activation_histogram.bin_edges,
                'bar_heights': analysis.activation_histogram.bar_heights,
                'title': analysis.activation_histogram.title,
                'x_label': analysis.activation_histogram.x_label,
                'y_label': analysis.activation_histogram.y_label
            },
            'logit_histogram': {
                'bin_edges': analysis.logit_histogram.bin_edges,
                'bar_heights': analysis.logit_histogram.bar_heights,
                'title': analysis.logit_histogram.title,
                'x_label': analysis.logit_histogram.x_label,
                'y_label': analysis.logit_histogram.y_label
            }
        }
    
    def _generate_feature_options(self, feature_analyses: Dict[int, FeatureAnalysis]) -> str:
        """Generate <option> elements for feature selection"""
        options = []
        for feature_idx in sorted(feature_analyses.keys()):
            analysis = feature_analyses[feature_idx]
            label = f"Feature {feature_idx} (sparsity: {analysis.sparsity:.3f})"
            options.append(f'<option value="{feature_idx}">{html.escape(label)}</option>')
        return '\n'.join(options)
    
    def _get_css(self) -> str:
        """Get CSS styles for the visualization"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .controls {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        #feature-select {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            min-width: 200px;
        }
        
        #feature-count {
            color: #666;
            font-size: 14px;
        }
        
        main {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .feature-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .section {
            margin-bottom: 30px;
        }
        
        .section h2 {
            color: #34495e;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
        }
        
        .histograms-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .histogram-container {
            background: #fafafa;
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            padding: 15px;
        }
        
        .histogram-container h3 {
            margin-bottom: 10px;
            color: #2c3e50;
            font-size: 16px;
        }
        
        .histogram {
            width: 100%;
            height: 300px;
        }
        
        .examples-grid {
            display: grid;
            gap: 15px;
        }
        
        .example {
            padding: 15px;
            border: 1px solid #e1e8ed;
            border-radius: 6px;
            background: #fafafa;
        }
        
        .example-header {
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
        }
        
        .token-sequence {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 14px;
            line-height: 1.8;
            word-wrap: break-word;
        }
        
        .token {
            display: inline-block;
            padding: 2px 4px;
            margin: 1px;
            border-radius: 3px;
            border: 1px solid transparent;
        }
        
        .token.active {
            background-color: #3498db;
            color: white;
            border-color: #2980b9;
        }
        
        .token.medium {
            background-color: #85c1e9;
            color: #2c3e50;
            border-color: #5dade2;
        }
        
        .token.low {
            background-color: #d6eaf8;
            color: #2c3e50;
            border-color: #aed6f1;
        }
        
        .logits-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .logits-column h3 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 16px;
        }
        
        .boosted h3 {
            color: #27ae60;
        }
        
        .suppressed h3 {
            color: #e74c3c;
        }
        
        .token-list {
            list-style: none;
        }
        
        .token-list li {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            margin: 2px 0;
            background: #f8f9fa;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 13px;
        }
        
        .boosted .token-list li {
            border-left: 3px solid #27ae60;
        }
        
        .suppressed .token-list li {
            border-left: 3px solid #e74c3c;
        }
        
        .token-effect {
            font-weight: bold;
            font-size: 12px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        @media (max-width: 1200px) {
            .histograms-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .controls {
                flex-direction: column;
                align-items: stretch;
            }
            
            #feature-select {
                min-width: auto;
            }
            
            .logits-grid {
                grid-template-columns: 1fr;
            }
            
            .feature-stats {
                grid-template-columns: 1fr 1fr;
            }
        }
        """
    
    def _get_javascript(self) -> str:
        """Get JavaScript code for interactivity including histogram rendering"""
        return """
        // Initialize the visualization
        let currentFeature = null;
        
        function init() {
            const select = document.getElementById('feature-select');
            const featureIndices = Object.keys(FEATURE_DATA).map(Number).sort((a, b) => a - b);
            
            if (featureIndices.length > 0) {
                currentFeature = featureIndices[0];
                select.value = currentFeature;
                displayFeature(currentFeature);
            }
            
            select.addEventListener('change', (e) => {
                currentFeature = parseInt(e.target.value);
                displayFeature(currentFeature);
            });
        }
        
        function displayFeature(featureIdx) {
            const data = FEATURE_DATA[featureIdx];
            if (!data) return;
            
            const content = document.getElementById('feature-content');
            content.innerHTML = generateFeatureHTML(data);
            
            // Render histograms after DOM is updated
            setTimeout(() => {
                renderHistograms(data);
            }, 100);
        }
        
        function renderHistograms(data) {
            // Render activation histogram
            renderActivationHistogram(data.activation_histogram);
            
            // Render logit histogram  
            renderLogitHistogram(data.logit_histogram);
        }
        
        function renderActivationHistogram(histData) {
            const container = document.getElementById('activation-histogram');
            if (!container || !histData.bar_heights.length) return;
            
            // Calculate bin width for narrower bars
            const binWidths = [];
            for (let i = 0; i < histData.bin_edges.length - 1; i++) {
                binWidths.push(histData.bin_edges[i + 1] - histData.bin_edges[i]);
            }
            const avgBinWidth = binWidths.reduce((a, b) => a + b, 0) / binWidths.length;
            
            // Create bin centers for x-axis
            const binCenters = [];
            for (let i = 0; i < histData.bin_edges.length - 1; i++) {
                binCenters.push((histData.bin_edges[i] + histData.bin_edges[i + 1]) / 2);
            }
            
            const trace = {
                x: binCenters,
                y: histData.bar_heights,
                type: 'bar',
                width: avgBinWidth * 0.8,  // Make bars 80% of bin width (narrower)
                marker: {
                    color: '#3498db',
                    opacity: 0.7,
                    line: {
                        color: '#2980b9',
                        width: 0.5
                    }
                },
                name: 'Activations'
            };
            
            const layout = {
                title: {
                    text: histData.title,
                    font: { size: 14 }
                },
                xaxis: {
                    title: histData.x_label,
                    showgrid: true,
                    gridcolor: '#e0e0e0',
                    zeroline: false,
                    range: [0, 10]  // Fixed x-axis from 0 to 10
                },
                yaxis: {
                    title: histData.y_label,
                    showgrid: true,
                    gridcolor: '#e0e0e0',
                    zeroline: false,
                    rangemode: 'tozero'  // Y-axis starts at 0
                },
                margin: { l: 50, r: 30, t: 50, b: 50 },
                height: 300,
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                bargap: 0.1  // Small gap between bars for clean look
            };
            
            const config = {
                responsive: true,
                displayModeBar: false
            };
            
            Plotly.newPlot(container, [trace], layout, config);
        }
        
        function renderLogitHistogram(histData) {
            const container = document.getElementById('logit-histogram');
            if (!container || !histData.bar_heights.length) return;
            
            // Calculate bin width for narrower bars
            const binWidths = [];
            for (let i = 0; i < histData.bin_edges.length - 1; i++) {
                binWidths.push(histData.bin_edges[i + 1] - histData.bin_edges[i]);
            }
            const avgBinWidth = binWidths.reduce((a, b) => a + b, 0) / binWidths.length;
            
            // Create bin centers for x-axis
            const binCenters = [];
            for (let i = 0; i < histData.bin_edges.length - 1; i++) {
                binCenters.push((histData.bin_edges[i] + histData.bin_edges[i + 1]) / 2);
            }
            
            const trace = {
                x: binCenters,
                y: histData.bar_heights,
                type: 'bar',
                width: avgBinWidth * 0.8,  // Make bars 80% of bin width (narrower)
                marker: {
                    color: '#e74c3c',
                    opacity: 0.7,
                    line: {
                        color: '#c0392b',
                        width: 0.5
                    }
                },
                name: 'Logit Effects'
            };
            
            const layout = {
                title: {
                    text: histData.title,
                    font: { size: 14 }
                },
                xaxis: {
                    title: histData.x_label,
                    showgrid: true,
                    gridcolor: '#e0e0e0',
                    zeroline: true,
                    zerolinecolor: '#666',
                    zerolinewidth: 1
                },
                yaxis: {
                    title: histData.y_label,
                    showgrid: true,
                    gridcolor: '#e0e0e0',
                    zeroline: false
                },
                margin: { l: 50, r: 30, t: 50, b: 50 },
                height: 300,
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                bargap: 0.1  // Small gap between bars for clean look
            };
            
            const config = {
                responsive: true,
                displayModeBar: false
            };
            
            Plotly.newPlot(container, [trace], layout, config);
        }
        
        function generateFeatureHTML(data) {
            return `
                <div class="feature-stats">
                    <div class="stat-item">
                        <div class="stat-label">Feature Index</div>
                        <div class="stat-value">${data.feature_idx}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Sparsity</div>
                        <div class="stat-value">${(data.sparsity * 100).toFixed(2)}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Mean Activation</div>
                        <div class="stat-value">${data.mean_activation.toFixed(3)}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Max Activation</div>
                        <div class="stat-value">${data.max_activation.toFixed(3)}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Decoder Norm</div>
                        <div class="stat-value">${data.decoder_norm.toFixed(3)}</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Feature Analysis</h2>
                    <div class="histograms-grid">
                        <div class="histogram-container">
                            <h3>Activation Distribution</h3>
                            <div id="activation-histogram" class="histogram"></div>
                        </div>
                        <div class="histogram-container">
                            <h3>Logit Effects</h3>
                            <div id="logit-histogram" class="histogram"></div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Top Activating Examples</h2>
                    <div class="examples-grid">
                        ${data.top_examples.map(generateExampleHTML).join('')}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Logit Effects</h2>
                    <div class="logits-grid">
                        <div class="logits-column boosted">
                            <h3>↑ Promoted Tokens</h3>
                            <ul class="token-list">
                                ${data.top_boosted_tokens.map(([token, effect]) => 
                                    `<li><span>"${escapeHtml(token)}"</span><span class="token-effect">+${effect.toFixed(3)}</span></li>`
                                ).join('')}
                            </ul>
                        </div>
                        <div class="logits-column suppressed">
                            <h3>↓ Suppressed Tokens</h3>
                            <ul class="token-list">
                                ${data.top_suppressed_tokens.map(([token, effect]) => 
                                    `<li><span>"${escapeHtml(token)}"</span><span class="token-effect">${effect.toFixed(3)}</span></li>`
                                ).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function generateExampleHTML(example) {
            const maxAct = example.max_activation;
            const threshold1 = maxAct * 0.3;
            const threshold2 = maxAct * 0.6;
            
            const tokenHTML = example.tokens.map((token, i) => {
                const activation = example.activations[i] || 0;
                let className = 'token';
                
                if (activation > threshold2) {
                    className += ' active';
                } else if (activation > threshold1) {
                    className += ' medium';
                } else if (activation > 0) {
                    className += ' low';
                }
                
                return `<span class="${className}" title="Activation: ${activation.toFixed(4)}">${escapeHtml(token)}</span>`;
            }).join('');
            
            return `
                <div class="example">
                    <div class="example-header">
                        Max activation: ${maxAct.toFixed(4)} at position ${example.seq_pos}
                    </div>
                    <div class="token-sequence">
                        ${tokenHTML}
                    </div>
                </div>
            `;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', init);
        """