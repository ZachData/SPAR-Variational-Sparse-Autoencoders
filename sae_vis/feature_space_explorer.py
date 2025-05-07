"""
Feature Space Explorer: Interactive visualization of dictionary feature relationships

This module creates interactive visualizations showing the continuous space between
features in trained dictionaries, allowing exploration of feature relationships
and semantic transitions between features as described in the "Towards Monosemanticity" paper.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import json
import os
import sys

# For embeddings
try:
    import umap
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: Some embedding methods may not be available. Please install umap-learn, scikit-learn.")


def compute_feature_embedding(
    dictionary: Any, 
    method: str = 'umap',
    n_components: int = 2, 
    random_state: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Project dictionary features into a 2D/3D space using specified dimensionality reduction method.
    
    Args:
        dictionary: The trained dictionary/SAE model
        method: Dimensionality reduction method ('umap', 'tsne', 'pca', 'cosine')
        n_components: Number of dimensions for the embedding (usually 2)
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters passed to the specific embedding method
        
    Returns:
        embedding: Array of shape (n_features, n_components) containing the embedding coordinates
    """
    # Extract feature vectors from dictionary (support different formats)
    if hasattr(dictionary, 'decoder') and hasattr(dictionary.decoder, 'weight'):
        feature_matrix = dictionary.decoder.weight.detach().cpu().numpy()
    elif hasattr(dictionary, 'W_dec'):
        feature_matrix = dictionary.W_dec.detach().cpu().numpy().T
    else:
        raise ValueError("Dictionary format not recognized")
    
    # Apply dimensionality reduction based on selected method
    if method == 'umap':
        if 'umap' not in sys.modules:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        
        # Default UMAP parameters
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'cosine',
            'random_state': random_state
        }
        umap_params.update(kwargs)
        
        reducer = umap.UMAP(n_components=n_components, **umap_params)
        embedding = reducer.fit_transform(feature_matrix)
        
    elif method == 'tsne':
        # Default t-SNE parameters
        tsne_params = {
            'perplexity': 30,
            'metric': 'cosine',
            'random_state': random_state
        }
        tsne_params.update(kwargs)
        
        tsne = TSNE(n_components=n_components, **tsne_params)
        embedding = tsne.fit_transform(feature_matrix)
        
    elif method == 'pca':
        # PCA is simpler, fewer parameters
        pca = PCA(n_components=n_components, random_state=random_state)
        embedding = pca.fit_transform(feature_matrix)
        
    elif method == 'cosine':
        # Use the first two principal components of the cosine similarity matrix
        # This is a simpler alternative to more complex embedding methods
        sim_matrix = cosine_similarity(feature_matrix)
        pca = PCA(n_components=n_components, random_state=random_state)
        embedding = pca.fit_transform(sim_matrix)
        
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    return embedding


def compute_feature_relationships(
    dictionary: Any, 
    embedding: np.ndarray,
    n_neighbors: int = 5,
    similarity_threshold: float = 0.3,
    max_relationships: int = 500
) -> List[Dict[str, Any]]:
    """
    Compute relationships between features based on embedding distances and feature similarities.
    
    Args:
        dictionary: The trained dictionary/SAE model
        embedding: Feature embedding from compute_feature_embedding
        n_neighbors: Number of connections per feature
        similarity_threshold: Minimum cosine similarity to create a connection
        max_relationships: Maximum number of relationships to return
        
    Returns:
        relationships: List of dictionaries with source, target, similarity, and distance
    """
    # Calculate pairwise distances in embedding space
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(embedding))
    
    # Get feature weight matrix for calculating similarities
    if hasattr(dictionary, 'decoder') and hasattr(dictionary.decoder, 'weight'):
        weight_matrix = dictionary.decoder.weight.detach().cpu()
    elif hasattr(dictionary, 'W_dec'):
        weight_matrix = dictionary.W_dec.detach().cpu().T
    else:
        raise ValueError("Dictionary format not recognized")
    
    # Normalize weight matrix for cosine similarity
    normalized_weights = weight_matrix / weight_matrix.norm(dim=1, keepdim=True)
    similarities = normalized_weights @ normalized_weights.T
    similarities = similarities.numpy()
    
    # Find connections for each feature
    relationships = []
    
    for i in range(embedding.shape[0]):
        # Find nearest neighbors by embedding distance
        neighbors = np.argsort(distances[i])[1:n_neighbors+1]
        
        for j in neighbors:
            # Only add relationship if similarity is above threshold
            # and we haven't exceeded max relationships
            if similarities[i, j] > similarity_threshold and len(relationships) < max_relationships:
                # Only add relationship if j > i to avoid duplicates
                if j > i:
                    relationships.append({
                        "source": int(i),
                        "target": int(j),
                        "distance": float(distances[i, j]),
                        "similarity": float(similarities[i, j])
                    })
    
    return relationships


def collect_feature_activations(
    model: Any, 
    dictionary: Any, 
    buffer: Any,
    num_features: int = 256,
    n_examples: int = 5,
    batch_size: int = 32
) -> Dict[int, Dict[str, List]]:
    """
    Collect examples of tokens that activate each feature.
    
    Args:
        model: The transformer model
        dictionary: The trained dictionary/SAE model
        buffer: Activation buffer for collecting examples
        num_features: Number of features to visualize
        n_examples: Number of examples to collect per feature
        batch_size: Batch size for processing
        
    Returns:
        feature_data: Dictionary mapping feature indices to example data
    """
    feature_data = {}
    
    # Initialize dictionary for each feature
    for feat_idx in range(min(num_features, dictionary.dict_size)):
        feature_data[feat_idx] = {
            "tokens": [],
            "texts": [],
            "activations": [],
            "positions": []
        }
    
    # Get batch of tokens and their activations
    # Using text_batch instead of get_batch
    batch_tokens = buffer.tokenized_batch(batch_size=batch_size)
    tokens = batch_tokens["input_ids"]
    
    # Run model to get activations
    with torch.no_grad():
        hook_dict = {}
        activations = []
        
        # Define hook function to collect activations
        def activation_hook(tensor, hook):
            activations.append(tensor.detach().clone())
            return tensor
        
        # Determine hook name
        hook_name = buffer.hook_name
        
        # Get activations using hook
        with model.hooks(fwd_hooks=[(hook_name, activation_hook)]):
            model(tokens.to(model.cfg.device))
            
        # Process the activations
        if activations:
            act_tensor = activations[0].to('cpu')
            
            # Encode with dictionary to get feature activations
            feature_acts = dictionary.encode(act_tensor)
            
            # Find top activating tokens for each feature
            for feat_idx in range(min(num_features, dictionary.dict_size)):
                # Get the activation values for this feature
                feat_activations = feature_acts[..., feat_idx]
                
                # Find indices of tokens with top activations
                flat_indices = torch.flatten(feat_activations).argsort(descending=True)[:n_examples]
                
                # Convert flat indices to batch and sequence indices
                batch_size, seq_len = tokens.shape[0], tokens.shape[1]
                batch_indices = flat_indices // seq_len
                seq_indices = flat_indices % seq_len
                
                # Store information for each example
                for b_idx, s_idx in zip(batch_indices, seq_indices):
                    if b_idx < batch_size and s_idx < seq_len:
                        # Get token ID
                        token_id = tokens[b_idx, s_idx].item()
                        
                        # Get text representation
                        try:
                            text = model.tokenizer.decode([token_id])
                        except:
                            text = f"<token_{token_id}>"
                        
                        # Get activation value
                        activation = feat_activations[b_idx, s_idx].item()
                        
                        # Store example
                        feature_data[feat_idx]["tokens"].append(token_id)
                        feature_data[feat_idx]["texts"].append(text)
                        feature_data[feat_idx]["activations"].append(activation)
                        feature_data[feat_idx]["positions"].append(f"({b_idx}, {s_idx})")
    
    return feature_data


def generate_interpolation_examples(
    model: Any, 
    dictionary: Any, 
    buffer: Any, 
    relationships: List[Dict],
    num_steps: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate examples of interpolated feature activations between pairs of features.
    
    Args:
        model: The transformer model
        dictionary: The trained dictionary/SAE model
        buffer: Activation buffer for collecting examples
        relationships: Feature relationships from compute_feature_relationships
        num_steps: Number of interpolation steps between features
        
    Returns:
        interpolation_examples: List of dictionaries with interpolation data
    """
    # This is a simplified implementation
    interpolation_examples = []
    
    # Just create placeholder data
    for rel in relationships[:10]:  # Limit to 10 relationships
        feat_i, feat_j = rel["source"], rel["target"]
        
        steps = []
        for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
            steps.append({
                "alpha": alpha,
                "examples": []
            })
        
        interpolation_examples.append({
            "feature_i": int(feat_i),
            "feature_j": int(feat_j),
            "steps": steps
        })
    
    print("Warning: generate_interpolation_examples is not fully implemented")
    return interpolation_examples


def create_visualization(
    output_file: str,
    embedding: np.ndarray,
    feature_data: Dict[int, Dict[str, List]],
    relationships: List[Dict],
    interpolation_examples: List[Dict],
    model_info: Dict[str, Any]
) -> None:
    """
    Create the HTML visualization with interactive features.
    
    Args:
        output_file: Path to save the HTML file
        embedding: Feature embedding coordinates
        feature_data: Data about each feature (from collect_feature_activations)
        relationships: Feature relationships
        interpolation_examples: Interpolation examples
        model_info: Information about the model and dictionary
    """
    print(f"Creating visualization at {output_file}...")
    
    # Prepare data for visualization
    vis_data = {
        "embedding": embedding.tolist(),
        "features": feature_data,
        "relationships": relationships,
        "interpolation_examples": interpolation_examples,
        "model_info": model_info
    }
    
    # React component for the visualization
    react_component = """
    // Feature Space Explorer Component
    const FeatureSpaceExplorer = ({ data }) => {
        const [selectedFeature, setSelectedFeature] = useState(null);
        const svgRef = useRef(null);
        
        useEffect(() => {
            if (!data || !data.embedding || !svgRef.current) return;
            
            // Basic D3 visualization setup
            const width = 800;
            const height = 600;
            const margin = { top: 20, right: 20, bottom: 30, left: 40 };
            
            // Extract embedding data
            const points = data.embedding.map((coords, i) => ({
                id: i,
                x: coords[0],
                y: coords[1]
            }));
            
            // Find min/max for scaling
            const xExtent = d3.extent(points, d => d.x);
            const yExtent = d3.extent(points, d => d.y);
            
            // Create scales
            const xScale = d3.scaleLinear()
                .domain([xExtent[0] - 0.5, xExtent[1] + 0.5])
                .range([margin.left, width - margin.right]);
                
            const yScale = d3.scaleLinear()
                .domain([yExtent[0] - 0.5, yExtent[1] + 0.5])
                .range([height - margin.bottom, margin.top]);
            
            // Clear previous SVG content
            d3.select(svgRef.current).selectAll("*").remove();
            
            // Create SVG element
            const svg = d3.select(svgRef.current)
                .attr("width", width)
                .attr("height", height)
                .attr("viewBox", [0, 0, width, height]);
            
            // Draw relationships (edges)
            if (data.relationships && data.relationships.length > 0) {
                svg.selectAll("line")
                    .data(data.relationships)
                    .enter()
                    .append("line")
                    .attr("x1", d => xScale(points[d.source].x))
                    .attr("y1", d => yScale(points[d.source].y))
                    .attr("x2", d => xScale(points[d.target].x))
                    .attr("y2", d => yScale(points[d.target].y))
                    .attr("stroke", "#ccc")
                    .attr("stroke-width", d => Math.max(0.5, d.similarity * 2))
                    .attr("opacity", 0.6);
            }
            
            // Draw points (features)
            const dots = svg.selectAll("circle")
                .data(points)
                .enter()
                .append("circle")
                .attr("cx", d => xScale(d.x))
                .attr("cy", d => yScale(d.y))
                .attr("r", 5)
                .attr("fill", "#1f77b4")
                .attr("opacity", 0.8)
                .attr("stroke", "#fff")
                .attr("stroke-width", 1)
                .on("mouseover", (event, d) => {
                    setSelectedFeature(d.id);
                    d3.select(event.target)
                        .attr("r", 8)
                        .attr("fill", "#ff7f0e");
                })
                .on("mouseout", (event, d) => {
                    d3.select(event.target)
                        .attr("r", 5)
                        .attr("fill", "#1f77b4");
                });
            
            // Add axes
            const xAxis = d3.axisBottom(xScale);
            const yAxis = d3.axisLeft(yScale);
            
            svg.append("g")
                .attr("transform", `translate(0,${height - margin.bottom})`)
                .call(xAxis);
                
            svg.append("g")
                .attr("transform", `translate(${margin.left},0)`)
                .call(yAxis);
                
        }, [data]);
        
        return (
            <div className="feature-space-explorer">
                <h2>Feature Space Explorer</h2>
                <div className="model-info">
                    <p><strong>Model:</strong> {data.model_info.name}</p>
                    <p><strong>Hook:</strong> {data.model_info.hook_name}</p>
                    <p><strong>Dictionary Size:</strong> {data.model_info.dict_size}</p>
                </div>
                
                <div className="visualization-container">
                    <svg ref={svgRef}></svg>
                </div>
                
                {selectedFeature !== null && (
                    <div className="feature-details">
                        <h3>Feature {selectedFeature}</h3>
                        {data.features[selectedFeature] && (
                            <div className="examples">
                                <h4>Top Activating Tokens:</h4>
                                <ul>
                                    {data.features[selectedFeature].texts.map((text, i) => (
                                        <li key={i}>
                                            "{text}" (activation: {data.features[selectedFeature].activations[i].toFixed(2)})
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                )}
            </div>
        );
    };
    
    export default FeatureSpaceExplorer;
    """
    
    # Full HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Space Explorer</title>
        <meta charset="utf-8">
        <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .feature-space-explorer {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h2 {{
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        
        .model-info {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
            background-color: #f0f4f8;
            padding: 10px;
            border-radius: 4px;
        }}
        
        .model-info p {{
            margin: 0;
        }}
        
        .visualization-container {{
            width: 100%;
            overflow: auto;
            margin-bottom: 20px;
        }}
        
        svg {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        
        .feature-details {{
            background-color: #f0f4f8;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }}
        
        .examples ul {{
            padding-left: 20px;
        }}
        
        .examples li {{
            margin-bottom: 5px;
        }}
        </style>
    </head>
    <body>
        <div id="app"></div>
        
        <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
        <script src="https://unpkg.com/d3@7"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        
        <script type="text/babel">
        // Import React hooks directly
        const {{ useState, useEffect, useRef }} = React;
        
        {react_component}
        
        // Visualization data
        const visData = {json.dumps(vis_data)};
        
        ReactDOM.render(React.createElement(FeatureSpaceExplorer, {{ data: visData }}), document.getElementById('app'));
        </script>
    </body>
    </html>
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_template)
    
    print(f"Visualization saved to {output_file}")


def create_feature_space_visualization(
    model: Any,
    dictionary: Any,
    buffer: Any,
    output_file: str,
    embedding_method: str = 'umap',
    num_features: int = 256,
    n_neighbors: int = 5,
    n_examples: int = 5,
    **kwargs
) -> None:
    """
    Main function to create feature space visualization.
    
    Args:
        model: The transformer model
        dictionary: The trained dictionary/SAE model
        buffer: Activation buffer
        output_file: Path to save visualization
        embedding_method: Method for dimensionality reduction ('umap', 'tsne', 'pca', 'cosine')
        num_features: Number of features to visualize
        n_neighbors: Number of connections per feature
        n_examples: Number of examples to collect per feature
        **kwargs: Additional parameters for embedding method
    """
    try:
        # Import torch explicitly here to avoid "name 'torch' is not defined" error
        import torch
        
        # Get model info
        model_info = {
            "name": getattr(model, 'name', 'unknown'),
            "hook_name": getattr(buffer, 'hook_name', 'unknown'),
            "dict_size": getattr(dictionary, 'dict_size', 'unknown'),
            "activation_dim": getattr(dictionary, 'activation_dim', 'unknown')
        }
        
        # 1. Compute feature embedding
        print(f"Computing {embedding_method} embedding...")
        embedding = compute_feature_embedding(
            dictionary=dictionary,
            method=embedding_method,
            **kwargs
        )
        
        # Limit to specified number of features
        embedding = embedding[:num_features]
        
        # 2. Compute feature relationships
        print("Computing feature relationships...")
        relationships = compute_feature_relationships(
            dictionary=dictionary,
            embedding=embedding,
            n_neighbors=n_neighbors
        )
        
        # 3. Collect feature activations
        print("Collecting feature activations...")
        try:
            feature_data = collect_feature_activations(
                model=model,
                dictionary=dictionary,
                buffer=buffer,
                num_features=num_features,
                n_examples=n_examples
            )
        except Exception as e:
            print(f"Warning: Error collecting feature activations: {e}")
            # Create empty feature data as fallback
            feature_data = {
                feat_idx: {"tokens": [], "texts": [], "activations": [], "positions": []}
                for feat_idx in range(min(num_features, dictionary.dict_size))
            }
        
        # 4. Generate interpolation examples
        print("Generating interpolation examples...")
        try:
            interpolation_examples = generate_interpolation_examples(
                model=model,
                dictionary=dictionary,
                buffer=buffer,
                relationships=relationships
            )
        except Exception as e:
            print(f"Warning: Error generating interpolation examples: {e}")
            interpolation_examples = []
        
        # 5. Create visualization
        print("Creating visualization...")
        create_visualization(
            output_file=output_file,
            embedding=embedding,
            feature_data=feature_data,
            relationships=relationships,
            interpolation_examples=interpolation_examples,
            model_info=model_info
        )
        
        print(f"Feature space visualization saved to {output_file}")
        
    except Exception as e:
        print(f"Error creating feature space visualization: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a minimal visualization with just the embedding if possible
        try:
            print("Attempting to create minimal visualization...")
            
            model_info = {
                "name": getattr(model, 'name', 'unknown'),
                "hook_name": getattr(buffer, 'hook_name', 'unknown'),
                "dict_size": getattr(dictionary, 'dict_size', 'unknown'),
                "activation_dim": getattr(dictionary, 'activation_dim', 'unknown')
            }
            
            # Just compute embedding
            embedding = compute_feature_embedding(
                dictionary=dictionary,
                method='pca' if embedding_method == 'umap' else embedding_method,  # PCA as fallback
                **kwargs
            )
            
            # Create very basic visualization
            create_visualization(
                output_file=output_file,
                embedding=embedding[:num_features],
                feature_data={},
                relationships=[],
                interpolation_examples=[],
                model_info=model_info
            )
            
            print(f"Minimal visualization saved to {output_file}")
            
        except Exception as e:
            print(f"Failed to create even minimal visualization: {e}")