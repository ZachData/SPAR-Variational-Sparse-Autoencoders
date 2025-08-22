#!/usr/bin/env python3
"""
Main orchestrator for SAE feature analysis.
Provides command-line interface to run real activations, synthetic maximizers, or both.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformer_lens import HookedTransformer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Add local imports
from real_activations import analyze_real_activations
from synth_activations import generate_synthetic_maximizers
from visualizer import visualize_features, compare_real_vs_synthetic


def load_models(model_name: str, sae_path: str, device: Optional[str] = None):
    """Load transformer and SAE models"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = "gelu-1l"
    print(f"Loading transformer model: {model_name}")
    transformer = HookedTransformer.from_pretrained(model_name, device=device)
    print(f"Model loaded. d_model: {transformer.cfg.d_model}")
    
    print(f"Loading SAE model from: {sae_path}")
    
    # Try different SAE loading methods
    try:
        # Try VSAETopK first (from parent directory)
        from dictionary_learning.trainers.vsae_topk import VSAETopK
        sae_model = VSAETopK.from_pretrained(sae_path, device=device)
        print(f"Loaded VSAETopK: {sae_model.dict_size} features, k={sae_model.k}")
    except:
        try:
            # Try AutoEncoderTopK
            from trainers.top_k import AutoEncoderTopK
            sae_model = AutoEncoderTopK.from_pretrained(sae_path, device=device)
            print(f"Loaded AutoEncoderTopK: {sae_model.dict_size} features")
        except Exception as e:
            print(f"Failed to load SAE model: {e}")
            raise
    
    return transformer, sae_model


def run_real_analysis(transformer, sae_model, feature_indices: List[int],
                     output_dir: str, **kwargs):
    """Run real activation analysis"""
    print("\n" + "="*50)
    print("RUNNING REAL ACTIVATION ANALYSIS")
    print("="*50)
    
    results = analyze_real_activations(
        sae_model=sae_model,
        transformer_model=transformer,
        feature_indices=feature_indices,
        **kwargs
    )
    
    print(f"✅ Real analysis complete: {len(results)} features analyzed")
    
    # Visualize results
    image_paths = visualize_features(results, output_dir, "real")
    print(f"📊 Generated {len(image_paths)} visualization images")
    
    return results


def run_synthetic_analysis(transformer, sae_model, feature_indices: List[int],
                          output_dir: str, **kwargs):
    """Run synthetic maximizer analysis"""
    print("\n" + "="*50)
    print("RUNNING SYNTHETIC MAXIMIZER ANALYSIS")
    print("="*50)
    
    results = generate_synthetic_maximizers(
        sae_model=sae_model,
        transformer_model=transformer,
        feature_indices=feature_indices,
        **kwargs
    )
    
    print(f"✅ Synthetic analysis complete: {len(results)} features optimized")
    
    # Visualize results
    image_paths = visualize_features(results, output_dir, "synthetic")
    print(f"📊 Generated {len(image_paths)} visualization images")
    
    return results


def run_comparison_analysis(transformer, sae_model, feature_indices: List[int],
                           output_dir: str, **kwargs):
    """Run both analyses and compare results"""
    print("\n" + "="*60)
    print("RUNNING COMPARISON ANALYSIS (REAL vs SYNTHETIC)")
    print("="*60)
    
    # Filter kwargs for real analysis
    real_kwargs = {k: v for k, v in kwargs.items() 
                   if k in ['max_examples', 'batch_size', 'buffer_size']}
    if 'seq_len' in kwargs:
        real_kwargs['ctx_len'] = kwargs['seq_len']  # Map seq_len to ctx_len
    
    # Filter kwargs for synthetic analysis  
    synthetic_kwargs = {k: v for k, v in kwargs.items()
                       if k in ['max_examples', 'seq_len', 'n_steps', 'learning_rate']}
    
    # Run real analysis
    real_results = run_real_analysis(
        transformer, sae_model, feature_indices, 
        f"{output_dir}/real", **real_kwargs
    )
    
    # Run synthetic analysis
    synthetic_results = run_synthetic_analysis(
        transformer, sae_model, feature_indices,
        f"{output_dir}/synthetic", **synthetic_kwargs
    )
    
    # Create comparison visualizations
    print("\n📊 Creating comparison visualizations...")
    comparison_paths = compare_real_vs_synthetic(
        real_results, synthetic_results, f"{output_dir}/comparison"
    )
    
    print(f"✅ Comparison complete! Check {output_dir}/ for all results")
    
    return real_results, synthetic_results


def main():
    parser = argparse.ArgumentParser(
        description="SAE Feature Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Analysis type')
    
    # Real activation analysis
    real_parser = subparsers.add_parser('real', help='Analyze real activations from c4-code dataset')
    real_parser.add_argument('--buffer-size', type=int, default=1000, help='Activation buffer size')
    real_parser.add_argument('--max-examples', type=int, default=20, help='Max examples per feature')
    real_parser.add_argument('--batch-size', type=int, default=8, help='Processing batch size')
    
    # Synthetic maximizer analysis
    synth_parser = subparsers.add_parser('synthetic', help='Generate synthetic maximizers')
    synth_parser.add_argument('--max-examples', type=int, default=5, help='Synthetic examples per feature')
    synth_parser.add_argument('--seq-len', type=int, default=64, help='Sequence length for optimization')
    synth_parser.add_argument('--n-steps', type=int, default=200, help='Optimization steps')
    synth_parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    
    # Comparison analysis
    comp_parser = subparsers.add_parser('compare', help='Run both analyses and compare')
    comp_parser.add_argument('--buffer-size', type=int, default=1000, help='Activation buffer size')
    comp_parser.add_argument('--max-examples', type=int, default=15, help='Max examples per feature')
    comp_parser.add_argument('--batch-size', type=int, default=8, help='Processing batch size')
    comp_parser.add_argument('--seq-len', type=int, default=64, help='Synthetic sequence length')
    comp_parser.add_argument('--n-steps', type=int, default=150, help='Synthetic optimization steps')
    
    # Common arguments for all subcommands
    for subparser in [real_parser, synth_parser, comp_parser]:
        subparser.add_argument('--model', type=str, default="gelu-1l", 
                             help='Transformer model name (e.g., gelu-1l)')
        subparser.add_argument('--sae', type=str, required=True,
                             help='Path to SAE model file')
        subparser.add_argument('--features', type=str, required=True,
                             help='Feature indices to analyze (e.g., "0,1,2" or "0-10")')
        subparser.add_argument('--output', type=str, default='sae_analysis',
                             help='Output directory')
        subparser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                             default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Parse feature indices
    if '-' in args.features:
        start, end = map(int, args.features.split('-'))
        feature_indices = list(range(start, end + 1))
    else:
        feature_indices = [int(x.strip()) for x in args.features.split(',')]
    
    print(f"🎯 Analyzing {len(feature_indices)} features: {feature_indices}")
    
    # Set device
    device = None if args.device == 'auto' else args.device
    
    try:
        # Load models
        transformer, sae_model = load_models(args.model, args.sae, device)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if args.command == 'real':
            # Run analysis
            run_real_analysis(
                transformer, sae_model, feature_indices, args.output,
                max_examples=args.max_examples, batch_size=args.batch_size,
                buffer_size=args.buffer_size
            )
            
        elif args.command == 'synthetic':
            # Run synthetic analysis
            run_synthetic_analysis(
                transformer, sae_model, feature_indices, args.output,
                max_examples=args.max_examples, seq_len=args.seq_len,
                n_steps=args.n_steps, learning_rate=args.learning_rate
            )
            
        elif args.command == 'compare':
            # Run comparison
            run_comparison_analysis(
                transformer, sae_model, feature_indices, args.output,
                max_examples=args.max_examples, batch_size=args.batch_size,
                seq_len=args.seq_len, n_steps=args.n_steps,
                buffer_size=args.buffer_size
            )
        
        print(f"\n🎉 Analysis complete! Results saved to: {args.output}")
        print(f"📁 Open the generated PNG files to explore your SAE features")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Examples:
  # Real activation analysis (uses c4-code dataset like training script)
  python main.py real --model gelu-1l --sae path/to/sae.pt --features 0,1,2,3,4

  # Synthetic maximizer analysis  
  python main.py synthetic --model gelu-1l --sae path/to/sae.pt --features 0,1,2,3,4

  # Both analyses with comparison
  python main.py compare --model gelu-1l --sae path/to/sae.pt --features 0-10

  # Custom buffer settings for real analysis
  python main.py real --model gelu-1l --sae path/to/sae.pt --features 0-50 --buffer-size 2000
"""