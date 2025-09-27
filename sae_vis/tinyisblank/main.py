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
    
    model_name = "pythia-70m"
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
    if args.features == "active":
        # Your active features from the JSON 
        feature_indices = [479,481,484,490,524,534,540,543,545,548,556,561,568,573,581,] #vsae256

        # [10,13,21,26,30,53,56,89,92,103,110,112,117,129,130,133,143,150,176,179,193,205,223,232,236,252,291,295,313,317,327,333,346,368,388,390,397,400,402,432,433,453,472,479,481,484,490,524,534,540,543,545,548,556,561,568,573,581,586,603,604,624,628,632,634,642,654,655,662,665,669,728,731,737,740,746,775,787,792,795,796,800,810,821,841,844,852,858,860,866,868,873,898,904,912,918,922,939,940,941,953,967,974,983,1001,1008,1017,1026,1035,1061,1062,1100,1135,1145,1153,1173,1174,1176,1181,1197,1198,1230,1235,1242,1252,1259,1270,1271,1274,1276,1284,1316,1321,1324,1341,1356,1383,1384,1386,1388,1413,1424,1427,1432,1435,1462,1468,1482,1483,1502,1549,1551,1564,1566,1575,1589,1601,1602,1611,1628,1657,1660,1662,1663,1674,1682,1691,1693,1695,1704,1706,1710,1714,1722,1725,1730,1732,1736,1739,1745,1749,1760,1804,1806,1831,1842,1848,1853,1859,1866,1886,1892,1896,1898,1908,1916,1927,1930,1944,1948,1960,1967,1975,1986,1991,1993,2016,2026,2028,2030,2035,2039,2051,2052,2061,2063,2068,2069,2076,2088,2092,2095,2108,2153,2154,2175,2205,2206,2213,2219,2248,2254,2258,2262,2266,2276,2288,2303,2308,2317,2343,2344,2361,2371,2378,2379,2384,2407,2411,2421,2428,2434,2455,2467,2471,2472,2480,2487,2500,2516,2534,2537,2539,2545,2551,2552,2566,2573,2576,2582,2584,2621,2627,2634,2663,2673,2675,2676,2682,2700,2721,2723,2728,2729,2730,2749,2770,2801,2809,2819,2829,2841,2842,2843,2853,2863,2874,2903,2916,2920,2922,2940,2942,2943,2949,2950,2961,2963,2972,2987,2990,2999,3008,3014,3021,3029,3031,3054,3069,3074,3084,3093,3094,3096,3106,3108,3120,3125,3128,3132,3134,3139,3141,3157,3197,3201,3223,3230,3237,3242,3244,3265,3266,3268,3273,3280,3289,3305,3309,3313,3342,3348,3350,3353,3359,3363,3377,3402,3412,3414,3415,3416,3419,3427,3434,3454,3458,3488,3499,3505,3512,3522,3531,3540,3541,3571,3581,3589,3602,3606,3632,3660,3668,3671,3679,3686,3695,3710,3711,3718,3722,3738,3741,3746,3747,3750,3763,3791,3803,3808,3810,3848,3854,3876,3880,3893,3895,3910,3911,3918,3926,3932,3944,3969,3998,4011,4016,4032,4039,4061,4063,4070,4077,4104,4114,4124,4127,4137,4143,4156,4165,4166,4183,4185,4191,4205,4212,4224,4236,4252,4285,4297,4306,4311,4341,4350,4359,4362,4365,4368,4376,4377,4378,4380,4381,4389,4400,4408,4414,4444,4454,4456,4458,4470,4473,4484,4487,4512,4526,4538,4549,4566,4618,4630,4650,4662,4667,4671,4678,4681,4684,4685,4691,4713,4728,4729,4746,4748,4760,4762,4771,4795,4803,4815,4827,4830,4857,4875,4889,4902,4903,4909,4917,4919,4928,4935,4959,4961,4962,4965,4980,4982,4988,4992,5002,5006,5030,5036,5037,5042,5051,5052,5079,5083,5088,5090,5092,5093,5101,5157,5164,5169,5188,5206,5211,5213,5237,5241,5242,5249,5258,5261,5264,5267,5278,5282,5291,5297,5298,5299,5302,5312,5335,5344,5352,5406,5422,5424,5426,5432,5433,5447,5452,5484,5515,5523,5535,5552,5571,5598,5606,5609,5621,5625,5640,5661,5679,5691,5705,5745,5753,5772,5773,5787,5798,5799,5820,5822,5827,5833,5840,5848,5888,5905,5915,5917,5926,5940,5941,5968,5970,5971,5972,5985,5994,6002,6008,6011,6067,6068,6075,6095,6106,6108,6112,6113,6117,6141,6157,6161,6164,6186,6198,6221,6243,6273,6280,6311,6316,6317,6319,6322,6341,6343,6347,6349,6356,6361,6379,6384,6409,6415,6417,6436,6443,6444,6447,6469,6488,6493,6503,6514,6516,6546,6554,6561,6562,6565,6596,6621,6630,6639,6640,6647,6657,6662,6670,6677,6695,6697,6700,6701,6704,6708,6717,6731,6758,6769,6771,6775,6798,6807,6834,6845,6848,6849,6864,6876,6880,6886,6892,6923,6927,6942,6947,6958,6961,6966,6967,6969,6973,6978,6979,6982,6985,6990,6991,6996,7001,7002,7005,7007,7014,7024,7025,7044,7052,7071,7076,7087,7091,7093,7094,7096,7100,7119,7121,7131,7132,7137,7161,7183,7207,7208,7218,7220,7227,7229,7235,7241,7245,7266,7271,7279,7280,7285,7295,7300,7309,7322,7324,7328,7365,7366,7369,7403,7405,7423,7429,7431,7437,7441,7443,7449,7461,7482,7484,7491,7514,7525,7527,7531,7534,7536,7537,7551,7564,7566,7572,7576,7593,7605,7617,7620,7629,7631,7640,7655,7662,7666,7672,7680,7684,7696,7702,7719,7722,7744,7753,7757,7758,7759,7770,7788,7791,7819,7821,7828,7835,7842,7851,7857,7861,7868,7897,7899,7919,7922,7938,7941,7954,7960,7963,7969,7976,7982,8002,8004,8015,8026,8031,8057,8069,8089,8106,8112,8121,8123,8127,8135,8171,8182,
        # ] #vsae256
    else:
        # Original parsing logic
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
    # python main.py real --model pythia-70m --features active --sae "C:/Users/WeeSnaw/Desktop/spar2/experiments/VSAETopK_pythia70m_d8192_k256_lr0.0008_kl1.0_aux0_fixed_var/trainer_0/ae.pt"

  # Real activation analysis (uses c4-code dataset like training script)
  python main.py real --model gelu-1l --sae path/to/sae.pt --features 0,1,2,3,4

  # Synthetic maximizer analysis  
  python main.py synthetic --model gelu-1l --sae path/to/sae.pt --features 0,1,2,3,4

  # Both analyses with comparison
  python main.py compare --model gelu-1l --sae path/to/sae.pt --features 0-10

  # Custom buffer settings for real analysis
  python main.py real --model gelu-1l --sae path/to/sae.pt --features 0-50 --buffer-size 2000
"""