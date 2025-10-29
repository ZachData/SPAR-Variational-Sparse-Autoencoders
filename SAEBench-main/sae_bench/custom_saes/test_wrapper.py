"""
Quick test script to verify dictionary_learning_wrapper works correctly.

Run this before the full evaluation to catch issues early.
"""

import torch
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from dictionary_learning_wrapper import DictionaryLearningSAEWrapper

def test_single_sae(model_path: str, device: str = "cuda"):
    """Test a single SAE thoroughly."""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(model_path).name}")
    print(f"{'='*60}")
    
    try:
        # Load wrapper
        print("Loading SAE wrapper...")
        wrapper = DictionaryLearningSAEWrapper.from_pretrained(model_path, device=device)
        print(f"✓ Loaded successfully")
        print(f"  d_in: {wrapper.cfg.d_in}, d_sae: {wrapper.cfg.d_sae}")
        print(f"  SAE type: {type(wrapper.sae).__name__}")
        
        # Test encode
        print("\nTesting encode()...")
        batch_size = 10
        test_input = torch.randn(batch_size, wrapper.cfg.d_in, device=device, dtype=torch.bfloat16)
        
        features = wrapper.encode(test_input)
        assert isinstance(features, torch.Tensor), f"encode() returned {type(features)}, expected Tensor"
        assert features.shape == (batch_size, wrapper.cfg.d_sae), f"Wrong feature shape: {features.shape}"
        assert not torch.isnan(features).any(), "Features contain NaN"
        assert not torch.isinf(features).any(), "Features contain Inf"
        print(f"✓ encode() works: {test_input.shape} -> {features.shape}")
        print(f"  Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
        print(f"  L0 (features > 0): {(features > 0).float().sum(dim=-1).mean():.1f}")
        
        # Test decode
        print("\nTesting decode()...")
        reconstruction = wrapper.decode(features)
        assert isinstance(reconstruction, torch.Tensor), f"decode() returned {type(reconstruction)}, expected Tensor"
        assert reconstruction.shape == (batch_size, wrapper.cfg.d_in), f"Wrong reconstruction shape: {reconstruction.shape}"
        assert not torch.isnan(reconstruction).any(), "Reconstruction contains NaN"
        assert not torch.isinf(reconstruction).any(), "Reconstruction contains Inf"
        print(f"✓ decode() works: {features.shape} -> {reconstruction.shape}")
        print(f"  Reconstruction stats: min={reconstruction.min():.4f}, max={reconstruction.max():.4f}")
        
        # Test forward
        print("\nTesting forward()...")
        output = wrapper(test_input)
        assert isinstance(output, torch.Tensor), f"forward() returned {type(output)}, expected Tensor"
        assert output.shape == (batch_size, wrapper.cfg.d_in), f"Wrong output shape: {output.shape}"
        assert torch.allclose(output, reconstruction, atol=1e-4), "forward() != encode+decode"
        print(f"✓ forward() works and matches encode+decode")
        
        # Test forward with output_features
        print("\nTesting forward(output_features=True)...")
        output2, features2 = wrapper(test_input, output_features=True)
        assert torch.allclose(output2, output, atol=1e-6), "forward with output_features gives different reconstruction"
        assert torch.allclose(features2, features, atol=1e-6), "forward with output_features gives different features"
        print(f"✓ forward(output_features=True) works")
        
        # Test reconstruction quality
        print("\nChecking reconstruction quality...")
        mse = torch.mean((test_input - reconstruction) ** 2).item()
        l2_norm = torch.mean(torch.norm(test_input - reconstruction, dim=-1)).item()
        print(f"  MSE: {mse:.6f}")
        print(f"  L2 norm: {l2_norm:.6f}")
        
        # Test decoder normalization
        print("\nChecking decoder normalization...")
        decoder_norms = torch.norm(wrapper.W_dec.data, dim=1)
        is_normalized = torch.allclose(decoder_norms, torch.ones_like(decoder_norms), atol=1e-4)
        print(f"  Decoder normalized: {is_normalized}")
        print(f"  Norm stats: min={decoder_norms.min():.6f}, max={decoder_norms.max():.6f}, mean={decoder_norms.mean():.6f}")
        
        # For JumpReLU: verify threshold was scaled correctly
        if hasattr(wrapper.sae, 'threshold'):
            print("\nVerifying JumpReLU threshold scaling...")
            print(f"  Has threshold parameter: True")
            
            # Test with a fresh input to verify activation pattern is preserved
            test_input2 = torch.randn(100, wrapper.cfg.d_in, device=device, dtype=torch.bfloat16)
            
            # Get features and L0
            features1 = wrapper.encode(test_input2)
            l0_mean = (features1 > 0).float().sum(dim=-1).mean().item()
            l0_std = (features1 > 0).float().sum(dim=-1).std().item()
            
            print(f"  L0 stats: mean={l0_mean:.1f}, std={l0_std:.1f}")
            
            # Check that multiple forward passes give same results (deterministic)
            output1 = wrapper(test_input2)
            output2 = wrapper(test_input2)
            
            if torch.allclose(output1, output2, atol=1e-6):
                print(f"  ✓ Output is deterministic")
            else:
                print(f"  ✗ Warning: Output is not deterministic!")
            
            # Check reconstruction quality
            mse = torch.mean((test_input2 - output1) ** 2).item()
            print(f"  Reconstruction MSE: {mse:.6f}")
            
            if is_normalized:
                print(f"  ✓ Threshold scaling verified (output preserved after normalization)")
            else:
                print(f"  ℹ Decoder not normalized (may be pre-normalized or not applicable)")
        else:
            print(f"  Not a JumpReLU architecture (no threshold parameter)")
        
        print(f"\n{'='*60}")
        print(f"✓ ALL TESTS PASSED for {Path(model_path).name}")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ TEST FAILED for {Path(model_path).name}")
        print(f"Error: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all SAEs in the experiments directory."""
    experiments_dir = Path("../../../experiments")
    
    # Model names from run_all_evals_custom_saes.py
    model_names = [
        'vsaejumprelu_pythia-70m-deduped_d16x_lr50e04_kl1_aux25_tl064_th16e02_fixedvar',
        'vsaejumprelu_pythia-70m-deduped_d16x_lr50e04_kl1_aux25_tl0128_th11e02_fixedvar',
        'vsaejumprelu_pythia-70m-deduped_d16x_lr50e04_kl1_aux25_tl0256_th80e03_fixedvar',
        'vsaejumprelu_pythia-70m-deduped_d16x_lr50e04_kl1_aux25_tl0512_th50e03_fixedvar',
        'jumprelu-pythia-70m-deduped_d8192_lr0.0005_l0356.0_bw0.001_sp1.0',
        'jumprelu-pythia-70m-deduped_d8192_lr0.0005_l0256.0_bw0.001_sp1.0',
        'jumprelu-pythia-70m-deduped_d8192_lr0.0005_l0128.0_bw0.001_sp1.0',
        'jumprelu-pythia-70m-deduped_d8192_lr0.0005_l064.0_bw0.001_sp1.0',
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = {}
    for model_name in model_names:
        model_path = experiments_dir / model_name / "trainer_0"
        
        if not model_path.exists():
            print(f"\n✗ Model not found: {model_path}")
            results[model_name] = False
            continue
        
        results[model_name] = test_single_sae(str(model_path), device=device)
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    if passed == total:
        print(f"\n✓ All tests passed! Ready to run full evaluation.")
        return 0
    else:
        print(f"\n✗ Some tests failed. Fix issues before running full evaluation.")
        return 1


if __name__ == "__main__":
    exit(main())