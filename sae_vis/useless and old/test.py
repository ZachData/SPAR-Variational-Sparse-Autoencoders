#!/usr/bin/env python3
"""
Test suite for SAE feature analysis components.
"""
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import sys

# Mock classes for testing when real models aren't available
class MockSAE:
    def __init__(self, dict_size=100, activation_dim=512, k=16):
        self.dict_size = dict_size
        self.activation_dim = activation_dim
        self.k = torch.tensor(k)
        # Decoder: sparse features (dict_size) -> original activations (activation_dim)
        self.decoder = torch.nn.Linear(dict_size, activation_dim, bias=False)
        # Encoder: original activations (activation_dim) -> sparse features (dict_size)  
        self.encoder = torch.nn.Linear(activation_dim, dict_size, bias=False)
        
    def parameters(self):
        """Mock parameters method for UnifiedSAEInterface compatibility"""
        for param in self.decoder.parameters():
            yield param
        for param in self.encoder.parameters():
            yield param
    
    def to(self, device):
        """Mock to() method for device movement"""
        self.decoder = self.decoder.to(device)
        self.encoder = self.encoder.to(device)
        return self
        
    def encode(self, x, return_topk=True, training=False):
        batch_size, seq_len, _ = x.shape
        
        # Mock sparse features
        sparse_features = torch.zeros(batch_size, seq_len, self.dict_size)
        
        # Randomly activate some features
        for b in range(batch_size):
            for s in range(seq_len):
                n_active = torch.randint(1, min(10, self.k.item()), (1,)).item()
                active_indices = torch.randperm(self.dict_size)[:n_active]
                sparse_features[b, s, active_indices] = torch.rand(n_active) * 2
        
        # Top-k values and indices
        top_values, top_indices = sparse_features.topk(self.k.item(), dim=-1)
        
        # Pre-activation (mock)
        pre_activation = sparse_features + torch.randn_like(sparse_features) * 0.1
        
        return sparse_features, top_values, top_indices, pre_activation
    
    def decode(self, features):
        # features: (batch, seq, dict_size) -> (batch, seq, activation_dim)
        return self.decoder(features)


class MockTransformer:
    def __init__(self):
        self.cfg = type('Config', (), {
            'd_model': 512,
            'device': 'cpu'
        })()
        
        self.W_E = torch.randn(1000, 512)  # Embedding matrix
        self.W_pos = torch.randn(256, 512)  # Positional embeddings
        self.W_U = torch.randn(512, 1000)  # Unembedding matrix
        self.W_out = [torch.randn(2048, 512)]  # MLP output for layer 0
        
        # Mock tokenizer
        self.tokenizer = type('Tokenizer', (), {
            'decode': lambda self, tokens, skip_special_tokens=True: ' '.join([f'tok_{t}' for t in tokens]),
            'pad_token_id': 0,
            'eos_token_id': 1,
            'bos_token_id': 2,
            'unk_token_id': 3,
        })()
        
        # Mock blocks for forward pass
        self.blocks = [type('Block', (), {
            '__call__': lambda self, x: x + torch.randn_like(x) * 0.1
        })()]
    
    def parameters(self):
        """Yield parameters like a real PyTorch module"""
        yield self.W_E
        yield self.W_pos
        yield self.W_U
        for w_out in self.W_out:
            yield w_out
    
    def run_with_cache(self, input_ids, stop_at_layer=None):
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(batch_size, seq_len, 512)
        
        cache = {
            'blocks.0.hook_resid_post': hidden_states,
            'blocks.0.hook_resid_pre': hidden_states,
            'blocks.0.mlp.hook_post': torch.randn(batch_size, seq_len, 2048),
            'hook_embed': hidden_states
        }
        
        return None, cache


def test_sae_interface():
    """Test the UnifiedSAEInterface"""
    print("Testing SAE Interface...")
    
    try:
        from sae_interface import UnifiedSAEInterface
        
        # Test with mock SAE
        mock_sae = MockSAE()
        interface = UnifiedSAEInterface(mock_sae)
        
        # Test basic properties
        assert interface.dict_size == 100
        assert interface.activation_dim == 512
        assert interface.k == 16
        
        # Test encoding
        test_input = torch.randn(2, 10, 512)
        output = interface.encode(test_input)
        
        assert output.sparse_features.shape == (2, 10, 100)
        assert output.top_values.shape == (2, 10, 16)
        assert output.top_indices.shape == (2, 10, 16)
        
        # Test decoding
        decoded = interface.decode(output.sparse_features)
        assert decoded.shape == test_input.shape
        
        print("✅ SAE Interface tests passed")
        return True
        
    except Exception as e:
        print(f"❌ SAE Interface test failed: {e}")
        return False


def test_real_activations():
    """Test real activation analysis with buffer functionality"""
    print("Testing Real Activations...")
    
    try:
        from real_activations import analyze_real_activations
        
        # Create mock models
        mock_transformer = MockTransformer()
        mock_sae = MockSAE()
        
        # Mock the activation buffer functionality
        class MockActivationBuffer:
            def __init__(self, data, model, hook_name, d_submodule, n_ctxs, ctx_len, 
                        refresh_batch_size, out_batch_size, device):
                self.n_ctxs = n_ctxs
                self.ctx_len = ctx_len
                self.device = device
                self.batch_count = 0
                
            def __iter__(self):
                return self
                
            def __next__(self):
                if self.batch_count >= 3:  # Limit to 3 batches for testing
                    raise StopIteration
                
                self.batch_count += 1
                # Return mock activations with shape [batch_size, seq_len, d_model]
                batch_size = min(4, self.n_ctxs // 10)
                return torch.randn(batch_size, self.ctx_len, 512)  # d_model = 512
            
            def __len__(self):
                return self.n_ctxs
        
        # Mock the data generator
        def mock_hf_dataset_to_generator(dataset_name, split, return_tokens=True):
            """Mock data generator"""
            def generator():
                for i in range(100):
                    yield {"text": f"mock programming code {i}", "tokens": [f"token_{j}" for j in range(10)]}
            return generator()
        
        # Patch the imports
        import real_activations
        original_create_buffer = None
        original_hf_generator = None
        
        # Mock create_activation_buffer
        def mock_create_activation_buffer(model, device, buffer_size=1000, ctx_len=128):
            print(f"Mock creating buffer: size={buffer_size}, ctx_len={ctx_len}")
            return MockActivationBuffer(
                data=None, model=model, hook_name="blocks.0.hook_resid_post",
                d_submodule=512, n_ctxs=buffer_size, ctx_len=ctx_len,
                refresh_batch_size=16, out_batch_size=64, device=device
            )
        
        # Apply patches
        if hasattr(real_activations, 'create_activation_buffer'):
            original_create_buffer = real_activations.create_activation_buffer
            real_activations.create_activation_buffer = mock_create_activation_buffer
        
        try:
            # Also try to patch the hf_dataset_to_generator if it's imported
            if hasattr(real_activations, 'hf_dataset_to_generator'):
                original_hf_generator = real_activations.hf_dataset_to_generator
                real_activations.hf_dataset_to_generator = mock_hf_dataset_to_generator
        except:
            pass  # Might not be imported yet
        
        try:
            # Test parameters
            feature_indices = [0, 1, 2]
            
            # Run analysis with new buffer-based approach
            results = analyze_real_activations(
                sae_model=mock_sae,
                transformer_model=mock_transformer,
                feature_indices=feature_indices,
                max_examples=5,
                batch_size=2,
                buffer_size=100,  # Small buffer for testing
                ctx_len=32
            )
            
            # Verify results
            assert len(results) == 3
            for feature_idx in feature_indices:
                assert feature_idx in results
                stats = results[feature_idx]
                assert stats.feature_idx == feature_idx
                assert isinstance(stats.sparsity, float)
                assert isinstance(stats.max_activation, float)
                assert isinstance(stats.examples, list)
            
            print("✅ Real Activations tests passed")
            return True
            
        finally:
            # Restore original functions
            if original_create_buffer is not None:
                real_activations.create_activation_buffer = original_create_buffer
            if original_hf_generator is not None:
                real_activations.hf_dataset_to_generator = original_hf_generator
        
    except Exception as e:
        print(f"❌ Real Activations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthetic_activations():
    """Test synthetic maximizer generation"""
    print("Testing Synthetic Activations...")
    
    try:
        from synth_activations import generate_synthetic_maximizers
        
        # Create mock models
        mock_transformer = MockTransformer()
        mock_sae = MockSAE()
        
        # Test with minimal parameters for speed
        feature_indices = [0, 1]
        
        results = generate_synthetic_maximizers(
            sae_model=mock_sae,
            transformer_model=mock_transformer,
            feature_indices=feature_indices,
            max_examples=2,
            seq_len=16,
            n_steps=10,  # Very few steps for testing
            learning_rate=0.5
        )
        
        # Verify results
        assert len(results) == 2
        for feature_idx in feature_indices:
            assert feature_idx in results
            stats = results[feature_idx]
            assert stats.feature_idx == feature_idx
            assert isinstance(stats.examples, list)
            assert len(stats.examples) <= 2
        
        print("✅ Synthetic Activations tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic Activations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizer():
    """Test visualization generation"""
    print("Testing Visualizer...")
    
    try:
        from visualizer import visualize_features
        from real_activations import FeatureStats, ActivationExample
        
        # Create mock data
        mock_examples = [
            ActivationExample(
                text="Test example text",
                tokens=["Test", "example", "text"],
                token_ids=[100, 200, 300],
                activations=[0.5, 1.2, 0.3],
                max_activation=1.2,
                peak_position=1
            )
        ]
        
        mock_stats = {
            0: FeatureStats(
                feature_idx=0,
                examples=mock_examples,
                sparsity=0.95,
                mean_activation=0.8,
                max_activation=1.2,
                decoder_norm=2.1,
                top_boosted_tokens=[("hello", 0.5), ("world", 0.3)],
                top_suppressed_tokens=[("bad", -0.4), ("evil", -0.2)]
            ),
            1: FeatureStats(
                feature_idx=1,
                examples=mock_examples[:1],  # Less examples
                sparsity=0.98,
                mean_activation=0.6,
                max_activation=0.9,
                decoder_norm=1.8,
                top_boosted_tokens=[("good", 0.4), ("nice", 0.2)],
                top_suppressed_tokens=[("terrible", -0.5), ("awful", -0.3)]
            )
        }
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test visualization
            image_paths = visualize_features(
                feature_stats=mock_stats,
                output_dir=temp_dir,
                analysis_type="test"
            )
            
            # Verify files were created
            assert len(image_paths) > 0
            for path in image_paths:
                assert Path(path).exists()
                assert Path(path).suffix == '.png'
        
        print("✅ Visualizer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_imports():
    """Test that main.py can import all dependencies"""
    print("Testing Main Imports...")
    
    try:
        # Test imports (but don't run main)
        import main
        
        # Test mock model loading
        mock_transformer = MockTransformer() 
        mock_sae = MockSAE()
        
        # Test that main.py loads successfully (it doesn't need load_text_corpus)
        assert hasattr(main, 'main')
        assert hasattr(main, 'load_models')
        assert hasattr(main, 'run_real_analysis')
        assert hasattr(main, 'run_synthetic_analysis')
        
        print("✅ Main imports tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Main imports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Integration test with all components"""
    print("Testing Integration...")
    
    try:
        # Import all components
        from real_activations import analyze_real_activations
        from synth_activations import generate_synthetic_maximizers
        from visualizer import visualize_features, compare_real_vs_synthetic
        
        # Create mock models
        mock_transformer = MockTransformer()
        mock_sae = MockSAE()
        
        # Mock the buffer system for real analysis
        import real_activations
        
        class MockActivationBuffer:
            def __init__(self, data, model, hook_name, d_submodule, n_ctxs, ctx_len, 
                        refresh_batch_size, out_batch_size, device):
                self.n_ctxs = n_ctxs
                self.ctx_len = ctx_len
                self.batch_count = 0
                
            def __iter__(self):
                return self
                
            def __next__(self):
                if self.batch_count >= 2:
                    raise StopIteration
                self.batch_count += 1
                return torch.randn(2, self.ctx_len, 512)
            
            def __len__(self):
                return self.n_ctxs
        
        def mock_create_activation_buffer(model, device, buffer_size=1000, ctx_len=128):
            return MockActivationBuffer(
                data=None, model=model, hook_name="blocks.0.hook_resid_post",
                d_submodule=512, n_ctxs=buffer_size, ctx_len=ctx_len,
                refresh_batch_size=16, out_batch_size=64, device=device
            )
        
        original_create_buffer = real_activations.create_activation_buffer
        real_activations.create_activation_buffer = mock_create_activation_buffer
        
        try:
            # Test data
            feature_indices = [0, 1]
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run real analysis (using correct signature)
                real_results = analyze_real_activations(
                    sae_model=mock_sae,
                    transformer_model=mock_transformer,
                    feature_indices=feature_indices,
                    max_examples=3,
                    batch_size=1,
                    buffer_size=50,
                    ctx_len=20
                )
                
                # Run synthetic analysis
                synthetic_results = generate_synthetic_maximizers(
                    sae_model=mock_sae,
                    transformer_model=mock_transformer,
                    feature_indices=feature_indices,
                    max_examples=2,
                    seq_len=10,
                    n_steps=5,
                    learning_rate=0.5
                )
                
                # Create visualizations
                real_images = visualize_features(real_results, f"{temp_dir}/real", "real")
                synthetic_images = visualize_features(synthetic_results, f"{temp_dir}/synthetic", "synthetic")
                comparison_images = compare_real_vs_synthetic(real_results, synthetic_results, f"{temp_dir}/comparison")
                
                # Verify all images were created
                all_images = real_images + synthetic_images + comparison_images
                assert len(all_images) > 0
                
                for image_path in all_images:
                    assert Path(image_path).exists()
            
            print("✅ Integration tests passed")
            return True
            
        finally:
            real_activations.create_activation_buffer = original_create_buffer
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test suites"""
    print("🧪 Running SAE Feature Analysis Test Suite")
    print("=" * 50)
    
    tests = [
        ("SAE Interface", test_sae_interface),
        ("Real Activations", test_real_activations),
        ("Synthetic Activations", test_synthetic_activations),
        ("Visualizer", test_visualizer),
        ("Main Imports", test_main_imports),
        ("Integration", test_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your SAE analysis system is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)