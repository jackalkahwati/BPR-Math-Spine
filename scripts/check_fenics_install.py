#!/usr/bin/env python3
"""
FEniCS Installation Test Script

Tests whether FEniCS is properly installed and working with BPR-Math-Spine.

Usage:
    python scripts/test_fenics_install.py
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic Python package imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import numpy as np
        print("  ✅ numpy")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
        return False
    
    try:
        import scipy
        print("  ✅ scipy")
    except ImportError as e:
        print(f"  ❌ scipy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  ✅ matplotlib")
    except ImportError as e:
        print(f"  ❌ matplotlib: {e}")
        return False
        
    try:
        import sympy as sp
        print("  ✅ sympy")
    except ImportError as e:
        print(f"  ❌ sympy: {e}")
        return False
    
    return True


def test_fenics_import():
    """Test FEniCS import."""
    print("\n🔍 Testing FEniCS import...")
    
    fenics_available = False
    
    try:
        import fenics as fe
        print("  ✅ fenics (legacy)")
        fenics_available = True
    except ImportError as e:
        print(f"  ❌ fenics (legacy): {e}")
    
    try:
        import mshr
        print("  ✅ mshr")
    except ImportError as e:
        print(f"  ❌ mshr: {e}")
    
    # Test FEniCSX as alternative
    try:
        import dolfinx
        print("  ✅ dolfinx (FEniCSX)")
        fenics_available = True
    except ImportError as e:
        print(f"  ❌ dolfinx (FEniCSX): {e}")
    
    return fenics_available


def test_basic_fenics_functionality():
    """Test basic FEniCS functionality."""
    print("\n🔍 Testing basic FEniCS functionality...")
    
    try:
        import fenics as fe
        
        # Create simple mesh
        mesh = fe.UnitSquareMesh(4, 4)
        print(f"  ✅ Created 2D mesh with {mesh.num_vertices()} vertices")
        
        # Create function space
        V = fe.FunctionSpace(mesh, "CG", 1)
        print(f"  ✅ Created function space with {V.dim()} DOFs")
        
        # Simple function
        u = fe.Function(V)
        print("  ✅ Created function")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FEniCS functionality test failed: {e}")
        return False


def test_bpr_package():
    """Test BPR package import."""
    print("\n🔍 Testing BPR package...")
    
    try:
        import bpr
        print("  ✅ bpr package")
        
        # Test main functions
        available_functions = bpr.__all__
        print(f"  ✅ Available functions: {', '.join(available_functions)}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ bpr package: {e}")
        return False


def test_bpr_with_fenics():
    """Test BPR package with FEniCS integration."""
    print("\n🔍 Testing BPR-FEniCS integration...")
    
    try:
        from bpr import make_boundary
        
        # Test boundary mesh creation
        boundary_mesh = make_boundary(mesh_size=0.3, geometry="sphere", radius=1.0)
        print("  ✅ Boundary mesh creation")
        
        # Test mesh quality
        from bpr.geometry import get_mesh_quality_metrics
        quality = get_mesh_quality_metrics(boundary_mesh)
        print(f"  ✅ Mesh quality: {quality.get('num_vertices', 'N/A')} vertices")
        
        return True
        
    except Exception as e:
        print(f"  ❌ BPR-FEniCS integration: {e}")
        return False


def test_symbolic_functionality():
    """Test symbolic computation without FEniCS."""
    print("\n🔍 Testing symbolic functionality (FEniCS-independent)...")
    
    try:
        from bpr.metric import metric_perturbation
        import sympy as sp
        
        # Create symbolic field
        x, y, z = sp.symbols('x y z', real=True)
        phi_field = sp.sin(x) * sp.cos(y)
        
        # Test metric perturbation
        delta_g = metric_perturbation(phi_field, coupling_lambda=0.1)
        print("  ✅ Metric perturbation computation")
        
        # Test stress tensor
        from bpr.metric import compute_stress_tensor
        T_stress = compute_stress_tensor(phi_field, delta_g.delta_g, delta_g.coordinates)
        print("  ✅ Stress tensor computation")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Symbolic functionality: {e}")
        return False


def main():
    """Main test runner."""
    print("=" * 60)
    print("BPR-Math-Spine: FEniCS Installation Test")
    print("=" * 60)
    
    # Track test results
    results = {}
    
    # Test 1: Basic imports
    results['basic_imports'] = test_basic_imports()
    
    # Test 2: FEniCS import
    results['fenics_import'] = test_fenics_import()
    
    # Test 3: BPR package
    results['bpr_package'] = test_bpr_package()
    
    # Test 4: Symbolic functionality (always works)
    results['symbolic'] = test_symbolic_functionality()
    
    # Test 5: FEniCS functionality (if available)
    if results['fenics_import']:
        results['fenics_functionality'] = test_basic_fenics_functionality()
        results['bpr_fenics'] = test_bpr_with_fenics()
    else:
        results['fenics_functionality'] = False
        results['bpr_fenics'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        test_display = test_name.replace('_', ' ').title()
        print(f"{test_display:25s} {status}")
    
    print("-" * 60)
    
    # Overall assessment
    critical_tests = ['basic_imports', 'bpr_package', 'symbolic']
    fenics_tests = ['fenics_import', 'fenics_functionality', 'bpr_fenics']
    
    critical_passed = all(results[test] for test in critical_tests)
    fenics_passed = all(results[test] for test in fenics_tests)
    
    if critical_passed and fenics_passed:
        print("🎉 ALL TESTS PASSED")
        print("   Full BPR-Math-Spine functionality available!")
        print("   You can run: python scripts/run_casimir_demo.py")
        
    elif critical_passed:
        print("⚠️  PARTIAL FUNCTIONALITY")
        print("   Core BPR package works, but FEniCS not available.")
        print("   You can run: python scripts/run_casimir_demo.py --no-fenics")
        print("\n💡 To install FEniCS:")
        print("   conda install fenics -c conda-forge")
        print("   # or try: conda env create -f environment-fenicsx.yml")
        
    else:
        print("❌ CRITICAL FAILURE")
        print("   Basic dependencies missing. Please install required packages:")
        print("   conda env create -f environment-minimal.yml")
    
    print("\n📚 For more installation help, see README.md")
    print("=" * 60)
    
    return critical_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)