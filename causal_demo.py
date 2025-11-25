#!/usr/bin/env python3
"""
Causal Inference Demo for Drug Sales Prediction
Demonstrates the causal inference capabilities of the system
"""

import sys
import os
sys.path.append('.')

from src.models.advanced.causal_inference import CausalInferenceEngine
import pandas as pd
import numpy as np

def main():
    """Demonstrate causal inference capabilities"""
    print("🔍 Drug Sales Prediction - Causal Inference Demo")
    print("=" * 60)

    # Initialize the causal inference engine
    print("Initializing Causal Inference Engine...")
    engine = CausalInferenceEngine()
    print("✅ Engine initialized successfully\n")

    # Test causal discovery
    print("🔬 Testing Causal Discovery...")
    try:
        results = engine.discover_causal_relationships('C1', max_lags=3)
        print("✅ Causal discovery completed")
        print(f"   - Analyzed {results['data_points']} data points")
        print(f"   - Found {len(results['causal_drivers'])} causal drivers")
        print(f"   - Granger causality tests: {len(results['granger_causality'])}")
        print()
    except Exception as e:
        print(f"❌ Causal discovery failed: {e}\n")

    # Test effect estimation
    print("📊 Testing Causal Effect Estimation...")
    try:
        results = engine.estimate_causal_effects('C1', 'sales_lag1')
        print("✅ Effect estimation completed")
        print(f"   - Treatment variable: sales_lag1")
        print(f"   - Available estimation methods: {list(results['estimates'].keys())}")
        print()
    except Exception as e:
        print(f"❌ Effect estimation failed: {e}\n")

    # Test counterfactual analysis
    print("🔮 Testing Counterfactual Analysis...")
    try:
        results = engine.counterfactual_analysis('C1', 'sales_lag1', 20)
        print("✅ Counterfactual analysis completed")
        print(f"   - Intervention: 20% increase in sales_lag1")
        print(".1f")
        print(".2f")
        print()
    except Exception as e:
        print(f"❌ Counterfactual analysis failed: {e}\n")

    # Test complete analysis
    print("🎯 Testing Complete Causal Analysis...")
    try:
        results = engine.complete_causal_analysis('C1')
        print("✅ Complete causal analysis completed")
        print(f"   - Category: {results['category']}")
        print(f"   - Data points: {results['data_points']}")
        print(f"   - Causal relationships discovered: {len(results['causal_relationships']['causal_drivers'])}")
        print(f"   - Effect estimates: {len(results['causal_effects'])}")
        print(f"   - Counterfactual scenarios: {len(results['counterfactuals'])}")
        if results['causal_graph_path']:
            print(f"   - Causal graph saved: {results['causal_graph_path']}")
        print()
    except Exception as e:
        print(f"❌ Complete analysis failed: {e}\n")

    print("🎉 Causal Inference Demo Completed!")
    print("\nKey Features Demonstrated:")
    print("• Causal Discovery - Identify what causes sales changes")
    print("• Effect Estimation - Quantify causal impact")
    print("• Counterfactual Analysis - What-if scenarios")
    print("• Complete Pipeline - End-to-end causal analysis")
    print("\nResearch Applications:")
    print("• Move beyond correlation to causation")
    print("• Enable evidence-based pharmaceutical decisions")
    print("• Support policy-making with causal evidence")
    print("• Publishable in top AI/ML conferences")

if __name__ == "__main__":
    main()