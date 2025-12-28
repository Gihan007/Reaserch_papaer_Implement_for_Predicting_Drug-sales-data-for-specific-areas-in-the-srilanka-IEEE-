
# Meta-Learning Demo for Drug Sales Prediction
# This demonstrates the three meta-learning approaches implemented

import sys
import os
sys.path.append('./src')

from models.meta_learning import MetaLearningSystem
import torch

def demo_meta_learning():
    """Demonstrate meta-learning capabilities"""

    print("Drug Sales Meta-Learning Demo")
    print("=" * 40)

    # Initialize system
    meta_system = MetaLearningSystem()

    # 1. MAML Training
    print("\n1. Training MAML Model...")
    categories = ['C1', 'C2', 'C3', 'C4']
    maml_model = meta_system.train_maml(categories, n_epochs=5)
    print("MAML training completed")

    # 2. Few-shot Adaptation
    print("\n2. Few-shot Adaptation to new category...")
    adapted_model, scaler = meta_system.few_shot_adaptation(
        target_category='C5',
        support_samples=10,
        adaptation_steps=20
    )
    print("Few-shot adaptation completed")

    # 3. Transfer Learning
    print("\n3. Transfer Learning between categories...")
    transfer_model, scaler = meta_system.transfer_learning(
        source_category='C1',
        target_category='C6',
        fine_tune_steps=50
    )
    print("Transfer learning completed")

    # 4. Make Predictions
    print("\n4. Making Predictions with Meta-Learned Models...")

    test_categories = ['C7', 'C8']
    for category in test_categories:
        # MAML prediction
        pred_maml = meta_system.predict_with_meta_model(category, 'maml')
        print(f"{category} MAML prediction: {pred_maml:.2f}")

        # Few-shot prediction
        pred_fewshot = meta_system.predict_with_meta_model(category, 'few_shot')
        print(f"{category} Few-shot prediction: {pred_fewshot:.2f}")

        # Transfer prediction
        pred_transfer = meta_system.predict_with_meta_model(category, 'transfer')
        print(f"{category} Transfer prediction: {pred_transfer:.2f}")

    print("\nMeta-Learning Demo Completed!")
    print("\nKey Features Demonstrated:")
    print("• Model-Agnostic Meta-Learning (MAML)")
    print("• Few-shot Learning Adaptation")
    print("• Transfer Learning across Categories")
    print("• Cross-category Knowledge Transfer")

if __name__ == "__main__":
    demo_meta_learning()
