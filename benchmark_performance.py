#!/usr/bin/env python3
"""
Drug Sales Prediction System - Performance Benchmarking Suite
Comprehensive benchmarking for model accuracy, speed, and scalability
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import psutil
import GPUtil
from contextlib import contextmanager
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from forecast_utils import forecast_sales
from src.models.meta_learning import MetaLearningSystem
from src.evaluation.ensemble_methods import EnsemblePredictor
import src.models.transformer_model as tm
import src.models.lstm_model as lm

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""

    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/plots", exist_ok=True)

    @contextmanager
    def measure_resources(self, operation_name: str):
        """Context manager to measure CPU, memory, and GPU usage"""
        # Start monitoring
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent

        gpu_start = None
        if GPUtil.getGPUs():
            gpu_start = GPUtil.getGPUs()[0].memoryUsed

        yield

        # End monitoring
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent

        gpu_end = None
        if GPUtil.getGPUs():
            gpu_end = GPUtil.getGPUs()[0].memoryUsed

        # Calculate metrics
        duration = end_time - start_time
        cpu_usage = (start_cpu + end_cpu) / 2
        memory_delta = end_memory - start_memory
        gpu_memory_delta = gpu_end - gpu_start if gpu_end and gpu_start else 0

        self.results[operation_name] = {
            'duration': duration,
            'cpu_usage': cpu_usage,
            'memory_delta': memory_delta,
            'gpu_memory_delta': gpu_memory_delta
        }

    def benchmark_model_accuracy(self, categories: List[str] = ['C1', 'C2', 'C3'],
                               models: List[str] = ['ensemble', 'transformer', 'lstm', 'xgboost', 'sarimax'],
                               test_dates: List[str] = ['2023-12-01', '2024-01-01', '2024-02-01']):
        """Benchmark model accuracy across categories and time periods"""

        print("🔬 Benchmarking Model Accuracy...")
        accuracy_results = {}

        for category in categories:
            print(f"  📊 Testing category: {category}")
            category_results = {}

            for model in models:
                print(f"    🤖 Testing model: {model}")
                model_results = []

                for test_date in test_dates:
                    try:
                        with self.measure_resources(f"forecast_{category}_{model}_{test_date}"):
                            forecast, actual_date, plot_path, model_name = forecast_sales(
                                category, test_date, model
                            )

                        # Load actual data for comparison (assuming data structure)
                        actual_value = self._get_actual_sales(category, actual_date)

                        if actual_value is not None:
                            mae = abs(forecast - actual_value)
                            mape = abs((forecast - actual_value) / actual_value) * 100 if actual_value != 0 else 0

                            model_results.append({
                                'date': test_date,
                                'forecast': forecast,
                                'actual': actual_value,
                                'mae': mae,
                                'mape': mape,
                                'duration': self.results[f"forecast_{category}_{model}_{test_date}"]['duration']
                            })

                    except Exception as e:
                        print(f"      ❌ Error with {model} on {test_date}: {str(e)}")
                        continue

                if model_results:
                    category_results[model] = {
                        'predictions': model_results,
                        'avg_mae': np.mean([r['mae'] for r in model_results]),
                        'avg_mape': np.mean([r['mape'] for r in model_results]),
                        'avg_duration': np.mean([r['duration'] for r in model_results])
                    }

            accuracy_results[category] = category_results

        self.results['accuracy_benchmark'] = accuracy_results
        return accuracy_results

    def benchmark_meta_learning(self, categories: List[str] = ['C1', 'C2', 'C3', 'C4']):
        """Benchmark meta-learning performance"""

        print("🧠 Benchmarking Meta-Learning...")

        meta_results = {}

        # Initialize meta-learning system
        with self.measure_resources("meta_learning_init"):
            meta_system = MetaLearningSystem()

        # Train MAML
        with self.measure_resources("meta_learning_train"):
            maml_model = meta_system.train_maml(categories[:3])  # Train on first 3 categories

        meta_results['training_time'] = self.results['meta_learning_train']['duration']

        # Test transfer learning
        transfer_results = []
        for target_category in categories[3:]:  # Test on remaining categories
            print(f"  🔄 Testing transfer to {target_category}")

            with self.measure_resources(f"transfer_{target_category}"):
                transfer_model = meta_system.transfer_learning(categories[0], target_category)

            # Test few-shot adaptation
            few_shot_results = []
            for shots in [1, 5, 10, 20]:
                try:
                    with self.measure_resources(f"few_shot_{target_category}_{shots}"):
                        adapted_model = meta_system.few_shot_adapt(target_category, shots)

                    few_shot_results.append({
                        'shots': shots,
                        'adaptation_time': self.results[f"few_shot_{target_category}_{shots}"]['duration']
                    })
                except Exception as e:
                    print(f"    ❌ Few-shot adaptation failed for {shots} shots: {str(e)}")

            transfer_results.append({
                'target_category': target_category,
                'transfer_time': self.results[f"transfer_{target_category}"]['duration'],
                'few_shot_results': few_shot_results
            })

        meta_results['transfer_results'] = transfer_results
        self.results['meta_learning_benchmark'] = meta_results
        return meta_results

    def benchmark_scalability(self, concurrent_requests: List[int] = [1, 5, 10, 20, 50]):
        """Benchmark system scalability under concurrent load"""

        print("⚡ Benchmarking Scalability...")

        import concurrent.futures
        import threading

        scalability_results = {}

        def single_request(category: str = 'C1', date: str = '2024-01-01', model: str = 'ensemble'):
            """Single forecast request for threading"""
            try:
                start_time = time.time()
                forecast, _, _, _ = forecast_sales(category, date, model)
                end_time = time.time()
                return end_time - start_time, forecast
            except Exception as e:
                return None, str(e)

        for num_requests in concurrent_requests:
            print(f"  🚀 Testing {num_requests} concurrent requests")

            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [
                    executor.submit(single_request, 'C1', '2024-01-01', 'ensemble')
                    for _ in range(num_requests)
                ]

                results = []
                for future in concurrent.futures.as_completed(futures):
                    duration, result = future.result()
                    if duration is not None:
                        results.append(duration)

            total_time = time.time() - start_time

            if results:
                scalability_results[num_requests] = {
                    'total_time': total_time,
                    'avg_response_time': np.mean(results),
                    'min_response_time': np.min(results),
                    'max_response_time': np.max(results),
                    'success_rate': len(results) / num_requests,
                    'requests_per_second': num_requests / total_time
                }

        self.results['scalability_benchmark'] = scalability_results
        return scalability_results

    def benchmark_memory_usage(self, models: List[str] = ['transformer', 'lstm', 'ensemble']):
        """Benchmark memory usage of different models"""

        print("💾 Benchmarking Memory Usage...")

        memory_results = {}

        for model in models:
            print(f"  📈 Testing memory usage for {model}")

            # Force garbage collection before test
            import gc
            gc.collect()

            initial_memory = psutil.virtual_memory().used

            try:
                with self.measure_resources(f"memory_{model}"):
                    # Load model and make multiple predictions
                    for _ in range(10):
                        forecast_sales('C1', '2024-01-01', model)

                peak_memory = psutil.virtual_memory().used
                memory_usage = peak_memory - initial_memory

                memory_results[model] = {
                    'memory_usage_mb': memory_usage / (1024 * 1024),
                    'duration': self.results[f"memory_{model}"]['duration']
                }

            except Exception as e:
                print(f"    ❌ Memory test failed for {model}: {str(e)}")
                memory_results[model] = {'error': str(e)}

        self.results['memory_benchmark'] = memory_results
        return memory_results

    def generate_reports(self):
        """Generate comprehensive benchmark reports and visualizations"""

        print("📊 Generating Benchmark Reports...")

        # Save raw results
        results_file = f"{self.results_dir}/benchmark_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Generate accuracy comparison plot
        if 'accuracy_benchmark' in self.results:
            self._plot_accuracy_comparison()

        # Generate meta-learning performance plot
        if 'meta_learning_benchmark' in self.results:
            self._plot_meta_learning_performance()

        # Generate scalability plot
        if 'scalability_benchmark' in self.results:
            self._plot_scalability()

        # Generate memory usage plot
        if 'memory_benchmark' in self.results:
            self._plot_memory_usage()

        # Generate summary report
        self._generate_summary_report()

        print(f"✅ Reports saved to {self.results_dir}/")

    def _plot_accuracy_comparison(self):
        """Plot model accuracy comparison"""

        accuracy_data = self.results['accuracy_benchmark']

        models = []
        mae_scores = []
        mape_scores = []
        durations = []

        for category, category_data in accuracy_data.items():
            for model, model_data in category_data.items():
                models.append(f"{model} ({category})")
                mae_scores.append(model_data['avg_mae'])
                mape_scores.append(model_data['avg_mape'])
                durations.append(model_data['avg_duration'])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # MAE comparison
        ax1.bar(range(len(models)), mae_scores)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Model Accuracy Comparison (MAE)')

        # MAPE comparison
        ax2.bar(range(len(models)), mape_scores)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Mean Absolute Percentage Error (%)')
        ax2.set_title('Model Accuracy Comparison (MAPE)')

        # Duration comparison
        ax3.bar(range(len(models)), durations)
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.set_ylabel('Average Prediction Time (s)')
        ax3.set_title('Model Speed Comparison')

        # Performance overview
        categories = list(accuracy_data.keys())
        best_models = []
        for category in categories:
            category_data = accuracy_data[category]
            best_model = min(category_data.keys(),
                           key=lambda m: category_data[m]['avg_mae'])
            best_models.append(f"{best_model}\n({category_data[best_model]['avg_mae']:.2f})")

        ax4.bar(range(len(categories)), [accuracy_data[c][best_models[i].split('\n')[0]]['avg_mae']
                                        for i, c in enumerate(categories)])
        ax4.set_xticks(range(len(categories)))
        ax4.set_xticklabels(categories)
        ax4.set_ylabel('Best Model MAE')
        ax4.set_title('Best Performing Model per Category')

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_meta_learning_performance(self):
        """Plot meta-learning performance"""

        meta_data = self.results['meta_learning_benchmark']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Transfer learning times
        transfer_times = [r['transfer_time'] for r in meta_data['transfer_results']]
        categories = [r['target_category'] for r in meta_data['transfer_results']]

        ax1.bar(range(len(categories)), transfer_times)
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories)
        ax1.set_ylabel('Transfer Time (s)')
        ax1.set_title('Transfer Learning Performance')

        # Few-shot adaptation
        few_shot_data = meta_data['transfer_results'][0]['few_shot_results']  # First category
        shots = [r['shots'] for r in few_shot_data]
        adaptation_times = [r['adaptation_time'] for r in few_shot_data]

        ax2.plot(shots, adaptation_times, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Shots')
        ax2.set_ylabel('Adaptation Time (s)')
        ax2.set_title('Few-Shot Adaptation Performance')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/meta_learning_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_scalability(self):
        """Plot scalability results"""

        scalability_data = self.results['scalability_benchmark']

        concurrent_requests = list(scalability_data.keys())
        response_times = [scalability_data[n]['avg_response_time'] for n in concurrent_requests]
        throughput = [scalability_data[n]['requests_per_second'] for n in concurrent_requests]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Response time vs concurrent requests
        ax1.plot(concurrent_requests, response_times, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Concurrent Requests')
        ax1.set_ylabel('Average Response Time (s)')
        ax1.set_title('Scalability: Response Time')
        ax1.grid(True, alpha=0.3)

        # Throughput vs concurrent requests
        ax2.plot(concurrent_requests, throughput, 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Concurrent Requests')
        ax2.set_ylabel('Requests per Second')
        ax2.set_title('Scalability: Throughput')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/scalability_benchmark.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_memory_usage(self):
        """Plot memory usage comparison"""

        memory_data = self.results['memory_benchmark']

        models = list(memory_data.keys())
        memory_usage = [memory_data[m].get('memory_usage_mb', 0) for m in models]
        durations = [memory_data[m].get('duration', 0) for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Memory usage
        ax1.bar(range(len(models)), memory_usage)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models)
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Model Memory Usage Comparison')

        # Duration vs Memory
        ax2.scatter(memory_usage, durations, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax2.annotate(model, (memory_usage[i], durations[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Memory Usage (MB)')
        ax2.set_ylabel('Duration (s)')
        ax2.set_title('Memory vs Speed Trade-off')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_summary_report(self):
        """Generate comprehensive summary report"""

        report = f"""
# Drug Sales Prediction System - Benchmark Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents comprehensive benchmarking results for the Drug Sales Prediction System,
evaluating accuracy, performance, scalability, and resource usage across multiple models and scenarios.

## Key Findings

"""

        # Accuracy summary
        if 'accuracy_benchmark' in self.results:
            accuracy_data = self.results['accuracy_benchmark']
            best_overall_mae = float('inf')
            best_model = None

            for category, category_data in accuracy_data.items():
                for model, model_data in category_data.items():
                    if model_data['avg_mae'] < best_overall_mae:
                        best_overall_mae = model_data['avg_mae']
                        best_model = f"{model} ({category})"

            report += f"### Best Performing Model: {best_model} (MAE: {best_overall_mae:.2f})\n\n"

        # Performance summary
        if 'scalability_benchmark' in self.results:
            scalability_data = self.results['scalability_benchmark']
            max_throughput = max([v['requests_per_second'] for v in scalability_data.values()])
            report += f"### Maximum Throughput: {max_throughput:.1f} requests/second\n\n"

        # Meta-learning summary
        if 'meta_learning_benchmark' in self.results:
            meta_data = self.results['meta_learning_benchmark']
            avg_transfer_time = np.mean([r['transfer_time'] for r in meta_data['transfer_results']])
            report += f"### Meta-Learning Transfer Time: {avg_transfer_time:.2f} seconds\n\n"

        # Detailed results
        report += "## Detailed Results\n\n"

        if 'accuracy_benchmark' in self.results:
            report += "### Model Accuracy\n\n"
            report += "| Category | Model | MAE | MAPE (%) | Avg Time (s) |\n"
            report += "|----------|--------|-----|-----------|---------------|\n"

            for category, category_data in self.results['accuracy_benchmark'].items():
                for model, model_data in category_data.items():
                    report += f"| {category} | {model} | {model_data['avg_mae']:.2f} | {model_data['avg_mape']:.2f} | {model_data['avg_duration']:.3f} |\n"

            report += "\n"

        if 'scalability_benchmark' in self.results:
            report += "### Scalability Results\n\n"
            report += "| Concurrent Requests | Avg Response Time (s) | Requests/sec | Success Rate |\n"
            report += "|---------------------|------------------------|--------------|--------------|\n"

            for requests, data in self.results['scalability_benchmark'].items():
                report += f"| {requests} | {data['avg_response_time']:.3f} | {data['requests_per_second']:.1f} | {data['success_rate']:.1%} |\n"

            report += "\n"

        if 'meta_learning_benchmark' in self.results:
            report += "### Meta-Learning Results\n\n"
            report += "| Target Category | Transfer Time (s) | Few-Shot Performance |\n"
            report += "|-----------------|-------------------|----------------------|\n"

            for result in self.results['meta_learning_benchmark']['transfer_results']:
                few_shot_summary = ", ".join([f"{r['shots']}shot: {r['adaptation_time']:.2f}s"
                                            for r in result['few_shot_results'][:2]])
                report += f"| {result['target_category']} | {result['transfer_time']:.2f} | {few_shot_summary} |\n"

            report += "\n"

        # Recommendations
        report += "## Recommendations\n\n"

        if 'accuracy_benchmark' in self.results:
            # Find best model per category
            accuracy_data = self.results['accuracy_benchmark']
            recommendations = []

            for category, category_data in accuracy_data.items():
                best_model = min(category_data.keys(),
                               key=lambda m: category_data[m]['avg_mae'])
                best_mae = category_data[best_model]['avg_mae']
                recommendations.append(f"- **{category}**: Use {best_model} (MAE: {best_mae:.2f})")

            report += "### Model Selection\n" + "\n".join(recommendations) + "\n\n"

        if 'scalability_benchmark' in self.results:
            scalability_data = self.results['scalability_benchmark']
            optimal_load = max(scalability_data.keys(),
                             key=lambda k: scalability_data[k]['requests_per_second'])
            report += f"### Scalability\n- Optimal concurrent load: {optimal_load} requests\n"
            report += f"- Maximum throughput: {scalability_data[optimal_load]['requests_per_second']:.1f} req/sec\n\n"

        # Save report
        with open(f"{self.results_dir}/benchmark_report_{self.timestamp}.md", 'w') as f:
            f.write(report)

    def _get_actual_sales(self, category: str, date: str) -> float:
        """Get actual sales value for comparison (placeholder implementation)"""
        # This should be implemented based on your data structure
        # For now, return a mock value
        try:
            # Load actual data file
            df = pd.read_csv(f'{category}.csv')
            df['date'] = pd.to_datetime(df['date'])

            # Find closest date
            target_date = pd.to_datetime(date)
            closest_idx = (df['date'] - target_date).abs().idxmin()
            return df.loc[closest_idx, 'sales']
        except:
            return None

def main():
    """Main benchmarking function"""

    print("🚀 Drug Sales Prediction System - Performance Benchmarking")
    print("=" * 60)

    # Initialize benchmark suite
    benchmark = PerformanceBenchmark()

    try:
        # Run accuracy benchmark
        benchmark.benchmark_model_accuracy()

        # Run meta-learning benchmark
        benchmark.benchmark_meta_learning()

        # Run scalability benchmark
        benchmark.benchmark_scalability()

        # Run memory benchmark
        benchmark.benchmark_memory_usage()

        # Generate reports
        benchmark.generate_reports()

        print("✅ Benchmarking completed successfully!")
        print(f"📊 Results saved to: {benchmark.results_dir}/")

    except Exception as e:
        print(f"❌ Benchmarking failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()