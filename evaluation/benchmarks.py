"""
Benchmark runners for fintech LLM evaluation.

This module provides different benchmark scenarios and automated evaluation
pipelines for comprehensive LLM assessment.
"""

import time
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from .evaluator import LLMEvaluator
from .datasets import FintechTestDataset


class FintechBenchmark:
    """Comprehensive benchmark suite for fintech LLMs."""
    
    def __init__(self, llm_function, output_dir: str = "evaluation_results"):
        """
        Initialize the benchmark suite.
        
        Args:
            llm_function: Function that takes a query string and returns an answer string
            output_dir: Directory to save evaluation results
        """
        self.evaluator = LLMEvaluator(llm_function)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dataset = FintechTestDataset()
    
    def run_quick_benchmark(self) -> Dict[str, Any]:
        """Run a quick benchmark with a small sample."""
        print("🚀 Running Quick Benchmark...")
        print("="*50)
        
        # Use a small random sample
        test_cases = self.dataset.get_random_sample(5)
        
        start_time = time.time()
        summary = self.evaluator.evaluate_all(test_cases)
        total_time = time.time() - start_time
        
        # Print results
        self.evaluator.print_summary()
        
        # Save results
        timestamp = int(time.time())
        results_file = self.output_dir / f"quick_benchmark_{timestamp}.json"
        self.evaluator.export_results(str(results_file))
        
        # Generate report
        report_file = self.output_dir / f"quick_benchmark_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self.evaluator.get_evaluation_report())
        
        print(f"\n✅ Quick benchmark completed in {total_time:.2f}s")
        print(f"📊 Results saved to: {results_file}")
        print(f"📄 Report saved to: {report_file}")
        
        return summary
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run a comprehensive benchmark with all test cases."""
        print("🔬 Running Comprehensive Benchmark...")
        print("="*50)
        
        start_time = time.time()
        summary = self.evaluator.evaluate_all()
        total_time = time.time() - start_time
        
        # Print results
        self.evaluator.print_summary()
        
        # Save results
        timestamp = int(time.time())
        results_file = self.output_dir / f"comprehensive_benchmark_{timestamp}.json"
        self.evaluator.export_results(str(results_file))
        
        # Generate detailed report
        report_file = self.output_dir / f"comprehensive_benchmark_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self.evaluator.get_evaluation_report())
        
        print(f"\n✅ Comprehensive benchmark completed in {total_time:.2f}s")
        print(f"📊 Results saved to: {results_file}")
        print(f"📄 Report saved to: {report_file}")
        
        return summary
    
    def run_category_benchmark(self, category: str) -> Dict[str, Any]:
        """Run benchmark for a specific category."""
        print(f"📋 Running Category Benchmark: {category}")
        print("="*50)
        
        start_time = time.time()
        summary = self.evaluator.evaluate_by_category(category)
        total_time = time.time() - start_time
        
        # Print results
        self.evaluator.print_summary()
        
        # Save results
        timestamp = int(time.time())
        results_file = self.output_dir / f"category_{category.lower()}_{timestamp}.json"
        self.evaluator.export_results(str(results_file))
        
        print(f"\n✅ Category benchmark completed in {total_time:.2f}s")
        print(f"📊 Results saved to: {results_file}")
        
        return summary
    
    def run_domain_benchmark(self, domain: str) -> Dict[str, Any]:
        """Run benchmark for a specific domain."""
        print(f"🌐 Running Domain Benchmark: {domain}")
        print("="*50)
        
        start_time = time.time()
        summary = self.evaluator.evaluate_by_domain(domain)
        total_time = time.time() - start_time
        
        # Print results
        self.evaluator.print_summary()
        
        # Save results
        timestamp = int(time.time())
        results_file = self.output_dir / f"domain_{domain.lower()}_{timestamp}.json"
        self.evaluator.export_results(str(results_file))
        
        print(f"\n✅ Domain benchmark completed in {total_time:.2f}s")
        print(f"📊 Results saved to: {results_file}")
        
        return summary
    
    def run_performance_benchmark(self, iterations: int = 3) -> Dict[str, Any]:
        """Run performance benchmark with multiple iterations."""
        print(f"⚡ Running Performance Benchmark ({iterations} iterations)...")
        print("="*50)
        
        start_time = time.time()
        benchmark_results = self.evaluator.benchmark_performance(iterations)
        total_time = time.time() - start_time
        
        # Print results
        avg_metrics = benchmark_results['average_metrics']
        std_metrics = benchmark_results['std_metrics']
        
        print(f"\n📊 Performance Benchmark Results:")
        print(f"   Iterations: {iterations}")
        print(f"   Average Accuracy: {avg_metrics['avg_accuracy']:.3f} ± {std_metrics.get('avg_accuracy_std', 0):.3f}")
        print(f"   Average Response Time: {avg_metrics['avg_response_time']:.3f}s ± {std_metrics.get('avg_response_time_std', 0):.3f}s")
        
        # Save results
        timestamp = int(time.time())
        results_file = self.output_dir / f"performance_benchmark_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\n✅ Performance benchmark completed in {total_time:.2f}s")
        print(f"📊 Results saved to: {results_file}")
        
        return benchmark_results
    
    def run_comparison_benchmark(self, baseline_file: str) -> Dict[str, Any]:
        """Run benchmark and compare with baseline results."""
        print("🔄 Running Comparison Benchmark...")
        print("="*50)
        
        # Load baseline results
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        # Run current evaluation
        summary = self.evaluator.evaluate_all()
        
        # Compare with baseline
        comparison = self.evaluator.compare_with_baseline(baseline_results)
        
        # Print comparison
        print(f"\n📊 Comparison Results:")
        current = comparison['current']
        baseline = comparison['baseline']
        improvement = comparison['improvement']
        
        print(f"   Current Accuracy: {current['avg_accuracy']:.3f}")
        print(f"   Baseline Accuracy: {baseline['avg_accuracy']:.3f}")
        print(f"   Improvement: {improvement.get('avg_accuracy', 0):.1f}%")
        
        # Save comparison
        timestamp = int(time.time())
        comparison_file = self.output_dir / f"comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"📊 Comparison saved to: {comparison_file}")
        
        return comparison
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all available benchmarks."""
        print("🎯 Running All Benchmarks...")
        print("="*50)
        
        all_results = {}
        
        # Quick benchmark
        print("\n1. Quick Benchmark")
        all_results['quick'] = self.run_quick_benchmark()
        
        # Category benchmarks
        print("\n2. Category Benchmarks")
        categories = ['Payment Processing', 'Compliance', 'Customer Service']
        for category in categories:
            print(f"\n   Running {category} benchmark...")
            all_results[f'category_{category.lower().replace(" ", "_")}'] = self.run_category_benchmark(category)
        
        # Domain benchmarks
        print("\n3. Domain Benchmarks")
        domains = ['payment', 'compliance', 'customer_service']
        for domain in domains:
            print(f"\n   Running {domain} benchmark...")
            all_results[f'domain_{domain}'] = self.run_domain_benchmark(domain)
        
        # Performance benchmark
        print("\n4. Performance Benchmark")
        all_results['performance'] = self.run_performance_benchmark()
        
        # Save all results
        timestamp = int(time.time())
        all_results_file = self.output_dir / f"all_benchmarks_{timestamp}.json"
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ All benchmarks completed!")
        print(f"📊 All results saved to: {all_results_file}")
        
        return all_results
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all benchmark results."""
        report = []
        report.append("="*80)
        report.append("FINANCIAL LLM BENCHMARK SUMMARY REPORT")
        report.append("="*80)
        report.append("")
        
        # Get all result files
        result_files = list(self.output_dir.glob("*.json"))
        
        if not result_files:
            report.append("No benchmark results found.")
            return "\n".join(report)
        
        report.append(f"Found {len(result_files)} benchmark result files:")
        report.append("")
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                if 'overall_metrics' in data:
                    metrics = data['overall_metrics']
                    report.append(f"📊 {result_file.stem}:")
                    report.append(f"   Accuracy: {metrics.get('avg_accuracy', 0):.3f}")
                    report.append(f"   Tests: {metrics.get('total_tests', 0)}")
                    report.append("")
                
            except Exception as e:
                report.append(f"❌ Error reading {result_file}: {e}")
                report.append("")
        
        return "\n".join(report) 