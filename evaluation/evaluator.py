"""
Main LLM evaluator for fintech applications.

This module provides the core evaluation functionality to test fintech LLMs
against curated datasets and generate comprehensive performance reports.
"""

import time
from typing import List, Dict, Any, Optional, Callable
from .datasets import FintechTestDataset, TestCase
from .metrics import EvaluationMetrics, EvaluationResult


class LLMEvaluator:
    """Main evaluator for fintech LLMs."""
    
    def __init__(self, llm_function: Callable[[str], str]):
        """
        Initialize the evaluator.
        
        Args:
            llm_function: Function that takes a query string and returns an answer string
        """
        self.llm_function = llm_function
        self.dataset = FintechTestDataset()
        self.metrics = EvaluationMetrics()
    
    def evaluate_single_case(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case."""
        start_time = time.time()
        
        try:
            # Get LLM response
            actual_answer = self.llm_function(test_case.query)
            response_time = time.time() - start_time
            
            # Calculate metrics
            accuracy_score = self.metrics.calculate_accuracy(
                test_case.expected_answer, actual_answer
            )
            relevance_score = self.metrics.calculate_relevance(
                test_case.query, actual_answer
            )
            completeness_score = self.metrics.calculate_completeness(
                test_case.expected_answer, actual_answer
            )
            
            # Create result
            result = EvaluationResult(
                test_case_id=test_case.id,
                query=test_case.query,
                expected_answer=test_case.expected_answer,
                actual_answer=actual_answer,
                accuracy_score=accuracy_score,
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                response_time=response_time,
                category=test_case.category,
                domain=test_case.domain,
                difficulty=test_case.difficulty,
                metadata=test_case.metadata
            )
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            response_time = time.time() - start_time
            result = EvaluationResult(
                test_case_id=test_case.id,
                query=test_case.query,
                expected_answer=test_case.expected_answer,
                actual_answer=f"ERROR: {str(e)}",
                accuracy_score=0.0,
                relevance_score=0.0,
                completeness_score=0.0,
                response_time=response_time,
                category=test_case.category,
                domain=test_case.domain,
                difficulty=test_case.difficulty,
                metadata={"error": str(e)}
            )
            return result
    
    def evaluate_all(self, test_cases: Optional[List[TestCase]] = None) -> Dict[str, Any]:
        """Evaluate all test cases or a specified subset."""
        if test_cases is None:
            test_cases = self.dataset.get_all_cases()
        
        print(f"Starting evaluation of {len(test_cases)} test cases...")
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"Evaluating case {i}/{len(test_cases)}: {test_case.id}")
            result = self.evaluate_single_case(test_case)
            results.append(result)
            self.metrics.add_result(result)
        
        # Generate summary
        summary = {
            'total_cases': len(test_cases),
            'completed_cases': len(results),
            'overall_metrics': self.metrics.get_overall_metrics(),
            'category_metrics': self.metrics.get_metrics_by_category(),
            'domain_metrics': self.metrics.get_metrics_by_domain(),
            'difficulty_metrics': self.metrics.get_metrics_by_difficulty()
        }
        
        return summary
    
    def evaluate_by_category(self, category: str) -> Dict[str, Any]:
        """Evaluate test cases by category."""
        test_cases = self.dataset.get_cases_by_category(category)
        return self.evaluate_all(test_cases)
    
    def evaluate_by_domain(self, domain: str) -> Dict[str, Any]:
        """Evaluate test cases by domain."""
        test_cases = self.dataset.get_cases_by_domain(domain)
        return self.evaluate_all(test_cases)
    
    def evaluate_random_sample(self, sample_size: int = 10) -> Dict[str, Any]:
        """Evaluate a random sample of test cases."""
        test_cases = self.dataset.get_random_sample(sample_size)
        return self.evaluate_all(test_cases)
    
    def get_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        return self.metrics.generate_report()
    
    def export_results(self, filepath: str):
        """Export evaluation results to JSON file."""
        self.metrics.export_results(filepath)
        print(f"Results exported to {filepath}")
    
    def print_summary(self):
        """Print a summary of evaluation results."""
        if not self.metrics.results:
            print("No evaluation results available.")
            return
        
        overall = self.metrics.get_overall_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {overall['total_tests']}")
        print(f"Average Accuracy: {overall['avg_accuracy']:.3f}")
        print(f"Average Relevance: {overall['avg_relevance']:.3f}")
        print(f"Average Completeness: {overall['avg_completeness']:.3f}")
        print(f"Average Response Time: {overall['avg_response_time']:.3f}s")
        print(f"Accuracy Range: {overall['min_accuracy']:.3f} - {overall['max_accuracy']:.3f}")
        
        # Show top performing categories
        category_metrics = self.metrics.get_metrics_by_category()
        if category_metrics:
            print("\nTop Performing Categories:")
            sorted_categories = sorted(
                category_metrics.items(),
                key=lambda x: x[1]['avg_accuracy'],
                reverse=True
            )
            for category, metrics in sorted_categories[:3]:
                print(f"  {category}: {metrics['avg_accuracy']:.3f}")
        
        # Show failed cases
        failed_cases = self.metrics.get_failed_cases()
        if failed_cases:
            print(f"\nFailed Cases: {len(failed_cases)}")
        
        print("="*60)
    
    def benchmark_performance(self, iterations: int = 3) -> Dict[str, Any]:
        """Run performance benchmarking with multiple iterations."""
        print(f"Running performance benchmark with {iterations} iterations...")
        
        all_metrics = []
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            # Clear previous results
            self.metrics.results = []
            
            # Run evaluation
            summary = self.evaluate_all()
            all_metrics.append(summary['overall_metrics'])
        
        # Calculate average metrics across iterations
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key == 'total_tests':
                avg_metrics[key] = all_metrics[0][key]  # Should be same for all
            else:
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        # Calculate standard deviation
        std_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'total_tests':
                values = [m[key] for m in all_metrics]
                std_metrics[f"{key}_std"] = (sum((v - avg_metrics[key])**2 for v in values) / len(values))**0.5
        
        benchmark_results = {
            'iterations': iterations,
            'average_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'all_iterations': all_metrics
        }
        
        return benchmark_results
    
    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline results."""
        current_metrics = self.metrics.get_overall_metrics()
        
        comparison = {
            'current': current_metrics,
            'baseline': baseline_results,
            'improvement': {}
        }
        
        # Calculate improvements
        for key in current_metrics.keys():
            if key in baseline_results and key != 'total_tests':
                current_val = current_metrics[key]
                baseline_val = baseline_results[key]
                
                if baseline_val != 0:
                    improvement = ((current_val - baseline_val) / baseline_val) * 100
                    comparison['improvement'][key] = improvement
                else:
                    comparison['improvement'][key] = float('inf') if current_val > 0 else 0
        
        return comparison 