"""
Evaluation metrics for fintech LLM assessment.

This module provides various metrics to evaluate the performance of fintech LLMs
including accuracy, relevance, completeness, and domain-specific metrics.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    test_case_id: str
    query: str
    expected_answer: str
    actual_answer: str
    accuracy_score: float
    relevance_score: float
    completeness_score: float
    response_time: float
    category: str
    domain: str
    difficulty: str
    metadata: Optional[Dict[str, Any]] = None


class EvaluationMetrics:
    """Collection of evaluation metrics for fintech LLMs."""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
    
    def calculate_accuracy(self, expected: str, actual: str) -> float:
        """Calculate accuracy using exact match and fuzzy matching."""
        # Normalize text
        expected_norm = self._normalize_text(expected)
        actual_norm = self._normalize_text(actual)
        
        # Exact match
        if expected_norm == actual_norm:
            return 1.0
        
        # Fuzzy matching using difflib
        similarity = difflib.SequenceMatcher(None, expected_norm, actual_norm).ratio()
        
        # Keyword matching
        expected_words = set(expected_norm.lower().split())
        actual_words = set(actual_norm.lower().split())
        
        if expected_words:
            keyword_overlap = len(expected_words.intersection(actual_words)) / len(expected_words)
        else:
            keyword_overlap = 0.0
        
        # Combine metrics
        accuracy = (similarity * 0.6) + (keyword_overlap * 0.4)
        return min(accuracy, 1.0)
    
    def calculate_relevance(self, query: str, answer: str) -> float:
        """Calculate relevance score based on query-answer alignment."""
        # Simple keyword relevance
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = query_words - stop_words
        answer_words = answer_words - stop_words
        
        if query_words:
            relevance = len(query_words.intersection(answer_words)) / len(query_words)
        else:
            relevance = 0.0
        
        return min(relevance, 1.0)
    
    def calculate_completeness(self, expected: str, actual: str) -> float:
        """Calculate completeness score based on information coverage."""
        expected_sentences = self._split_into_sentences(expected)
        actual_sentences = self._split_into_sentences(actual)
        
        if not expected_sentences:
            return 1.0 if actual_sentences else 0.0
        
        # Calculate how many expected information points are covered
        covered_points = 0
        total_points = len(expected_sentences)
        
        for expected_sent in expected_sentences:
            for actual_sent in actual_sentences:
                if self._sentences_similar(expected_sent, actual_sent):
                    covered_points += 1
                    break
        
        return covered_points / total_points if total_points > 0 else 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace and punctuation
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _sentences_similar(self, sent1: str, sent2: str, threshold: float = 0.7) -> bool:
        """Check if two sentences are similar."""
        similarity = difflib.SequenceMatcher(None, sent1.lower(), sent2.lower()).ratio()
        return similarity >= threshold
    
    def add_result(self, result: EvaluationResult):
        """Add an evaluation result."""
        self.results.append(result)
    
    def get_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall metrics across all results."""
        if not self.results:
            return {}
        
        metrics = {
            'total_tests': len(self.results),
            'avg_accuracy': np.mean([r.accuracy_score for r in self.results]),
            'avg_relevance': np.mean([r.relevance_score for r in self.results]),
            'avg_completeness': np.mean([r.completeness_score for r in self.results]),
            'avg_response_time': np.mean([r.response_time for r in self.results]),
            'min_accuracy': np.min([r.accuracy_score for r in self.results]),
            'max_accuracy': np.max([r.accuracy_score for r in self.results]),
            'std_accuracy': np.std([r.accuracy_score for r in self.results])
        }
        
        return metrics
    
    def get_metrics_by_category(self) -> Dict[str, Dict[str, float]]:
        """Get metrics grouped by category."""
        categories = {}
        
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        category_metrics = {}
        for category, results in categories.items():
            category_metrics[category] = {
                'count': len(results),
                'avg_accuracy': np.mean([r.accuracy_score for r in results]),
                'avg_relevance': np.mean([r.relevance_score for r in results]),
                'avg_completeness': np.mean([r.completeness_score for r in results]),
                'avg_response_time': np.mean([r.response_time for r in results])
            }
        
        return category_metrics
    
    def get_metrics_by_domain(self) -> Dict[str, Dict[str, float]]:
        """Get metrics grouped by domain."""
        domains = {}
        
        for result in self.results:
            if result.domain not in domains:
                domains[result.domain] = []
            domains[result.domain].append(result)
        
        domain_metrics = {}
        for domain, results in domains.items():
            domain_metrics[domain] = {
                'count': len(results),
                'avg_accuracy': np.mean([r.accuracy_score for r in results]),
                'avg_relevance': np.mean([r.relevance_score for r in results]),
                'avg_completeness': np.mean([r.completeness_score for r in results]),
                'avg_response_time': np.mean([r.response_time for r in results])
            }
        
        return domain_metrics
    
    def get_metrics_by_difficulty(self) -> Dict[str, Dict[str, float]]:
        """Get metrics grouped by difficulty level."""
        difficulties = {}
        
        for result in self.results:
            if result.difficulty not in difficulties:
                difficulties[result.difficulty] = []
            difficulties[result.difficulty].append(result)
        
        difficulty_metrics = {}
        for difficulty, results in difficulties.items():
            difficulty_metrics[difficulty] = {
                'count': len(results),
                'avg_accuracy': np.mean([r.accuracy_score for r in results]),
                'avg_relevance': np.mean([r.relevance_score for r in results]),
                'avg_completeness': np.mean([r.completeness_score for r in results]),
                'avg_response_time': np.mean([r.response_time for r in results])
            }
        
        return difficulty_metrics
    
    def get_failed_cases(self, threshold: float = 0.7) -> List[EvaluationResult]:
        """Get cases that failed to meet the accuracy threshold."""
        return [r for r in self.results if r.accuracy_score < threshold]
    
    def get_best_cases(self, threshold: float = 0.9) -> List[EvaluationResult]:
        """Get cases that performed exceptionally well."""
        return [r for r in self.results if r.accuracy_score >= threshold]
    
    def export_results(self, filepath: str):
        """Export evaluation results to JSON."""
        import json
        
        data = []
        for result in self.results:
            data.append({
                'test_case_id': result.test_case_id,
                'query': result.query,
                'expected_answer': result.expected_answer,
                'actual_answer': result.actual_answer,
                'accuracy_score': result.accuracy_score,
                'relevance_score': result.relevance_score,
                'completeness_score': result.completeness_score,
                'response_time': result.response_time,
                'category': result.category,
                'domain': result.domain,
                'difficulty': result.difficulty,
                'metadata': result.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.results:
            return "No evaluation results available."
        
        overall = self.get_overall_metrics()
        by_category = self.get_metrics_by_category()
        by_domain = self.get_metrics_by_domain()
        by_difficulty = self.get_metrics_by_difficulty()
        
        report = []
        report.append("=" * 80)
        report.append("FINANCIAL LLM EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Total Tests: {overall['total_tests']}")
        report.append(f"Average Accuracy: {overall['avg_accuracy']:.3f}")
        report.append(f"Average Relevance: {overall['avg_relevance']:.3f}")
        report.append(f"Average Completeness: {overall['avg_completeness']:.3f}")
        report.append(f"Average Response Time: {overall['avg_response_time']:.3f}s")
        report.append(f"Accuracy Range: {overall['min_accuracy']:.3f} - {overall['max_accuracy']:.3f}")
        report.append(f"Accuracy Std Dev: {overall['std_accuracy']:.3f}")
        report.append("")
        
        # By category
        report.append("PERFORMANCE BY CATEGORY")
        report.append("-" * 40)
        for category, metrics in by_category.items():
            report.append(f"{category}:")
            report.append(f"  Count: {metrics['count']}")
            report.append(f"  Accuracy: {metrics['avg_accuracy']:.3f}")
            report.append(f"  Relevance: {metrics['avg_relevance']:.3f}")
            report.append(f"  Completeness: {metrics['avg_completeness']:.3f}")
            report.append("")
        
        # By domain
        report.append("PERFORMANCE BY DOMAIN")
        report.append("-" * 40)
        for domain, metrics in by_domain.items():
            report.append(f"{domain}:")
            report.append(f"  Count: {metrics['count']}")
            report.append(f"  Accuracy: {metrics['avg_accuracy']:.3f}")
            report.append(f"  Relevance: {metrics['avg_relevance']:.3f}")
            report.append(f"  Completeness: {metrics['avg_completeness']:.3f}")
            report.append("")
        
        # By difficulty
        report.append("PERFORMANCE BY DIFFICULTY")
        report.append("-" * 40)
        for difficulty, metrics in by_difficulty.items():
            report.append(f"{difficulty}:")
            report.append(f"  Count: {metrics['count']}")
            report.append(f"  Accuracy: {metrics['avg_accuracy']:.3f}")
            report.append(f"  Relevance: {metrics['avg_relevance']:.3f}")
            report.append(f"  Completeness: {metrics['avg_completeness']:.3f}")
            report.append("")
        
        # Failed cases
        failed_cases = self.get_failed_cases()
        if failed_cases:
            report.append("FAILED CASES (Accuracy < 0.7)")
            report.append("-" * 40)
            for case in failed_cases[:5]:  # Show top 5
                report.append(f"ID: {case.test_case_id}")
                report.append(f"Query: {case.query}")
                report.append(f"Accuracy: {case.accuracy_score:.3f}")
                report.append("")
        
        return "\n".join(report) 