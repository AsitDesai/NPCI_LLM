#!/usr/bin/env python3
"""
RAG Performance Test Script

This script tests the RAG system with 30 queries against ground truth
and evaluates performance across multiple parameters.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import re

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from RAG_test_CLI import RAGCLI


@dataclass
class TestQuery:
    """Test query with ground truth."""
    query: str
    ground_truth: str
    category: str
    expected_keywords: List[str]
    expected_accuracy: float  # Expected accuracy score (0.0 to 1.0)


@dataclass
class TestResult:
    """Result of a single test."""
    query: str
    ground_truth: str
    prediction: str
    confidence: float
    response_time: float
    accuracy_score: float
    keyword_match_score: float
    relevance_score: float
    category: str


class RAGPerformanceTester:
    """Test RAG system performance against ground truth."""
    
    def __init__(self):
        """Initialize the tester."""
        self.rag_cli = RAGCLI()
        self.test_queries = self._create_test_queries()
        
    def _create_test_queries(self) -> List[TestQuery]:
        """Create test queries with ground truth."""
        return [
            # Payment & Billing Queries
            TestQuery(
                query="How can I check my payment status?",
                ground_truth="You can check your payment status by logging into your account dashboard and navigating to the 'Billing' section. Alternatively, you can contact our customer support team with your account number for immediate assistance.",
                category="payment",
                expected_keywords=["dashboard", "billing", "support"],
                expected_accuracy=0.9
            ),
            TestQuery(
                query="What payment methods do you accept?",
                ground_truth="We accept all major credit cards (Visa, MasterCard, American Express), debit cards, bank transfers, and digital wallets including PayPal and Apple Pay.",
                category="payment",
                expected_keywords=["credit cards", "debit cards", "PayPal", "Apple Pay"],
                expected_accuracy=0.95
            ),
            TestQuery(
                query="Can I get a refund?",
                ground_truth="Yes, we offer a 30-day money-back guarantee. If you're not completely satisfied with our service, you can request a full refund within 30 days of your purchase.",
                category="payment",
                expected_keywords=["30-day", "money-back", "refund"],
                expected_accuracy=0.9
            ),
            TestQuery(
                query="How do I update my billing information?",
                ground_truth="You can update your billing information by going to Account Settings > Billing Information. Make sure to update both your payment method and billing address.",
                category="payment",
                expected_keywords=["Account Settings", "Billing Information"],
                expected_accuracy=0.85
            ),
            
            # Account Management Queries
            TestQuery(
                query="How do I reset my password?",
                ground_truth="Click on 'Forgot Password' on the login page, enter your email address, and follow the instructions sent to your email. The reset link will expire in 24 hours.",
                category="account",
                expected_keywords=["Forgot Password", "email", "24 hours"],
                expected_accuracy=0.9
            ),
            TestQuery(
                query="How do I cancel my subscription?",
                ground_truth="You can cancel your subscription at any time by going to Account Settings > Subscription > Cancel. Your access will continue until the end of your current billing period.",
                category="account",
                expected_keywords=["Account Settings", "Subscription", "Cancel"],
                expected_accuracy=0.9
            ),
            TestQuery(
                query="Is my personal information secure?",
                ground_truth="Yes, we use industry-standard encryption and security measures to protect your personal and payment information. We never store your credit card details on our servers.",
                category="account",
                expected_keywords=["encryption", "security", "credit card"],
                expected_accuracy=0.85
            ),
            TestQuery(
                query="Can I have multiple users on one account?",
                ground_truth="Yes, depending on your subscription plan. Basic plans allow 1 user, Professional plans allow up to 5 users, and Enterprise plans support unlimited users.",
                category="account",
                expected_keywords=["Basic", "Professional", "Enterprise", "users"],
                expected_accuracy=0.9
            ),
            
            # Technical Support Queries
            TestQuery(
                query="How do I download my data?",
                ground_truth="You can export your data by going to Account Settings > Data Export. Choose the format you prefer (CSV, JSON, or PDF) and click 'Export Data.'",
                category="technical",
                expected_keywords=["Account Settings", "Data Export", "CSV", "JSON", "PDF"],
                expected_accuracy=0.95
            ),
            TestQuery(
                query="What browsers do you support?",
                ground_truth="We support all modern browsers including Chrome, Firefox, Safari, and Edge. For the best experience, we recommend using the latest version of Chrome.",
                category="technical",
                expected_keywords=["Chrome", "Firefox", "Safari", "Edge"],
                expected_accuracy=0.95
            ),
            TestQuery(
                query="The website is loading slowly, what should I do?",
                ground_truth="Try clearing your browser cache and cookies, or try accessing the site from a different browser. If the problem persists, contact our technical support team.",
                category="technical",
                expected_keywords=["cache", "cookies", "browser", "support"],
                expected_accuracy=0.85
            ),
            TestQuery(
                query="I can't log into my account, what's wrong?",
                ground_truth="Check that you're using the correct email and password. If you've forgotten your password, use the 'Forgot Password' feature. If you're still having issues, contact support.",
                category="technical",
                expected_keywords=["email", "password", "Forgot Password", "support"],
                expected_accuracy=0.9
            ),
            
            # Service & Features Queries
            TestQuery(
                query="What's included in the free trial?",
                ground_truth="Our free trial includes all premium features for 14 days. No credit card is required to start your trial, and you can cancel at any time.",
                category="service",
                expected_keywords=["premium features", "14 days", "no credit card"],
                expected_accuracy=0.9
            ),
            TestQuery(
                query="How often do you update your features?",
                ground_truth="We release new features and improvements monthly. You can stay updated by following our blog or checking the 'What's New' section in your dashboard.",
                category="service",
                expected_keywords=["monthly", "blog", "What's New", "dashboard"],
                expected_accuracy=0.85
            ),
            TestQuery(
                query="Do you offer customer support on weekends?",
                ground_truth="Our customer support team is available Monday through Friday, 9 AM to 6 PM EST. For urgent issues outside these hours, you can submit a ticket and we'll respond within 24 hours.",
                category="service",
                expected_keywords=["Monday through Friday", "9 AM to 6 PM", "24 hours"],
                expected_accuracy=0.9
            ),
            TestQuery(
                query="Can I integrate this with other software?",
                ground_truth="Yes, we offer API access and integrations with popular tools like Slack, Zapier, and Microsoft Office. Check our integrations page for the full list.",
                category="service",
                expected_keywords=["API", "Slack", "Zapier", "Microsoft Office"],
                expected_accuracy=0.9
            ),
            
            # Payment Ethics Queries
            TestQuery(
                query="What should I do to verify customer identity?",
                ground_truth="DO always verify customer identity before processing any payment transactions to prevent fraud and ensure security.",
                category="ethics",
                expected_keywords=["verify", "identity", "fraud", "security"],
                expected_accuracy=0.85
            ),
            TestQuery(
                query="How should I handle payment records?",
                ground_truth="DO maintain detailed records of all payment transactions for audit purposes and regulatory compliance.",
                category="ethics",
                expected_keywords=["detailed records", "audit", "regulatory compliance"],
                expected_accuracy=0.85
            ),
            TestQuery(
                query="What payment gateways should I use?",
                ground_truth="DO use secure, encrypted payment gateways and follow PCI DSS standards for handling sensitive payment data.",
                category="ethics",
                expected_keywords=["secure", "encrypted", "PCI DSS", "sensitive"],
                expected_accuracy=0.85
            ),
            TestQuery(
                query="Should I store credit card numbers?",
                ground_truth="DON'T store credit card numbers or sensitive payment information in plain text or unsecured databases.",
                category="ethics",
                expected_keywords=["DON'T", "credit card", "plain text", "unsecured"],
                expected_accuracy=0.9
            ),
            
            # Variant Queries (different ways to ask same thing)
            TestQuery(
                query="browser support",
                ground_truth="We support all modern browsers including Chrome, Firefox, Safari, and Edge. For the best experience, we recommend using the latest version of Chrome.",
                category="technical",
                expected_keywords=["Chrome", "Firefox", "Safari", "Edge"],
                expected_accuracy=0.8
            ),
            TestQuery(
                query="download data export",
                ground_truth="You can export your data by going to Account Settings > Data Export. Choose the format you prefer (CSV, JSON, or PDF) and click 'Export Data.'",
                category="technical",
                expected_keywords=["Account Settings", "Data Export", "CSV", "JSON", "PDF"],
                expected_accuracy=0.8
            ),
            TestQuery(
                query="cancel subscription",
                ground_truth="You can cancel your subscription at any time by going to Account Settings > Subscription > Cancel. Your access will continue until the end of your current billing period.",
                category="account",
                expected_keywords=["Account Settings", "Subscription", "Cancel"],
                expected_accuracy=0.8
            ),
            TestQuery(
                query="payment methods accepted",
                ground_truth="We accept all major credit cards (Visa, MasterCard, American Express), debit cards, bank transfers, and digital wallets including PayPal and Apple Pay.",
                category="payment",
                expected_keywords=["credit cards", "debit cards", "PayPal", "Apple Pay"],
                expected_accuracy=0.8
            ),
            TestQuery(
                query="free trial features",
                ground_truth="Our free trial includes all premium features for 14 days. No credit card is required to start your trial, and you can cancel at any time.",
                category="service",
                expected_keywords=["premium features", "14 days", "no credit card"],
                expected_accuracy=0.8
            ),
            
            # Edge Cases
            TestQuery(
                query="What is the meaning of life?",
                ground_truth="I don't have enough information to answer this question.",
                category="edge_case",
                expected_keywords=[],
                expected_accuracy=0.0
            ),
            TestQuery(
                query="How to make a sandwich?",
                ground_truth="I don't have enough information to answer this question.",
                category="edge_case",
                expected_keywords=[],
                expected_accuracy=0.0
            ),
            TestQuery(
                query="What is the weather like?",
                ground_truth="I don't have enough information to answer this question.",
                category="edge_case",
                expected_keywords=[],
                expected_accuracy=0.0
            ),
            TestQuery(
                query="Tell me a joke",
                ground_truth="I don't have enough information to answer this question.",
                category="edge_case",
                expected_keywords=[],
                expected_accuracy=0.0
            ),
            TestQuery(
                query="What time is it?",
                ground_truth="I don't have enough information to answer this question.",
                category="edge_case",
                expected_keywords=[],
                expected_accuracy=0.0
            ),
            
            # Complex Queries
            TestQuery(
                query="How do I check my payment status and update billing information?",
                ground_truth="You can check your payment status by logging into your account dashboard and navigating to the 'Billing' section. You can update your billing information by going to Account Settings > Billing Information.",
                category="complex",
                expected_keywords=["dashboard", "billing", "Account Settings"],
                expected_accuracy=0.7
            ),
            TestQuery(
                query="What browsers are supported and how do I download my data?",
                ground_truth="We support all modern browsers including Chrome, Firefox, Safari, and Edge. You can export your data by going to Account Settings > Data Export.",
                category="complex",
                expected_keywords=["Chrome", "Firefox", "Safari", "Edge", "Account Settings", "Data Export"],
                expected_accuracy=0.7
            ),
            TestQuery(
                query="How do I reset my password and cancel my subscription?",
                ground_truth="Click on 'Forgot Password' on the login page to reset your password. You can cancel your subscription by going to Account Settings > Subscription > Cancel.",
                category="complex",
                expected_keywords=["Forgot Password", "Account Settings", "Subscription", "Cancel"],
                expected_accuracy=0.7
            )
        ]
    
    def _calculate_accuracy_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate accuracy score between prediction and ground truth."""
        if not prediction or not ground_truth:
            return 0.0
        
        # Simple keyword-based accuracy
        prediction_lower = prediction.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Split into words and calculate overlap
        pred_words = set(re.findall(r'\b\w+\b', prediction_lower))
        truth_words = set(re.findall(r'\b\w+\b', ground_truth_lower))
        
        if not truth_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(pred_words.intersection(truth_words))
        union = len(pred_words.union(truth_words))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_keyword_match_score(self, prediction: str, expected_keywords: List[str]) -> float:
        """Calculate keyword match score."""
        if not expected_keywords:
            return 1.0
        
        prediction_lower = prediction.lower()
        matches = 0
        
        for keyword in expected_keywords:
            if keyword.lower() in prediction_lower:
                matches += 1
        
        return matches / len(expected_keywords)
    
    def _calculate_relevance_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate relevance score based on semantic similarity."""
        # Simple relevance based on length and content overlap
        if not prediction or not ground_truth:
            return 0.0
        
        # Check if prediction contains key phrases from ground truth
        key_phrases = [
            "account settings", "billing", "support", "password", 
            "subscription", "data export", "browsers", "chrome",
            "firefox", "safari", "edge", "payment", "refund"
        ]
        
        prediction_lower = prediction.lower()
        ground_truth_lower = ground_truth.lower()
        
        relevant_phrases = 0
        for phrase in key_phrases:
            if phrase in ground_truth_lower and phrase in prediction_lower:
                relevant_phrases += 1
        
        return min(relevant_phrases / len(key_phrases), 1.0)
    
    def run_single_test(self, test_query: TestQuery) -> TestResult:
        """Run a single test query."""
        try:
            start_time = time.time()
            
            # Run the query through RAG system
            result = self.rag_cli.run_query(test_query.query, style="concise")
            
            response_time = time.time() - start_time
            
            # Extract prediction and confidence from the result
            if result and isinstance(result, dict):
                prediction = (result.get('processed_response') or 
                            result.get('response') or 
                            result.get('original_response') or 
                            'No response')
                confidence = result.get('confidence', 0.5)
            else:
                prediction = str(result) if result else 'No response'
                confidence = 0.5
            
            # Calculate scores
            accuracy_score = self._calculate_accuracy_score(prediction, test_query.ground_truth)
            keyword_match_score = self._calculate_keyword_match_score(prediction, test_query.expected_keywords)
            relevance_score = self._calculate_relevance_score(prediction, test_query.ground_truth)
            
            return TestResult(
                query=test_query.query,
                ground_truth=test_query.ground_truth,
                prediction=prediction,
                confidence=confidence,
                response_time=response_time,
                accuracy_score=accuracy_score,
                keyword_match_score=keyword_match_score,
                relevance_score=relevance_score,
                category=test_query.category
            )
            
        except Exception as e:
            print(f"Error testing query '{test_query.query}': {e}")
            return TestResult(
                query=test_query.query,
                ground_truth=test_query.ground_truth,
                prediction="Error occurred",
                confidence=0.0,
                response_time=0.0,
                accuracy_score=0.0,
                keyword_match_score=0.0,
                relevance_score=0.0,
                category=test_query.category
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all test queries."""
        print("🚀 Starting RAG Performance Test")
        print("="*60)
        print(f"Testing {len(self.test_queries)} queries...")
        print()
        
        results = []
        
        for i, test_query in enumerate(self.test_queries, 1):
            print(f"Testing {i}/{len(self.test_queries)}: {test_query.query}")
            
            result = self.run_single_test(test_query)
            results.append(result)
            
            print(f"  Accuracy: {result.accuracy_score:.2f}")
            print(f"  Keyword Match: {result.keyword_match_score:.2f}")
            print(f"  Relevance: {result.relevance_score:.2f}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Response Time: {result.response_time:.2f}s")
            print()
        
        return results
    
    def analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """Analyze test results."""
        if not results:
            return {}
        
        # Overall statistics
        total_queries = len(results)
        avg_accuracy = sum(r.accuracy_score for r in results) / total_queries
        avg_keyword_match = sum(r.keyword_match_score for r in results) / total_queries
        avg_relevance = sum(r.relevance_score for r in results) / total_queries
        avg_confidence = sum(r.confidence for r in results) / total_queries
        avg_response_time = sum(r.response_time for r in results) / total_queries
        
        # Category-wise analysis
        categories = {}
        for result in results:
            cat = result.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        category_stats = {}
        for cat, cat_results in categories.items():
            cat_accuracy = sum(r.accuracy_score for r in cat_results) / len(cat_results)
            cat_keyword = sum(r.keyword_match_score for r in cat_results) / len(cat_results)
            cat_relevance = sum(r.relevance_score for r in cat_results) / len(cat_results)
            category_stats[cat] = {
                'count': len(cat_results),
                'avg_accuracy': cat_accuracy,
                'avg_keyword_match': cat_keyword,
                'avg_relevance': cat_relevance
            }
        
        # Success rate (accuracy > 0.5)
        successful_queries = sum(1 for r in results if r.accuracy_score > 0.5)
        success_rate = successful_queries / total_queries
        
        return {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': success_rate,
            'avg_accuracy': avg_accuracy,
            'avg_keyword_match': avg_keyword_match,
            'avg_relevance': avg_relevance,
            'avg_confidence': avg_confidence,
            'avg_response_time': avg_response_time,
            'category_stats': category_stats,
            'results': results
        }
    
    def print_report(self, analysis: Dict[str, Any]):
        """Print comprehensive test report."""
        print("📊 RAG PERFORMANCE TEST REPORT")
        print("="*60)
        print()
        
        print("📈 OVERALL STATISTICS:")
        print(f"  Total Queries: {analysis['total_queries']}")
        print(f"  Successful Queries: {analysis['successful_queries']}")
        print(f"  Success Rate: {analysis['success_rate']:.1%}")
        print(f"  Average Accuracy: {analysis['avg_accuracy']:.3f}")
        print(f"  Average Keyword Match: {analysis['avg_keyword_match']:.3f}")
        print(f"  Average Relevance: {analysis['avg_relevance']:.3f}")
        print(f"  Average Confidence: {analysis['avg_confidence']:.3f}")
        print(f"  Average Response Time: {analysis['avg_response_time']:.3f}s")
        print()
        
        print("📋 CATEGORY-WISE ANALYSIS:")
        for category, stats in analysis['category_stats'].items():
            print(f"  {category.upper()}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Accuracy: {stats['avg_accuracy']:.3f}")
            print(f"    Avg Keyword Match: {stats['avg_keyword_match']:.3f}")
            print(f"    Avg Relevance: {stats['avg_relevance']:.3f}")
            print()
        
        print("🎯 DETAILED RESULTS:")
        for i, result in enumerate(analysis['results'], 1):
            print(f"  {i:2d}. {result.query[:50]}...")
            print(f"      Accuracy: {result.accuracy_score:.3f}")
            print(f"      Keyword Match: {result.keyword_match_score:.3f}")
            print(f"      Relevance: {result.relevance_score:.3f}")
            print(f"      Confidence: {result.confidence:.3f}")
            print(f"      Time: {result.response_time:.3f}s")
            print()
    
    def save_results(self, analysis: Dict[str, Any], filename: str = "rag_test_results.json"):
        """Save test results to JSON file."""
        # File saving functionality has been removed
        print("💾 File saving functionality has been removed")


def main():
    """Run the RAG performance test."""
    tester = RAGPerformanceTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Analyze results
    analysis = tester.analyze_results(results)
    
    # Print report
    tester.print_report(analysis)
    
    # Save results
    tester.save_results(analysis)
    
    print("✅ RAG Performance Test Completed!")


if __name__ == "__main__":
    main()
