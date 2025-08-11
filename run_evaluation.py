#!/usr/bin/env python3
"""
Main evaluation script for NPCI_LLM fintech system.

This script demonstrates how to use the evaluation framework to benchmark
your fintech LLM against curated test datasets.

Usage:
    python run_evaluation.py --quick          # Run quick benchmark
    python run_evaluation.py --comprehensive  # Run comprehensive benchmark
    python run_evaluation.py --all            # Run all benchmarks
    python run_evaluation.py --category "Payment Processing"  # Run category benchmark
    python run_evaluation.py --domain "payment"               # Run domain benchmark
"""

import sys
import argparse
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.benchmarks import FintechBenchmark
from evaluation.datasets import FintechTestDataset


def mock_llm_function(query: str) -> str:
    """
    Mock LLM function for demonstration.
    
    In a real scenario, this would be your actual LLM inference function
    that takes a query and returns an answer.
    """
    # This is a simple mock - replace with your actual LLM function
    import random
    
    # Mock responses based on query keywords
    query_lower = query.lower()
    
    if "payment status" in query_lower:
        return "You can check your payment status by logging into your account dashboard and navigating to the 'Billing' section."
    
    elif "refund" in query_lower:
        return "We offer a 30-day money-back guarantee. If you're not completely satisfied with our service, you can request a full refund within 30 days of your purchase."
    
    elif "payment methods" in query_lower:
        return "We accept all major credit cards (Visa, MasterCard, American Express), debit cards, bank transfers, and digital wallets including PayPal and Apple Pay."
    
    elif "password" in query_lower:
        return "Click on 'Forgot Password' on the login page, enter your email address, and follow the instructions sent to your email. The reset link will expire in 24 hours."
    
    elif "support" in query_lower or "hours" in query_lower:
        return "Our customer support team is available Monday through Friday, 9 AM to 6 PM EST. For urgent issues outside these hours, you can submit a ticket and we'll respond within 24 hours."
    
    elif "security" in query_lower or "protect" in query_lower:
        return "We use industry-standard encryption and security measures to protect your personal and payment information. We never store your credit card details on our servers."
    
    elif "pci" in query_lower or "compliance" in query_lower:
        return "Yes, we use secure, encrypted payment gateways and follow PCI DSS standards for handling sensitive payment data."
    
    else:
        # Generic response for unknown queries
        responses = [
            "I understand your question about financial services. Let me help you with that.",
            "For financial inquiries, please contact our support team for detailed assistance.",
            "This is a common question about our services. Here's what you need to know.",
            "I can help you with information about our financial products and services."
        ]
        return random.choice(responses)


def integrate_with_your_llm():
    """
    Example of how to integrate with your actual LLM system.
    
    Replace this function with your actual LLM integration.
    """
    # Example 1: Using your RAG system
    def rag_llm_function(query: str) -> str:
        """
        Integrate with your existing RAG system.
        
        This would typically involve:
        1. Using your vector store to retrieve relevant documents
        2. Passing the query and context to your LLM
        3. Returning the generated response
        """
        try:
            # Import your RAG components
            # from embeddings.vector_store import QdrantVectorStore
            # from embeddings.embedder import LlamaIndexEmbedder
            # from generation.llm_generator import LLMGenerator
            
            # Example implementation:
            # vector_store = QdrantVectorStore()
            # embedder = LlamaIndexEmbedder()
            # generator = LLMGenerator()
            
            # query_embedding = embedder.embed_text(query)
            # relevant_docs = vector_store.search_similar(query_embedding, top_k=3)
            # context = "\n".join([doc['text'] for doc in relevant_docs])
            # response = generator.generate_response(query, context)
            
            # For now, return a placeholder
            return f"RAG Response: {query} (integration needed)"
            
        except Exception as e:
            return f"Error in RAG system: {str(e)}"
    
    return rag_llm_function


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Run fintech LLM evaluation benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--category", type=str, help="Run benchmark for specific category")
    parser.add_argument("--domain", type=str, help="Run benchmark for specific domain")
    parser.add_argument("--performance", action="store_true", help="Run performance benchmark")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG system instead of mock")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Choose LLM function
    if args.use_rag:
        print("🔗 Using RAG system for evaluation...")
        llm_function = integrate_with_your_llm()
    else:
        print("🎭 Using mock LLM function for demonstration...")
        llm_function = mock_llm_function
    
    # Initialize benchmark suite
    benchmark = FintechBenchmark(llm_function, args.output_dir)
    
    # Show available test cases
    dataset = FintechTestDataset()
    print(f"\n📊 Available test cases: {len(dataset.get_all_cases())}")
    print("Categories:", [cat for cat in set(case.category for case in dataset.get_all_cases())])
    print("Domains:", [dom for dom in set(case.domain for case in dataset.get_all_cases())])
    print()
    
    # Run selected benchmark
    if args.quick:
        print("🚀 Running Quick Benchmark...")
        benchmark.run_quick_benchmark()
    
    elif args.comprehensive:
        print("🔬 Running Comprehensive Benchmark...")
        benchmark.run_comprehensive_benchmark()
    
    elif args.category:
        print(f"📋 Running Category Benchmark: {args.category}")
        benchmark.run_category_benchmark(args.category)
    
    elif args.domain:
        print(f"🌐 Running Domain Benchmark: {args.domain}")
        benchmark.run_domain_benchmark(args.domain)
    
    elif args.performance:
        print("⚡ Running Performance Benchmark...")
        benchmark.run_performance_benchmark()
    
    elif args.all:
        print("🎯 Running All Benchmarks...")
        benchmark.run_all_benchmarks()
    
    else:
        # Default: run quick benchmark
        print("🚀 Running Quick Benchmark (default)...")
        benchmark.run_quick_benchmark()
    
    # Generate summary report
    print("\n" + "="*80)
    print("📄 GENERATING SUMMARY REPORT")
    print("="*80)
    summary_report = benchmark.generate_summary_report()
    print(summary_report)
    
    print(f"\n✅ Evaluation completed!")
    
    print("\n" + "="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print("1. Review the evaluation results in the output directory")
    print("2. Analyze performance by category and domain")
    print("3. Identify areas for improvement")
    print("4. Run comparison benchmarks against baseline models")
    print("5. Integrate with your actual LLM system using --use-rag flag")
    print("="*80)


if __name__ == "__main__":
    main() 