"""
Fintech-specific test datasets for LLM evaluation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import random


@dataclass
class TestCase:
    """A single test case for LLM evaluation."""
    id: str
    category: str
    subcategory: str
    query: str
    expected_answer: str
    context: Optional[str] = None
    difficulty: str = "medium"
    domain: str = "general"
    metadata: Optional[Dict[str, Any]] = None


class FintechTestDataset:
    """Curated test datasets for fintech LLM evaluation."""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self._load_default_datasets()
    
    def _load_default_datasets(self):
        """Load default fintech test datasets."""
        self._load_payment_cases()
        self._load_compliance_cases()
        self._load_customer_service_cases()
    
    def _load_payment_cases(self):
        """Payment processing test cases."""
        cases = [
            TestCase(
                id="PAY001",
                category="Payment Processing",
                subcategory="Payment Status",
                query="How can I check my payment status?",
                expected_answer="You can check your payment status by logging into your account dashboard and navigating to the 'Billing' section.",
                difficulty="easy",
                domain="payment"
            ),
            TestCase(
                id="PAY002", 
                category="Payment Processing",
                subcategory="Refunds",
                query="What is your refund policy?",
                expected_answer="We offer a 30-day money-back guarantee. If you're not completely satisfied with our service, you can request a full refund within 30 days of your purchase.",
                difficulty="easy",
                domain="payment"
            ),
            TestCase(
                id="PAY003",
                category="Payment Processing", 
                subcategory="Payment Methods",
                query="What payment methods do you accept?",
                expected_answer="We accept all major credit cards (Visa, MasterCard, American Express), debit cards, bank transfers, and digital wallets including PayPal and Apple Pay.",
                difficulty="easy",
                domain="payment"
            )
        ]
        self.test_cases.extend(cases)
    
    def _load_compliance_cases(self):
        """Compliance test cases."""
        cases = [
            TestCase(
                id="COMP001",
                category="Compliance",
                subcategory="Data Security",
                query="How do you protect customer payment information?",
                expected_answer="We use industry-standard encryption and security measures to protect your personal and payment information. We never store your credit card details on our servers.",
                difficulty="medium",
                domain="compliance"
            ),
            TestCase(
                id="COMP002",
                category="Compliance", 
                subcategory="PCI Compliance",
                query="Do you follow PCI DSS standards?",
                expected_answer="Yes, we use secure, encrypted payment gateways and follow PCI DSS standards for handling sensitive payment data.",
                difficulty="hard",
                domain="compliance"
            )
        ]
        self.test_cases.extend(cases)
    
    def _load_customer_service_cases(self):
        """Customer service test cases."""
        cases = [
            TestCase(
                id="CS001",
                category="Customer Service",
                subcategory="Account Management",
                query="How do I reset my password?",
                expected_answer="Click on 'Forgot Password' on the login page, enter your email address, and follow the instructions sent to your email. The reset link will expire in 24 hours.",
                difficulty="easy",
                domain="customer_service"
            ),
            TestCase(
                id="CS002",
                category="Customer Service",
                subcategory="Support Hours",
                query="What are your customer support hours?",
                expected_answer="Our customer support team is available Monday through Friday, 9 AM to 6 PM EST. For urgent issues outside these hours, you can submit a ticket and we'll respond within 24 hours.",
                difficulty="easy",
                domain="customer_service"
            )
        ]
        self.test_cases.extend(cases)
    
    def get_cases_by_category(self, category: str) -> List[TestCase]:
        """Get test cases by category."""
        return [case for case in self.test_cases if case.category == category]
    
    def get_cases_by_domain(self, domain: str) -> List[TestCase]:
        """Get test cases by domain."""
        return [case for case in self.test_cases if case.domain == domain]
    
    def get_random_sample(self, size: int = 10) -> List[TestCase]:
        """Get a random sample of test cases."""
        return random.sample(self.test_cases, min(size, len(self.test_cases)))
    
    def get_all_cases(self) -> List[TestCase]:
        """Get all test cases."""
        return self.test_cases.copy() 