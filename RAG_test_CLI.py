"""
Simplified RAG System Test CLI

Updated CLI for testing the simplified RAG pipeline with predefined queries.
"""

import sys
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from env.server
load_dotenv(project_root / "env.server")
print(f"✅ Environment loaded from: {project_root / 'env.server'}")
print(f"🔧 SERVER_MODEL_ENDPOINT: {os.getenv('SERVER_MODEL_ENDPOINT', '❌ Not set')}")
print(f"🔑 SERVER_MODEL_API_KEY: {os.getenv('SERVER_MODEL_API_KEY', '❌ Not set or default')}")

# Import simplified components
import sys
sys.path.insert(0, str(Path(__file__).parent / "retrieval"))
from simple_retriever import SimpleSemanticRetriever

from generation.prompt_templates import PromptTemplates, PromptStyle
from generation.generator import ResponseGenerator, MockResponseGenerator
from generation.post_processor import PostProcessor
from config.logging import get_logger

logger = get_logger(__name__)


class SimplifiedRAGCLI:
    """Simplified RAG CLI for testing with predefined queries."""
    
    def __init__(self):
        """Initialize the simplified RAG system."""
        print("🚀 Initializing Simplified RAG System...")
        
        try:
            # Initialize simplified retriever
            self.retriever = SimpleSemanticRetriever()
            print("✅ Simple retriever initialized")
            
            # Initialize prompt templates
            self.prompt_templates = PromptTemplates()
            print("✅ Prompt templates initialized")
            
            # Initialize generator
            try:
                self.generator = ResponseGenerator()
                print("✅ Real Server generator initialized")
            except Exception as e:
                print(f"⚠️ Using mock generator: {e}")
                self.generator = MockResponseGenerator()
            
            # Initialize post processor
            self.post_processor = PostProcessor()
            print("✅ Post processor initialized")
            
            print("✅ Simplified RAG System ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            raise
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve context using simplified semantic retrieval."""
        print(f"\n🔍 Retrieving context for: '{query}'")
        
        try:
            start_time = time.time()
            results = self.retriever.retrieve(query, top_k=5)
            retrieval_time = time.time() - start_time
            
            if not results:
                print("⚠️ No results found")
                return "Based on available information, I'll provide a general response."
            
            # Build context from retrieved documents
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(f"Document {i} (Score: {result.score:.3f}):\n{result.text}\n")
            
            context = "\n".join(context_parts)
            
            print(f"✅ Retrieved {len(results)} results in {retrieval_time:.3f}s")
            print(f"📏 Context length: {len(context)} characters")
            
            return context
            
        except Exception as e:
            print(f"❌ Error in retrieval: {e}")
            return "Based on available information, I'll provide a general response."
    
    def build_prompt(self, query: str, context: str, style: str = "concise") -> str:
        """Build prompt using simplified context."""
        try:
            if style == "concise":
                prompt_style = PromptStyle.CONCISE
            elif style == "detailed":
                prompt_style = PromptStyle.DETAILED
            elif style == "technical":
                prompt_style = PromptStyle.TECHNICAL
            else:
                prompt_style = PromptStyle.CONCISE
            
            prompt = self.prompt_templates.format_prompt(
                query=query,
                context=context,
                style=prompt_style
            )
            
            print(f"✅ Built {style} prompt ({len(prompt)} characters)")
            return prompt
            
        except Exception as e:
            print(f"❌ Error building prompt: {e}")
            return f"Query: {query}\nContext: {context}\nPlease provide a helpful response."
    
    def generate_response(self, prompt: str) -> str:
        """Generate response."""
        try:
            start_time = time.time()
            response = self.generator.generate(prompt)
            generation_time = time.time() - start_time
            
            print(f"✅ Generated response in {generation_time:.3f}s")
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response."
    
    def process_response(self, response: str) -> dict:
        """Process response."""
        try:
            processed = self.post_processor.process(response)
            print("✅ Response processed")
            return processed
            
        except Exception as e:
            print(f"❌ Error processing response: {e}")
            return {
                "original_response": response,
                "processed_response": response,
                "confidence": 0.5
            }
    
    def display_response(self, query: str, processed_response: dict, query_num: int, total_queries: int):
        """Display response with query information."""
        print(f"\n{'='*80}")
        print(f"🎯 RAG RESPONSE #{query_num}/{total_queries}")
        print(f"🔍 QUERY: {query}")
        print(f"{'='*80}")
        
        if isinstance(processed_response, dict):
            # Handle different possible response keys
            response = (processed_response.get('processed_response') or 
                       processed_response.get('response') or 
                       processed_response.get('original_response') or 
                       'No response')
            confidence = processed_response.get('confidence', 0.5)
            
            print(f"📝 Response: {response}")
            print(f"🎯 Confidence: {confidence:.2f}")
        else:
            print(f"📝 Response: {processed_response}")
        
        print(f"{'='*80}")
    
    def run_query(self, query: str, style: str = "concise", query_num: int = 1, total_queries: int = 1):
        """Run a complete RAG query."""
        print(f"\n{'='*80}")
        print(f"🔍 RAG QUERY #{query_num}/{total_queries}: '{query}'")
        print(f"🎨 STYLE: {style}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Context retrieval
            context = self.retrieve_context(query)
            
            # Step 2: Build prompt
            prompt = self.build_prompt(query, context, style)
            
            # Step 3: Generate response
            response = self.generate_response(prompt)
            
            # Step 4: Process response
            processed = self.process_response(response)
            
            # Step 5: Display result
            self.display_response(query, processed, query_num, total_queries)
            
            return processed
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            return None
    
    def get_test_queries(self) -> list:
        """Get predefined test queries rephrased from the UPI decline document."""
        return [
            # INSUFFICIENT BALANCE - Scenario Questions (Rephrased)
            "My account shows sufficient funds, so why am I getting an insufficient balance error?",
            "I have an overdraft facility linked to my account, but the transaction still failed. Why?",
            "Could the system have debited from a different account than intended?",
            "I had 5000 rupees but the transaction failed for 4800. What's happening?",
            "I verified my balance this morning and it was fine, but now it's showing as insufficient. How?",
            "Is there a way for you to verify my current account balance?",
            
            # INSUFFICIENT BALANCE - FAQ Questions (Rephrased)
            "Is it possible to attempt the same payment again?",
            "Can I switch to a different bank account for this transaction?",
            "How can I set up alerts for low balance situations?",
            "Why wasn't I warned about insufficient funds before attempting the payment?",
            "Does a failed transaction have any impact on my account?",
            "Can I use my credit card through UPI when my savings account is low?",
            
            # INVALID PIN - Scenario Questions (Rephrased)
            "I think I might have made a typing error in my PIN. What should I do?",
            "I can't remember my UPI PIN. How do I proceed?",
            "The transaction failed even though I didn't get a chance to enter my PIN. Why?",
            "I'm confident I entered the right PIN, but it still failed. What's wrong?",
            "The system says I've entered incorrect PIN multiple times. What now?",
            "My PIN was working fine before, but now it's not working. Why?",
            "Can I use my fingerprint or face recognition instead of entering a PIN?",
            "I'm worried someone else might be using my UPI PIN without my knowledge. What should I do?",
            "I recently changed my SIM card. Could that be causing the PIN issues?",
            "I entered the wrong PIN yesterday. Is my account still locked from that?",
            
            # INVALID PIN - FAQ Questions (Rephrased)
            "Is there a way to view my current UPI PIN?",
            "What's the process to create a new UPI PIN?",
            "Why does UPI have such strict rules about incorrect PIN attempts?",
            "Can I use biometric authentication as a replacement for UPI PIN?",
            "Can you help me reset my UPI PIN through this chat?",
            "Are UPI PIN and ATM PIN the same thing?",
            "Why is a debit card required for PIN reset?",
            "What if I don't have my debit card available for PIN reset?",
            
            # COLLECT REQUEST EXPIRED - Scenario Questions (Rephrased)
            "What caused my UPI transaction to fail?",
            "I never received any payment request notification. Why?",
            "I opened my UPI app but couldn't find any pending requests. What happened?",
            "I was reviewing the details before approving, but the request expired. Can I still approve it?",
            "The shopkeeper says they sent the payment request, but it failed. What's the issue?",
            "I approved the request after some delay, but it still failed. Why?",
            "This keeps happening with the same merchant. What's going on?",
            
            # COLLECT REQUEST EXPIRED - FAQ Questions (Rephrased)
            "Is it possible to increase the time limit for payment requests?",
            "Will my account be charged even if the request has expired?",
            "Can I still approve a request after it has expired?",
            "Can I schedule a payment request for a later time?",
            "Why am I not receiving notifications for payment requests?",
            
            # PAYEE PSP TIMEOUT - Scenario Questions (Rephrased)
            "What went wrong with my payment?",
            "Is the issue with my UPI app or my bank account?",
            "I need to make an urgent payment. What are my options now?",
            "I tried paying a shopkeeper but the transaction failed. What happened?",
            "Payments to this specific person keep failing. Why?",
            "The person I'm trying to pay says they didn't receive any payment request. Why?",
            "Should I attempt the payment again right now?",
            "Does the person I'm paying need to do anything on their end?",
            
            # PAYEE PSP TIMEOUT - FAQ Questions (Rephrased)
            "Has any money been deducted from my account?",
            "Is there any security concern with this failure?",
            "Can I file a complaint against the payee's UPI app?",
            "How can I find out which UPI provider the payee is using?",
            "If I try again, will the same problem occur?",
            
            # PAYER VPA RESOLUTION FAILURE (Rephrased)
            "Why did my UPI payment get declined?",
            "I've successfully used this UPI ID in the past. Why is it failing now?",
            "I need to transfer money immediately. What can I do?",
            "Other people can make payments without issues, but I can't. Why?",
            "Has any amount been deducted from my account?",
            "Is there anything I can do to resolve this issue?",
            
            # MANDATE HOLD (Rephrased)
            "I have 15,000 rupees in my account, but the transaction failed. Why?",
            "What exactly is a mandate hold and how does it work?",
            "My transaction failed after I applied for an IPO. Is this related?",
            "How can I check how much balance is actually available for use?",
            "What payment options do I have right now?",
            "I never authorized any mandate. Why is this happening?",
            "When will the blocked funds be released back to me?",
            "The IPO closed yesterday, but my money is still blocked. When will it be free?",
            "Is it possible to cancel the mandate I set up?",
            "Will I be charged twice for the same transaction?",
            "Where can I see all my active mandates?",
            "My account shows the full balance, but I can't use it all. Why?",
            
            # BANK UNAVAILABLE (Rephrased)
            "Why is my bank's UPI service not working?",
            "My UPI app is functioning normally, but payments aren't going through. Why?",
            "It was working fine earlier today. What changed?",
            "I need to make a payment right now. What are my options?",
            "This happens frequently with my bank. Is this normal?",
            "Has any money been taken from my account?",
            "Would switching to a different UPI app help?",
            "How long will it take for the service to be restored?",
            "Is the problem with the UPI system or specifically my bank?",
            "Has my account been blocked or suspended?",
            
            # PAYER'S BANK SLOW RESPONSE (Rephrased)
            "My payment failed with error code U90. What does this mean?",
            "This has been happening repeatedly. What's causing it?",
            "I need to send money immediately. What should I do?",
            "My app is working fine, so why did the transaction fail?",
            "It worked on the second attempt. Why didn't it work the first time?",
            "Has any amount been debited from my account?",
            "Can I try the payment again now?",
            "Is this a security-related issue?",
            "Who should I contact about this problem?",
            "Should I consider changing my bank?",
            
            # FIRST TIME USER COOLING-OFF (Rephrased)
            "I just registered for UPI and my first transaction failed. Why?",
            "Why has my account been frozen for UPI transactions?",
            "This is my very first UPI payment and it's not working. What's wrong?",
            "Can this restriction be removed from my account?",
            "I need to make a payment urgently. What can I do?",
            "How much longer will this restriction last?",
            "Is there any way to get around this restriction?"
        ]
    
    def run_all_test_queries(self, style: str = "concise"):
        """Run all predefined test queries."""
        print("\n🧪 SIMPLIFIED RAG SYSTEM TEST")
        print("="*80)
        print(f"🎨 Style: {style}")
        print(f"📊 Running 40 predefined test queries")
        print("="*80)
        
        queries = self.get_test_queries()
        total_queries = len(queries)
        
        successful_queries = 0
        failed_queries = 0
        
        start_time = time.time()
        
        for i, query in enumerate(queries, 1):
            try:
                result = self.run_query(query, style, i, total_queries)
                if result:
                    successful_queries += 1
                else:
                    failed_queries += 1
                    
                # Add a small delay between queries
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Query {i} failed: {e}")
                failed_queries += 1
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*80}")
        print("📊 TEST SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successful queries: {successful_queries}")
        print(f"❌ Failed queries: {failed_queries}")
        print(f"📈 Success rate: {(successful_queries/total_queries)*100:.1f}%")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"⚡ Average time per query: {total_time/total_queries:.2f}s")
        print(f"{'='*80}")
        
        return {
            "successful": successful_queries,
            "failed": failed_queries,
            "total": total_queries,
            "success_rate": (successful_queries/total_queries)*100,
            "total_time": total_time,
            "avg_time_per_query": total_time/total_queries
        }


def main():
    """Run the simplified RAG CLI with predefined queries."""
    print("🚀 SIMPLIFIED RAG SYSTEM TEST CLI")
    print("="*80)
    
    try:
        # Initialize simplified RAG CLI
        rag_cli = SimplifiedRAGCLI()
        
        # Check if command line arguments provided for style
        style = "concise"
        if len(sys.argv) > 1:
            style_arg = sys.argv[1].lower()
            if style_arg in ["concise", "detailed", "technical"]:
                style = style_arg
            else:
                print(f"⚠️ Unknown style '{style_arg}', using 'concise'")
        
        # Run all test queries
        summary = rag_cli.run_all_test_queries(style)
        
        if summary["success_rate"] >= 80:
            print("🎉 TEST PASSED: High success rate achieved!")
        else:
            print("⚠️ TEST WARNING: Some queries failed")
            
    except Exception as e:
        print(f"❌ Failed to start simplified RAG CLI: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main() 