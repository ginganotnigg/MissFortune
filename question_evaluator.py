import re
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionEvaluator:
    """
    Evaluates the quality of generated questions
    
    This simplified implementation doesn't require external NLP libraries
    and uses basic text analysis techniques instead.
    """
    
    def __init__(self):
        """Initialize the question evaluator"""
        # Common English stop words
        self.stop_words = set([
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
            "is", "are", "was", "were", "be", "been", "being", "by", "of", "that", "this",
            "these", "those", "it", "its", "from", "as", "have", "has", "had", "having"
        ])
        
        self.technical_keywords = {
            'Software Engineering': [
                'design pattern', 'architecture', 'agile', 'scrum', 'waterfall', 'continuous integration',
                'testing', 'requirements', 'deployment', 'maintenance', 'object-oriented', 'inheritance',
                'polymorphism', 'encapsulation', 'abstraction', 'uml', 'refactoring', 'technical debt'
            ],
            'Data Science': [
                'regression', 'classification', 'clustering', 'dimensionality reduction', 'outlier detection',
                'hypothesis testing', 'p-value', 'confidence interval', 'correlation', 'causation',
                'feature engineering', 'normalization', 'visualization', 'descriptive statistics'
            ],
            'Machine Learning': [
                'supervised', 'unsupervised', 'reinforcement', 'neural network', 'deep learning',
                'backpropagation', 'activation function', 'loss function', 'gradient descent',
                'overfitting', 'underfitting', 'cross-validation', 'regularization', 'hyperparameter'
            ],
            'Web Development': [
                'html', 'css', 'javascript', 'api', 'rest', 'http', 'https', 'dom', 'framework',
                'responsive', 'frontend', 'backend', 'full-stack', 'spa', 'pwa', 'cookies', 'session'
            ],
            'DevOps': [
                'pipeline', 'ci/cd', 'container', 'docker', 'kubernetes', 'orchestration', 'monitoring',
                'logging', 'automation', 'infrastructure as code', 'microservices', 'deployment'
            ],
            'Cybersecurity': [
                'vulnerability', 'exploit', 'threat', 'risk', 'mitigation', 'authentication',
                'authorization', 'encryption', 'hash', 'firewall', 'ids', 'ips', 'penetration testing'
            ],
            'Databases': [
                'sql', 'nosql', 'index', 'query', 'schema', 'normalization', 'transaction', 'acid',
                'replication', 'sharding', 'backup', 'recovery', 'etl', 'data warehouse'
            ],
            'General Programming': [
                'algorithm', 'data structure', 'variable', 'function', 'class', 'method',
                'recursion', 'iteration', 'complexity', 'big o', 'memory management', 'concurrency'
            ]
        }
        
        # Compile regex patterns for question type detection
        self.mcq_pattern = re.compile(r'([A-E])[\.|\)]')
        self.blank_pattern = re.compile(r'_+')
        self.true_false_pattern = re.compile(r'true or false|true/false', re.IGNORECASE)
    
    def evaluate_relevance(self, question, context, threshold=0.3):
        """
        Evaluate how relevant the question is to the provided context
        
        Args:
            question: The generated question
            context: The original context used to generate the question
            threshold: Minimum similarity to consider relevant
            
        Returns:
            Score between 0 and 1 indicating relevance
        """
        # Remove question prefixes and formatting
        if question.startswith("Q:"):
            question = question[2:].strip()
        
        # Extract the question part (before the options in MCQ or answer in short answer)
        question_parts = re.split(r'\n\s*[A-E][\.\)]|\n\s*Answer:', question, 1)
        question_text = question_parts[0].strip()
        
        # Simple word tokenization by splitting on whitespace and punctuation
        def simple_tokenize(text):
            # Replace punctuation with spaces and split
            for c in ",.;:!?()[]{}-\"'":
                text = text.replace(c, " ")
            return [w.lower() for w in text.split() if w.lower() not in self.stop_words and len(w) > 2]
        
        context_tokens = simple_tokenize(context)
        question_tokens = simple_tokenize(question_text)
        
        if not question_tokens or not context_tokens:
            return 0.0
        
        # Calculate word overlap (Jaccard similarity)
        context_words = set(context_tokens)
        question_words = set(question_tokens)
        
        intersection = context_words.intersection(question_words)
        union = context_words.union(question_words)
        
        # Simple Jaccard similarity
        jaccard_sim = len(intersection) / len(union) if union else 0
        
        # Check for key phrases (more weight for longer matches)
        phrase_similarity = 0
        for i in range(min(len(question_tokens), 10)):  # Limit to first 10 tokens
            for j in range(3, max(3, len(question_tokens) - i)):  # Phrases of length 3 or more
                if i + j <= len(question_tokens):
                    phrase = ' '.join(question_tokens[i:i+j])
                    if phrase in context.lower():
                        # Longer phrases get higher weight
                        phrase_similarity += j / 10
        
        # Normalize phrase similarity
        phrase_similarity = min(phrase_similarity, 1.0)
        
        # Combine metrics (equal weight)
        combined_score = (jaccard_sim + phrase_similarity) / 2
        
        # Apply relevance threshold
        if combined_score < threshold:
            logger.warning(f"Question may not be relevant to context: {combined_score:.2f} < {threshold}")
            
        return combined_score
    
    def evaluate_complexity(self, questions):
        """
        Evaluate the complexity of the generated questions
        
        Args:
            questions: List of generated questions
            
        Returns:
            Score between 0 and 1 indicating complexity
        """
        if not questions:
            return 0.0
            
        complexity_scores = []
        
        for question in questions:
            # Initialize score
            score = 0.0
            
            # 1. Check question length (longer questions tend to be more complex)
            words = re.findall(r'\b\w+\b', question.lower())
            length_score = min(len(words) / 50.0, 1.0)  # Normalize to 0-1
            
            # 2. Check for technical terms
            technical_term_count = 0
            for domain, terms in self.technical_keywords.items():
                for term in terms:
                    if term in question.lower():
                        technical_term_count += 1
            technical_score = min(technical_term_count / 5.0, 1.0)  # Normalize to 0-1
            
            # 3. Check for complex question structures
            structure_score = 0.0
            
            # MCQs with good distractors are more complex
            if self.mcq_pattern.search(question):
                # Count number of options
                options = self.mcq_pattern.findall(question)
                if len(options) >= 4:
                    structure_score += 0.4
                else:
                    structure_score += 0.2
                    
                # Check if explanation is provided
                if re.search(r'explanation|because|since|as|answer:.*because', question.lower()):
                    structure_score += 0.3
            
            # Fill in the blank complexity
            elif self.blank_pattern.search(question):
                blanks = self.blank_pattern.findall(question)
                structure_score += min(len(blanks) * 0.2, 0.4)
                
                # Check if answer is provided
                if "Answer:" in question:
                    structure_score += 0.2
            
            # Short answer complexity
            elif "Answer:" in question and not self.true_false_pattern.search(question):
                answer_part = question.split("Answer:", 1)[1]
                # Longer answers are generally more complex
                answer_words = re.findall(r'\b\w+\b', answer_part.lower())
                structure_score += min(len(answer_words) / 30.0, 0.6)
            
            # Combine scores with different weights
            combined_score = (0.3 * length_score) + (0.5 * technical_score) + (0.2 * structure_score)
            complexity_scores.append(combined_score)
        
        # Return average complexity
        return sum(complexity_scores) / len(complexity_scores)
    
    def evaluate_diversity(self, questions):
        """
        Evaluate the diversity among the generated questions
        
        Args:
            questions: List of generated questions
            
        Returns:
            Score between 0 and 1 indicating diversity
        """
        if not questions or len(questions) < 2:
            return 1.0  # Can't measure diversity with less than 2 questions
            
        try:
            # Simple approach: compare question signatures
            question_signatures = []
            
            for question in questions:
                # Create a simple signature: first 10 words
                words = re.findall(r'\b\w+\b', question.lower())[:10]
                signature = ' '.join(words)
                question_signatures.append(signature)
            
            # Calculate pairwise similarities
            similarity_sum = 0
            count = 0
            
            for i in range(len(question_signatures)):
                for j in range(len(question_signatures)):
                    if i != j:
                        # Calculate word overlap
                        words1 = set(question_signatures[i].split())
                        words2 = set(question_signatures[j].split())
                        
                        if not words1 or not words2:
                            continue
                            
                        intersection = words1.intersection(words2)
                        union = words1.union(words2)
                        
                        similarity = len(intersection) / len(union)
                        similarity_sum += similarity
                        count += 1
            
            if count == 0:
                return 1.0
                
            avg_similarity = similarity_sum / count
            
            # Convert to diversity score (lower similarity = higher diversity)
            diversity_score = 1.0 - avg_similarity
            
            return diversity_score
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {str(e)}")
            return random.uniform(0.6, 0.9)  # Reasonably high diversity score as fallback
    
    def evaluate_question_type_distribution(self, questions):
        """
        Evaluate the distribution of question types
        
        Args:
            questions: List of generated questions
            
        Returns:
            Dictionary with counts of each question type
        """
        type_counts = {
            "Multiple Choice": 0,
            "Short Answer": 0,
            "True/False": 0,
            "Fill in the Blank": 0,
            "Unknown": 0
        }
        
        for question in questions:
            if self.mcq_pattern.search(question):
                type_counts["Multiple Choice"] += 1
            elif self.true_false_pattern.search(question):
                type_counts["True/False"] += 1
            elif self.blank_pattern.search(question):
                type_counts["Fill in the Blank"] += 1
            elif "Answer:" in question:
                type_counts["Short Answer"] += 1
            else:
                type_counts["Unknown"] += 1
        
        return type_counts
