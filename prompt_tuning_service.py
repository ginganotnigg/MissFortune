import random
import re
import string
import logging
import hashlib
import os
from data_handler import DataHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptTuningService:
    """
    Optimized prompt tuning service for technical question generation.
    This implementation uses prompt engineering techniques and sample data
    to generate high-quality, context-relevant questions.
    """
    
    def __init__(self, model_name="customized-logic", device=None):
        """
        Initialize the prompt tuning service
        
        Args:
            model_name: Identifier for the prompt strategy to use
            device: Not used in this implementation but kept for API compatibility
        """
        self.model_name = model_name
        self.data_handler = DataHandler()
        self._load_question_templates()
        
    def _load_question_templates(self):
        """Load question templates for different question types and domains"""
        # Pre-defined templates with better context references
        self.templates = {
            "Multiple Choice": {
                "Software Engineering": [
                    "What is the main purpose of {concept} as described in the text?",
                    "According to the context, which of the following best describes {concept}?",
                    "Based on the provided information, when would you typically use {concept}?"
                ],
                "Machine Learning": [
                    "According to the context, what is the primary function of {concept} in machine learning?",
                    "Based on the provided information, which statement about {concept} is correct?",
                    "From the text, how does {concept} contribute to model performance?"
                ],
                "DEFAULT": [
                    "Based on the context, what is {concept} primarily used for?",
                    "According to the text, which description of {concept} is most accurate?",
                    "As described in the context, what is a key characteristic of {concept}?"
                ]
            },
            "Short Answer": {
                "Software Engineering": [
                    "Based on the provided context, explain the concept of {concept} and its importance in software development.",
                    "According to the text, how does {concept} help in creating maintainable code?",
                    "From the information provided, what problem does {concept} solve in software design?"
                ],
                "Machine Learning": [
                    "Using the context provided, describe how {concept} works in machine learning.",
                    "Based on the text, what role does {concept} play in model training?",
                    "According to the information given, explain the importance of {concept} in data preprocessing."
                ],
                "DEFAULT": [
                    "Using the provided context, explain {concept} and its primary application.",
                    "Based on the text, describe the key features of {concept}.",
                    "According to the information provided, how is {concept} implemented in practice?"
                ]
            },
            "True/False": {
                "DEFAULT": [
                    "Based on the context, {concept} is primarily used for {purpose}.",
                    "According to the text, the main benefit of {concept} is {benefit}.",
                    "As described in the context, {concept} is considered more efficient than {alternative}."
                ]
            },
            "Fill in the Blank": {
                "DEFAULT": [
                    "Based on the context, {concept} is a technique used to _______ in {domain}.",
                    "According to the text, the primary purpose of {concept} is to _______ within a system.",
                    "As described in the context, {concept} differs from {alternative} mainly in its ability to _______."
                ]
            }
        }
        
        # Try to load examples from sample data files to enhance templates
        self._enhance_templates_from_samples()
        
    def _enhance_templates_from_samples(self):
        """Load question patterns from sample data files to enhance templates"""
        # Load sample questions for each type
        for question_type in ["Multiple Choice", "Short Answer", "True/False", "Fill in the Blank"]:
            try:
                samples = self.data_handler.load_sample_questions(question_type, max_samples=10)
                
                if not samples:
                    continue
                    
                # Extract question patterns from samples
                for sample in samples:
                    question = sample.get("question", "")
                    
                    # Extract the pattern by replacing specific terms with placeholders
                    pattern = self._extract_question_pattern(question, question_type)
                    
                    if pattern:
                        # Add to templates if it's not already there
                        if question_type in ["Multiple Choice", "Short Answer"]:
                            for domain in self.templates[question_type]:
                                if pattern not in self.templates[question_type][domain]:
                                    self.templates[question_type][domain].append(pattern)
                        else:
                            if pattern not in self.templates[question_type]["DEFAULT"]:
                                self.templates[question_type]["DEFAULT"].append(pattern)
            except Exception as e:
                logger.warning(f"Error enhancing templates for {question_type}: {str(e)}")
                
    def _extract_question_pattern(self, question, question_type):
        """Extract a reusable pattern from a sample question"""
        if not question:
            return None
            
        # Remove True/False prefix or similar markers
        if question_type == "True/False" and question.lower().startswith("true or false:"):
            question = question[len("true or false:"):].strip()
            
        # Replace specific technical terms with {concept} placeholder
        # Look for capitalized words or phrases that might be concepts
        pattern = re.sub(r'\b[A-Z][a-zA-Z]*(?:\s+[a-z]+){0,2}\b', '{concept}', question)
        
        # For fill in the blank, ensure the blank is preserved
        if question_type == "Fill in the Blank":
            pattern = pattern.replace("_____", "_______")
            
        # Only use patterns that have the concept placeholder and are not too short
        if '{concept}' in pattern and len(pattern) > 15:
            return pattern
            
        return None
        
    def extract_key_concepts(self, context, num_concepts=5):
        """
        Extract key concepts from the context using improved NLP techniques
        
        Args:
            context: The technical content context
            num_concepts: Number of key concepts to extract
            
        Returns:
            List of extracted key concepts
        """
        # Remove common words
        common_words = set([
            "the", "and", "or", "a", "an", "in", "on", "at", "to", "for", "with", "by", 
            "about", "as", "is", "are", "was", "were", "be", "been", "being", "this", 
            "that", "these", "those", "it", "they", "them", "their", "his", "her", "its"
        ])
        
        # Look for technical terms using various patterns
        # First, multi-word phrases which are often important concepts
        phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){1,2}\b', context)
        
        # Second, look for relation phrases like "X of Y"
        relation_phrases = re.findall(r'\b([A-Za-z][a-z]+\s+(?:of|and|in|for|to)\s+[a-z]+(?:\s+[a-z]+)?)\b', context)
        
        # Third, find capitalized words which often indicate important terms
        capitalized = re.findall(r'\b([A-Z][a-z]{2,})\b', context)
        
        # Fourth, look for terms in quotes
        quoted = re.findall(r'[\'"]([^\'\"]+)[\'"]', context)
        
        # Fifth, extract words following bullets or numbering
        bullet_points = re.findall(r'(?:[-•*]\s+|\d+\.\s+)([A-Za-z][a-z]+(?:\s+[a-z]+){0,3})', context)
        
        # Finally, look for any longer words (likely technical terms)
        words = [w.lower() for w in re.findall(r'\b[A-Za-z][a-z]{4,}\b', context)]
        words = [w for w in words if w not in common_words]
        
        # Score all potential concepts
        concept_scores = {}
        
        # Give highest weight to multi-word technical phrases
        for phrase in phrases:
            phrase = phrase.lower()
            concept_scores[phrase] = concept_scores.get(phrase, 0) + 5
            
        # Relation phrases are also likely important
        for phrase in relation_phrases:
            phrase = phrase.lower()
            concept_scores[phrase] = concept_scores.get(phrase, 0) + 4
            
        # Capitalized terms often indicate important concepts
        for term in capitalized:
            term = term.lower()
            concept_scores[term] = concept_scores.get(term, 0) + 3
            
        # Quoted terms are explicitly highlighted
        for term in quoted:
            term = term.lower()
            concept_scores[term] = concept_scores.get(term, 0) + 3
            
        # Bullet points often have key concepts
        for point in bullet_points:
            point = point.lower()
            concept_scores[point] = concept_scores.get(point, 0) + 3
            
        # Regular words get lower weight
        for word in words:
            concept_scores[word] = concept_scores.get(word, 0) + 1
            
        # Remove common words and very short terms
        for word in list(concept_scores.keys()):
            if word in common_words or len(word) < 3:
                del concept_scores[word]
                
        # Get top concepts based on score
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, score in sorted_concepts[:num_concepts]]
        
    def apply_lora_tuning(self, training_data=None):
        """
        Simulate LoRA tuning by adapting internal templates based on training data
        
        Args:
            training_data: List of prompt/completion pairs
        """
        if not training_data:
            return
            
        # Extract patterns from training examples
        for item in training_data:
            completion = item.get("completion", "")
            
            if "A)" in completion or "A." in completion:
                mcq_pattern = self._extract_mcq_pattern(completion)
                if mcq_pattern:
                    # Update Multiple Choice templates
                    for domain in self.templates["Multiple Choice"]:
                        if mcq_pattern not in self.templates["Multiple Choice"][domain]:
                            self.templates["Multiple Choice"][domain].append(mcq_pattern)
            elif "True or False" in completion or "TRUE" in completion or "FALSE" in completion:
                # Update True/False templates
                tf_pattern = self._extract_true_false_pattern(completion)
                if tf_pattern and tf_pattern not in self.templates["True/False"]["DEFAULT"]:
                    self.templates["True/False"]["DEFAULT"].append(tf_pattern)
            elif "fill" in completion.lower() or "blank" in completion.lower() or "_____" in completion:
                # Update Fill in the Blank templates
                fb_pattern = self._extract_fill_blank_pattern(completion)
                if fb_pattern and fb_pattern not in self.templates["Fill in the Blank"]["DEFAULT"]:
                    self.templates["Fill in the Blank"]["DEFAULT"].append(fb_pattern)
            else:
                # Assume short answer
                sa_pattern = self._extract_short_answer_pattern(completion)
                if sa_pattern:
                    # Update Short Answer templates
                    for domain in self.templates["Short Answer"]:
                        if sa_pattern not in self.templates["Short Answer"][domain]:
                            self.templates["Short Answer"][domain].append(sa_pattern)
                            
    def _extract_mcq_pattern(self, completion):
        """Extract MCQ pattern from a completion example"""
        # Find the question text
        match = re.search(r'Q:(.+?)(?:A\)|A\.)', completion, re.DOTALL)
        if match:
            question = match.group(1).strip()
            # Replace specific concept with placeholder
            question = re.sub(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b', '{concept}', question, count=1)
            return question
        return None
        
    def _extract_short_answer_pattern(self, completion):
        """Extract short answer pattern from a completion example"""
        match = re.search(r'Q:(.+?)(?:A:|Answer:)', completion, re.DOTALL)
        if match:
            question = match.group(1).strip()
            # Replace specific concept with placeholder
            question = re.sub(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b', '{concept}', question, count=1)
            return question
        return None
        
    def _extract_true_false_pattern(self, completion):
        """Extract true/false pattern from a completion example"""
        match = re.search(r'True or False:(.+?)(?:Answer:|$)', completion, re.DOTALL)
        if match:
            statement = match.group(1).strip()
            # Replace specific concept with placeholder
            statement = re.sub(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b', '{concept}', statement, count=1)
            return statement
        return None
        
    def _extract_fill_blank_pattern(self, completion):
        """Extract fill-in-the-blank pattern from a completion example"""
        # Look for patterns with blanks
        match = re.search(r'Q:(.+?)(?:Answer:|$)', completion, re.DOTALL)
        if match:
            question = match.group(1).strip()
            # Make sure it contains a blank
            if "______" in question or "_____" in question:
                # Replace specific concept with placeholder
                question = re.sub(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b', '{concept}', question, count=1)
                return question
        return None
        
    def generate_from_prompt(self, prompt, max_length=512, temperature=0.7, top_p=0.9, 
                            num_return_sequences=1, do_sample=True):
        """
        Generate text based on the given prompt
        
        Args:
            prompt: Input prompt string
            max_length: Maximum length of the generated text (not used in this implementation)
            temperature: Controls randomness (higher = more random)
            top_p: Not used in this implementation
            num_return_sequences: Number of different responses to generate
            do_sample: Whether to use random sampling (not used in this implementation)
            
        Returns:
            List of generated text strings
        """
        # Extract parameters from the prompt
        domain = self._extract_domain(prompt)
        question_type = self._extract_question_type(prompt)
        difficulty = self._extract_difficulty(prompt)
        context = self._extract_context(prompt)
        
        # Try to find relevant sample questions from our dataset
        relevant_samples = self.data_handler.get_context_relevant_samples(
            context, question_type, max_samples=3
        )
        
        # Create a context hash to ensure consistent results for the same input
        context_hash = hashlib.md5(context.encode()).hexdigest()
        
        # Generate multiple responses if requested
        responses = []
        for i in range(num_return_sequences):
            # Use temperature to influence randomness
            randomness = temperature * (1 + i * 0.1)  # Slightly increase randomness for variety
            
            # Check if we have relevant samples to use as templates
            if relevant_samples and i < len(relevant_samples) and random.random() < 0.7:
                # Use a relevant sample as inspiration 70% of the time when available
                sample = relevant_samples[i]
                responses.append(self._generate_from_sample(
                    context, domain, question_type, difficulty, randomness, sample
                ))
            else:
                # Generate using our standard templates
                responses.append(self._generate_question(
                    context, domain, question_type, difficulty, randomness, context_hash
                ))
            
        return responses
        
    def _generate_from_sample(self, context, domain, question_type, difficulty, randomness, sample):
        """Generate a question using a sample as a template"""
        # Extract key concepts from our context
        concepts = self.extract_key_concepts(context)
        if not concepts:
            concepts = ["the concept", "the approach", "the technology"]
            
        # Choose a concept based on randomness
        random.seed(int(randomness * 1000))
        concept = random.choice(concepts[:min(3, len(concepts))])
        
        # Extract structure from the sample
        sample_question = sample.get("question", "")
        sample_answer = sample.get("answer", "")
        
        # Adapt the sample question to our context and concept
        new_question = self._adapt_question_to_context(sample_question, concept, domain)
        
        # Generate an appropriate answer based on question type
        if question_type == "Multiple Choice":
            return self._generate_mcq_from_sample(new_question, concept, context, domain, difficulty)
        elif question_type == "True/False":
            return self._generate_tf_from_sample(new_question, concept, context, domain, difficulty)
        elif question_type == "Fill in the Blank":
            return self._generate_fb_from_sample(new_question, concept, context, domain, difficulty)
        else:  # Short Answer
            return self._generate_sa_from_sample(new_question, concept, context, domain, difficulty)
            
    def _adapt_question_to_context(self, sample_question, concept, domain):
        """Adapt a sample question to use our context and concept"""
        # For True/False questions, remove the prefix
        if sample_question.startswith("True or False:"):
            question = sample_question[len("True or False:"):].strip()
        else:
            question = sample_question
            
        # Find technical terms in the sample question (likely concepts)
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b', question)
        
        # Replace the first technical term with our concept
        if tech_terms:
            new_question = question.replace(tech_terms[0], concept.title())
        else:
            # If no clear technical term, use a template approach
            if "what" in question.lower() or "how" in question.lower():
                new_question = f"What is the main purpose of {concept} in {domain}?"
            elif "explain" in question.lower() or "describe" in question.lower():
                new_question = f"Explain how {concept} is used in {domain}."
            elif "fill" in question.lower() or "_____" in question.lower():
                new_question = f"{concept.title()} is used to _______ in {domain}."
            else:
                new_question = f"{concept.title()} is a key component in {domain}."
                
        # For True/False questions, add back the prefix
        if sample_question.startswith("True or False:"):
            new_question = f"True or False: {new_question}"
            
        return new_question
        
    def _generate_mcq_from_sample(self, question, concept, context, domain, difficulty):
        """Generate a multiple-choice question based on a sample"""
        # Find sentences in the context related to the concept
        relevant_sentences = self._find_sentences_with_concept(context, concept)
        
        # Generate the correct answer based on the context
        correct_answer = self._extract_answer_from_context(concept, relevant_sentences, domain)
        
        # Generate wrong options
        wrong_answers = self._generate_wrong_options(concept, context, domain, difficulty)
        
        # Combine and shuffle options
        options = [correct_answer] + wrong_answers
        random.shuffle(options)
        
        # Format the question
        result = f"Q: {question}\n\n"
        for i, option in enumerate(options):
            result += f"{chr(65 + i)}. {option}\n"
        
        # Add the correct answer
        correct_index = options.index(correct_answer)
        result += f"\nAnswer: {chr(65 + correct_index)}"
        
        return result
        
    def _generate_tf_from_sample(self, question, concept, context, domain, difficulty):
        """Generate a true/false question based on a sample"""
        # Strip the "True or False:" prefix if it exists
        if question.startswith("True or False:"):
            statement = question[len("True or False:"):].strip()
        else:
            statement = question
            
        # Find sentences in the context related to the concept
        relevant_sentences = self._find_sentences_with_concept(context, concept)
        
        # Determine if the statement should be true or false
        is_true = random.choice([True, True, False])  # Bias toward true statements
        
        if not is_true:
            # Make the statement false by modifying it
            negations = ["not", "rarely", "seldom", "never"]
            
            # Look for places to insert negations
            if " is " in statement:
                statement = statement.replace(" is ", f" is {random.choice(negations)} ")
            elif " are " in statement:
                statement = statement.replace(" are ", f" are {random.choice(negations)} ")
            elif " can " in statement:
                statement = statement.replace(" can ", f" {random.choice(negations)} ")
            else:
                # If no good place to negate, replace with an opposite meaning
                statement = f"{concept.title()} is {random.choice(negations)} used in {domain} as described in the context."
                
        # Create an explanation based on the relevant sentences
        if relevant_sentences:
            explanation = f" {relevant_sentences[0]}"
        else:
            if is_true:
                explanation = f" The statement accurately reflects the purpose of {concept} in {domain}."
            else:
                explanation = f" The statement contradicts the actual purpose of {concept} in {domain}."
                
        # Format the true/false question
        return f"True or False: {statement}\n\nAnswer: {'TRUE' if is_true else 'FALSE'}.{explanation}"
        
    def _generate_fb_from_sample(self, question, concept, context, domain, difficulty):
        """Generate a fill-in-the-blank question based on a sample"""
        # Find sentences in the context related to the concept
        relevant_sentences = self._find_sentences_with_concept(context, concept)
        
        # Make sure the question has a blank marker
        if "_____" not in question and "______" not in question:
            question = f"{concept.title()} is used to _______ in {domain}."
            
        # Determine what should fill the blank based on the context
        answer = self._extract_blank_filler(concept, relevant_sentences, domain)
        
        # Format the fill-in-the-blank question
        return f"Q: {question}\n\nAnswer: {answer}"
        
    def _generate_sa_from_sample(self, question, concept, context, domain, difficulty):
        """Generate a short-answer question based on a sample"""
        # Find sentences in the context related to the concept
        relevant_sentences = self._find_sentences_with_concept(context, concept)
        
        # Generate an answer based on difficulty and available context
        if relevant_sentences:
            if difficulty == "Easy":
                answer = relevant_sentences[0]
            elif difficulty == "Medium":
                answer = " ".join(relevant_sentences[:min(2, len(relevant_sentences))])
            else:  # Hard or Expert
                answer = " ".join(relevant_sentences)
                
                # Add synthesis for higher difficulties
                if difficulty == "Expert" and len(relevant_sentences) > 1:
                    answer += f" This makes {concept} particularly valuable in {domain} when working on complex systems."
        else:
            # Fallback if no sentences mention the concept
            answer = f"{concept.title()} is an important concept in {domain} that helps solve common technical challenges."
            
            if difficulty != "Easy":
                answer += f" It's typically used during the design and implementation phases to ensure better code quality and maintainability."
                
            if difficulty == "Expert":
                answer += f" When properly applied, it can significantly reduce complexity and improve the long-term viability of technical solutions in {domain}."
                
        # Format the short-answer question
        return f"Q: {question}\n\nAnswer: {answer}"
        
    def _extract_domain(self, prompt):
        """Extract domain from prompt"""
        domains = ["Software Engineering", "Machine Learning", "Web Development", "Database Design"]
        for domain in domains:
            if domain.lower() in prompt.lower():
                return domain
        return "Software Engineering"  # Default domain
        
    def _extract_question_type(self, prompt):
        """Extract question type from prompt"""
        types = ["Multiple Choice", "Short Answer", "True/False", "Fill in the Blank"]
        for qtype in types:
            if qtype.lower() in prompt.lower():
                return qtype
        return "Multiple Choice"  # Default type
        
    def _extract_difficulty(self, prompt):
        """Extract difficulty from prompt"""
        difficulties = ["Easy", "Medium", "Hard", "Expert"]
        for diff in difficulties:
            if diff.lower() in prompt.lower():
                return diff
        return "Medium"  # Default difficulty
        
    def _extract_context(self, prompt):
        """Extract context from prompt"""
        # Look for the context after "context:" or similar markers
        markers = ["context:", "technical content:", "following context:", "following text:"]
        for marker in markers:
            if marker in prompt.lower():
                parts = prompt.lower().split(marker, 1)
                if len(parts) > 1:
                    return parts[1].strip()
                    
        # Fallback: Take the last paragraph of the prompt
        paragraphs = prompt.split('\n\n')
        return paragraphs[-1] if paragraphs else prompt
        
    def _generate_question(self, context, domain, question_type, difficulty, randomness, context_hash):
        """Generate a question based on the given parameters"""
        # Extract key concepts from the context
        concepts = self.extract_key_concepts(context)
        if not concepts:
            concepts = ["the concept", "the approach", "the technology"]
            
        # Choose a concept based on randomness and context hash
        seed = int(context_hash[:8], 16) % 1000
        random.seed(seed + int(randomness * 100))
        concept = random.choice(concepts[:3])  # Focus on top concepts
        
        # Find sentences related to the concept
        relevant_sentences = self._find_sentences_with_concept(context, concept)
        
        # Generate appropriate question type
        if question_type == "Multiple Choice":
            return self._generate_mcq(context, domain, difficulty, concepts, relevant_sentences)
        elif question_type == "Short Answer":
            return self._generate_short_answer(context, domain, difficulty, concepts, relevant_sentences)
        elif question_type == "True/False":
            return self._generate_true_false(context, domain, difficulty, concepts, relevant_sentences)
        elif question_type == "Fill in the Blank":
            return self._generate_fill_blank(context, domain, difficulty, concepts, relevant_sentences)
        else:
            return self._generate_mcq(context, domain, difficulty, concepts, relevant_sentences)
            
    def _generate_mcq(self, context, domain, difficulty, concepts, relevant_sentences):
        """Generate a multiple-choice question"""
        # Get domain-specific templates or default templates
        domain_templates = self.templates["Multiple Choice"].get(
            domain, self.templates["Multiple Choice"]["DEFAULT"]
        )
        
        # Choose a template and concept
        template = random.choice(domain_templates)
        concept = random.choice(concepts[:3])
        
        # Generate question text
        question_text = template.format(concept=concept.title())
        
        # Extract the correct answer from relevant sentences
        correct_answer = self._extract_answer_from_context(concept, relevant_sentences, domain)
        
        # Generate wrong options
        wrong_answers = self._generate_wrong_options(concept, context, domain, difficulty)
        
        # Combine and shuffle options
        options = [correct_answer] + wrong_answers
        random.shuffle(options)
        
        # Format the question
        result = f"Q: {question_text}\n\n"
        for i, option in enumerate(options):
            result += f"{chr(65 + i)}. {option}\n"
        
        # Add the correct answer
        correct_index = options.index(correct_answer)
        result += f"\nAnswer: {chr(65 + correct_index)}"
        
        return result
        
    def _extract_answer_from_context(self, concept, relevant_sentences, domain):
        """Extract a correct answer from the context for a concept"""
        if not relevant_sentences:
            # Fallback definitions based on domain
            if domain == "Software Engineering":
                return f"A technique for improving code organization and maintainability"
            elif domain == "Machine Learning":
                return f"A method for enhancing model performance and accuracy"
            else:
                return f"A fundamental concept that helps solve common problems in {domain}"
                
        # Use the most relevant sentence to create an answer
        main_sentence = relevant_sentences[0]
        
        # Look for definition patterns
        concept_lower = concept.lower()
        if concept_lower in main_sentence.lower():
            # Find where the concept appears
            idx = main_sentence.lower().find(concept_lower)
            
            # Look for definition patterns after the concept
            after_concept = main_sentence[idx + len(concept_lower):].strip()
            
            # Common verbs or phrases that introduce definitions
            definition_markers = ["is", "are", "refers to", "means", "represents", "is defined as", "helps", "allows", "enables"]
            
            for marker in definition_markers:
                if marker in after_concept.lower():
                    # Extract the definition part
                    start_idx = after_concept.lower().find(marker) + len(marker)
                    definition = after_concept[start_idx:].strip()
                    
                    # Clean up the definition
                    definition = definition.strip(".,;:()[] ")
                    if definition and len(definition) > 10:
                        return definition.capitalize()
                        
        # If no clear definition found, try to extract key phrases
        words = main_sentence.split()
        if len(words) > 8:
            # Extract a meaningful chunk of the sentence
            mid_point = len(words) // 2
            snippet = " ".join(words[max(0, mid_point - 4):min(len(words), mid_point + 4)])
            return snippet.capitalize()
            
        # Last resort: use the whole sentence
        return main_sentence.capitalize()
        
    def _generate_wrong_options(self, concept, context, domain, difficulty):
        """Generate plausible but incorrect options for multiple choice"""
        # Extract other concepts to use in wrong answers
        concepts = self.extract_key_concepts(context)
        other_concepts = [c for c in concepts if c != concept][:3]
        
        wrong_answers = []
        
        # Generate wrong answers based on other concepts
        for i in range(min(3, len(other_concepts))):
            other = other_concepts[i]
            sentences = self._find_sentences_with_concept(context, other)
            
            if sentences:
                # Extract information about the other concept
                wrong_info = self._extract_answer_from_context(other, sentences, domain)
                wrong_answers.append(wrong_info)
            else:
                # Fallback wrong answers
                if domain == "Software Engineering":
                    wrong_answers.append(f"A database optimization technique unrelated to {concept}")
                elif domain == "Machine Learning":
                    wrong_answers.append(f"A data visualization method not commonly used with {concept}")
                else:
                    wrong_answers.append(f"An alternative approach that conflicts with {concept} principles")
                    
        # Ensure we have exactly 3 wrong answers
        while len(wrong_answers) < 3:
            if domain == "Software Engineering":
                generic_wrongs = [
                    "A deprecated technique no longer used in modern software development",
                    "A method primarily used for hardware optimization, not software",
                    "A theoretical concept with limited practical applications"
                ]
            elif domain == "Machine Learning":
                generic_wrongs = [
                    "A technique only applicable to unsupervised learning problems",
                    "A method that decreases model accuracy but improves speed",
                    "A specialized approach only used in computer vision tasks"
                ]
            else:
                generic_wrongs = [
                    "A concept from a different field incorrectly applied to this domain",
                    "A common misconception that leads to design problems",
                    "An outdated approach replaced by modern techniques"
                ]
                
            for wrong in generic_wrongs:
                if wrong not in wrong_answers:
                    wrong_answers.append(wrong)
                    if len(wrong_answers) >= 3:
                        break
                        
        return wrong_answers[:3]
        
    def _generate_short_answer(self, context, domain, difficulty, concepts, relevant_sentences):
        """Generate a short-answer question"""
        # Get domain-specific templates or default templates
        domain_templates = self.templates["Short Answer"].get(
            domain, self.templates["Short Answer"]["DEFAULT"]
        )
        
        # Choose a template and concept
        template = random.choice(domain_templates)
        concept = random.choice(concepts[:3])
        
        # Choose a related concept for the template if needed
        related_concept = random.choice([c for c in concepts if c != concept] or ["related technology"])
        
        # Generate question text
        question_text = template.format(
            concept=concept.title(),
            related_concept=related_concept.title()
        )
        
        # Generate answer based on relevant sentences and difficulty
        if relevant_sentences:
            if difficulty == "Easy":
                answer = relevant_sentences[0]
            elif difficulty == "Medium":
                answer = " ".join(relevant_sentences[:min(2, len(relevant_sentences))])
            else:  # Hard or Expert
                answer = " ".join(relevant_sentences)
                
                # Add synthesis for expert level
                if difficulty == "Expert" and len(relevant_sentences) > 1:
                    answer += f" This makes {concept.title()} particularly valuable in {domain} when working on complex systems that require maintainable and scalable solutions."
        else:
            # Fallback answers based on domain and difficulty
            if domain == "Software Engineering":
                answer = f"{concept.title()} is a software engineering concept that helps developers organize and maintain code efficiently."
            elif domain == "Machine Learning":
                answer = f"{concept.title()} is a technique in machine learning that contributes to model accuracy and performance."
            else:
                answer = f"{concept.title()} is a fundamental concept in {domain} that helps address common challenges."
                
            if difficulty == "Medium":
                answer += f" It is typically implemented during the design phase of projects."
            elif difficulty == "Hard":
                answer += f" When properly applied, it can significantly reduce complexity and improve maintainability."
            elif difficulty == "Expert":
                answer += f" Its effectiveness varies based on the specific context, and it often needs to be balanced with other considerations like performance and scalability."
                
        # Format the question
        result = f"Q: {question_text}\n\nAnswer: {answer}"
        
        return result
        
    def _generate_true_false(self, context, domain, difficulty, concepts, relevant_sentences):
        """Generate a true/false question"""
        # Get templates
        templates = self.templates["True/False"]["DEFAULT"]
        
        # Choose concepts and template
        main_concept = random.choice(concepts[:3])
        other_concept = random.choice([c for c in concepts if c != main_concept] or ["alternative approach"])
        template = random.choice(templates)
        
        # Determine if the statement will be true or false
        is_true = random.choice([True, False])
        
        # Prepare purpose and benefit based on context and domain
        if domain == "Software Engineering":
            purposes = ["organizing code", "improving maintainability", "solving design problems"]
            benefits = ["code reusability", "easier maintenance", "better organization"]
            alternatives = ["conventional approaches", "ad-hoc solutions", other_concept]
        elif domain == "Machine Learning":
            purposes = ["improving model accuracy", "processing data efficiently", "feature extraction"]
            benefits = ["better predictions", "reduced overfitting", "faster training"]
            alternatives = ["traditional algorithms", "manual methods", other_concept]
        else:
            purposes = ["solving domain problems", "improving efficiency", "standardizing approaches"]
            benefits = ["better results", "simpler implementation", "increased productivity"]
            alternatives = ["other techniques", "conventional methods", other_concept]
            
        # Select values for the statement
        purpose = random.choice(purposes)
        benefit = random.choice(benefits)
        alternative = random.choice(alternatives)
        
        # Generate the statement
        statement = template.format(
            concept=main_concept.title(),
            purpose=purpose,
            benefit=benefit,
            related_concept=other_concept.title(),
            alternative=alternative.title()
        )
        
        # For false statements, modify something to make it incorrect
        if not is_true:
            # Various ways to make the statement false
            modifications = [
                # Wrong purpose
                lambda s: s.replace(purpose, random.choice([p for p in purposes if p != purpose])),
                # Wrong benefit
                lambda s: s.replace(benefit, random.choice([b for b in benefits if b != benefit])),
                # Negate the statement
                lambda s: s.replace(f"is {benefit}", f"is not {benefit}").replace(f"is {purpose}", f"is not {purpose}")
            ]
            
            # Apply a random modification
            statement = random.choice(modifications)(statement)
            
        # Include an explanation based on the context when available
        explanation = ""
        if relevant_sentences:
            if is_true:
                explanation = f" {relevant_sentences[0]}"
            else:
                explanation = f" According to the context, {main_concept.title()} actually {random.choice(['is related to', 'is used for', 'helps with'])} {purpose}."
        else:
            if is_true:
                explanation = f" This is correct based on standard principles in {domain}."
            else:
                explanation = f" This contradicts standard principles in {domain}."
            
        # Format the true/false question
        result = f"True or False: {statement}\n\nAnswer: {'TRUE' if is_true else 'FALSE'}.{explanation}"
        
        return result
        
    def _generate_fill_blank(self, context, domain, difficulty, concepts, relevant_sentences):
        """Generate a fill-in-the-blank question"""
        # Get templates
        templates = self.templates["Fill in the Blank"]["DEFAULT"]
        
        # Choose concepts and template
        main_concept = random.choice(concepts[:3])
        other_concept = random.choice([c for c in concepts if c != main_concept] or ["alternative approach"])
        template = random.choice(templates)
        
        # Generate the question with a blank
        question = template.format(
            concept=main_concept.title(),
            domain=domain,
            alternative=other_concept.title()
        )
        
        # Determine the answer based on context and domain
        answer = self._extract_blank_filler(main_concept, relevant_sentences, domain)
        
        # Format the fill-in-the-blank question
        result = f"Q: {question}\n\nAnswer: {answer}"
        
        return result
        
    def _extract_blank_filler(self, concept, relevant_sentences, domain):
        """Extract appropriate text to fill in a blank based on the context"""
        if not relevant_sentences:
            # Fallback based on domain
            if domain == "Software Engineering":
                return "organize and maintain code effectively"
            elif domain == "Machine Learning":
                return "improve model accuracy and generalization"
            else:
                return f"solve common problems in {domain}"
                
        # Use the most relevant sentence to find an appropriate filler
        main_sentence = relevant_sentences[0]
        concept_lower = concept.lower()
        
        if concept_lower in main_sentence.lower():
            # Find what comes after the concept
            idx = main_sentence.lower().find(concept_lower) + len(concept_lower)
            after_concept = main_sentence[idx:].strip()
            
            # Look for action phrases
            verb_phrases = [
                "is used for", "helps with", "enables", "facilitates", 
                "improves", "solves", "addresses", "manages"
            ]
            
            for phrase in verb_phrases:
                if phrase in after_concept.lower():
                    idx = after_concept.lower().find(phrase) + len(phrase)
                    answer_part = after_concept[idx:].strip()
                    
                    # Truncate at next punctuation
                    for punct in ".,:;?!":
                        if punct in answer_part:
                            answer_part = answer_part.split(punct)[0].strip()
                            
                    if answer_part and len(answer_part) > 3:
                        return answer_part
                        
        # Try to extract a meaningful phrase from the sentence
        words = main_sentence.split()
        if len(words) > 5:
            # Extract a potentially meaningful chunk
            mid_point = len(words) // 2
            return " ".join(words[max(0, mid_point - 2):min(len(words), mid_point + 3)]).strip(".,;:?!")
            
        # Fallback based on domain
        if domain == "Software Engineering":
            return "organize and maintain code effectively"
        elif domain == "Machine Learning":
            return "improve model accuracy and generalization"
        else:
            return f"solve common problems in {domain}"
            
    def _find_sentences_with_concept(self, context, concept):
        """Find sentences in the context that contain the concept"""
        sentences = re.split(r'(?<=[.!?])\s+', context)
        concept_lower = concept.lower()
        
        relevant_sentences = []
        for sentence in sentences:
            if concept_lower in sentence.lower():
                relevant_sentences.append(sentence.strip())
                
        return relevant_sentences
        
    def create_question_prompt(self, context, domain, question_type, difficulty, topic=None):
        """
        Create an optimized prompt for generating technical questions
        
        Args:
            context: The technical content context
            domain: Knowledge domain (e.g., "Software Engineering")
            question_type: Type of question to generate
            difficulty: Difficulty level
            topic: Optional specific topic
            
        Returns:
            Formatted prompt string
        """
        prompt = f"Generate a {difficulty} {question_type} question "
        
        if topic:
            prompt += f"about {topic} "
            
        prompt += f"in the domain of {domain} based on the following context:\n\n{context}"
        
        return prompt