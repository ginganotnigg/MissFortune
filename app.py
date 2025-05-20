import streamlit as st
import random
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Technical Question Generator",
    page_icon="🧠",
    layout="wide"
)

# Simple sample prompts for different domains
def get_sample_prompts(domain):
    samples = {
        "Software Engineering": {
            "Design Patterns": """Design patterns are typical solutions to common problems in software design. Each pattern is like a blueprint that you can customize to solve a particular design problem in your code.

Types of Design Patterns:
1. Creational Patterns: These patterns provide various object creation mechanisms.
   - Singleton: Ensures a class has only one instance and provides a global point of access to it.
   - Factory Method: Creates objects without specifying the exact class to create.
   - Abstract Factory: Creates families of related objects without specifying their concrete classes.

2. Structural Patterns: These patterns explain how to assemble objects and classes into larger structures.
   - Adapter: Allows objects with incompatible interfaces to collaborate.
   - Bridge: Separates an abstraction from its implementation so that the two can vary independently.
   - Composite: Composes objects into tree structures to represent part-whole hierarchies.

3. Behavioral Patterns: These patterns are concerned with algorithms and the assignment of responsibilities.
   - Observer: Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified.
   - Strategy: Defines a family of algorithms, encapsulates each one, and makes them interchangeable.
   - Command: Turns a request into a stand-alone object that contains all information about the request.""",
            
            "SOLID Principles": """SOLID is an acronym for five design principles in object-oriented programming:

1. Single Responsibility Principle (SRP):
   - A class should have only one reason to change, meaning it should have only one responsibility.
   - When a class handles multiple responsibilities, it becomes coupled, making it more difficult to maintain.

2. Open/Closed Principle (OCP):
   - Software entities should be open for extension but closed for modification.
   - This means you should be able to add new functionality without changing existing code.
   - Using interfaces and abstract classes helps follow this principle."""
        },
        "Machine Learning": {
            "Neural Networks": """Neural networks are computing systems inspired by the biological neural networks in animal brains. They consist of artificial neurons organized in layers, capable of learning patterns from data.

Key Components of Neural Networks:
1. Neurons (or Nodes): Basic computational units that take inputs, apply an activation function, and produce an output.
2. Layers: Input Layer, Hidden Layers, and Output Layer.
3. Weights and Biases: Parameters that determine the strength of connections between neurons.""",
            
            "Supervised Learning": """Supervised Learning is where models learn from labeled training data to predict outputs for unseen data.

Key Concepts in Supervised Learning:
1. Training Data: Consists of input features (X) and target variables (Y).
2. Model Training: The algorithm learns patterns from training data by minimizing a loss function.
3. Model Evaluation: Assessing performance on unseen data using metrics like accuracy and precision."""
        }
    }
    
    return samples.get(domain, {"Basic Concepts": "Enter your technical content here to generate questions."})

class QuestionGenerator:
    """Advanced question generator using context analysis and template-based generation"""
    
    def __init__(self):
        # Sample questions and answers dataset to learn from
        self.sample_qa_dataset = [
            {
                "context": "Design patterns are typical solutions to common problems in software design. Each pattern is like a blueprint that you can customize to solve a particular design problem in your code.",
                "question": "What is the primary purpose of design patterns in software engineering?",
                "answer": "To provide typical solutions to common problems in software design",
                "type": "Multiple Choice"
            },
            {
                "context": "The Observer pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.",
                "question": "In the Observer pattern, what happens when the observed object changes state?",
                "answer": "All dependent objects are notified and updated automatically",
                "type": "Multiple Choice"
            },
            {
                "context": "The Single Responsibility Principle (SRP) states that a class should have only one reason to change, meaning it should have only one responsibility.",
                "question": "What does the Single Responsibility Principle dictate about a class?",
                "answer": "It should have only one responsibility",
                "type": "Multiple Choice"
            },
            {
                "context": "Neural networks consist of artificial neurons organized in layers, capable of learning patterns from data.",
                "question": "How are artificial neurons organized in neural networks?",
                "answer": "In layers",
                "type": "Short Answer"
            },
            {
                "context": "Supervised Learning is where models learn from labeled training data to predict outputs for unseen data.",
                "question": "True or False: In supervised learning, models learn from unlabeled data.",
                "answer": "FALSE. Supervised learning uses labeled training data.",
                "type": "True/False"
            }
        ]
        
        # Advanced templates for different question types
        self.templates = {
            "Multiple Choice": [
                "What is the main purpose of {concept} in {domain}?",
                "Which of the following best describes {concept} as mentioned in the text?",
                "According to the context, what is a key characteristic of {concept}?",
                "Based on the provided information, what role does {concept} play in {domain}?",
                "Which statement about {concept} is most accurate according to the context?"
            ],
            "Short Answer": [
                "Based on the context, explain the concept of {concept} and its importance in {domain}.",
                "According to the information provided, how does {concept} work in practice?",
                "From the text, what are the key benefits of using {concept} in {domain}?",
                "As described in the context, what is the relationship between {concept} and {related_concept}?",
                "Using information from the context, describe how {concept} is implemented in {domain}."
            ],
            "True/False": [
                "Based on the given context, {concept} is primarily used for {purpose}.",
                "According to the information provided, the main benefit of {concept} is {benefit}.",
                "The text suggests that {concept} and {related_concept} serve the same purpose.",
                "Based on the context, {concept} is considered more effective than {alternative} for {purpose}.",
                "According to the provided information, {concept} is a fundamental component of {domain}."
            ],
            "Fill in the Blank": [
                "According to the context, {concept} is a technique used to _______ in {domain}.",
                "Based on the information provided, the primary purpose of {concept} is to _______ within {domain}.",
                "The text states that in {domain}, _______ is a key property of {concept}.",
                "As described in the context, {concept} differs from {alternative} mainly in its ability to _______.",
                "According to the provided information, {concept} helps developers to _______ when building {domain} systems."
            ]
        }
        
    def extract_key_phrases(self, context, n=8):
        """
        Extract key phrases from context that are likely to be important concepts
        This improved version looks for noun phrases and technical terminology
        """
        # Remove common words and normalize text
        common_words = set([
            "the", "and", "or", "a", "an", "in", "on", "at", "to", "for", "with", "by", 
            "about", "as", "is", "are", "was", "were", "be", "been", "being", "this", 
            "that", "these", "those", "it", "they", "them", "their", "his", "her", "its"
        ])
        
        # Clean up text and convert to lowercase for processing
        clean_text = context.lower()
        
        # Look for technical terms - longer phrases first then single words
        # First, try to find multi-word technical phrases (2-3 words)
        # Regex pattern for potentially meaningful phrases
        phrases_pattern = r'\b([A-Za-z][a-z]+(?:\s+[a-z]+){1,2})\b'
        multi_word_phrases = re.findall(phrases_pattern, context)
        
        # Second, look for "X of Y" or "X and Y" patterns which often indicate important concepts
        relation_phrases = re.findall(r'\b([A-Za-z][a-z]+\s+(?:of|and|in|for|to)\s+[a-z]+(?:\s+[a-z]+)?)\b', context)
        
        # Third, find standalone technical terms that start with capital letters or are in quotes
        technical_terms = re.findall(r'\b([A-Z][a-z]+)\b', context)  # Terms starting with capital letter
        quoted_terms = re.findall(r'[\'"]([^\'\"]+)[\'"]', context)  # Terms in quotes
        
        # Fourth, extract any words that appear after bullets or numbered items
        bullet_points = re.findall(r'(?:[-•*]\s+|\d+\.\s+)([A-Za-z][a-z]+(?:\s+[a-z]+){0,3})', context)
        
        # Fifth, fall back to single words that might be technical terms (longer words are more likely to be important)
        standalone_words = [w.lower() for w in re.findall(r'\b([A-Za-z][a-z]{5,})\b', context)]
        standalone_words = [w for w in standalone_words if w not in common_words]
        
        # Count all discovered potential concepts with different weights
        concept_scores = {}
        
        # Give highest weight to multi-word phrases as they're most likely to be meaningful concepts
        for phrase in multi_word_phrases:
            phrase = phrase.lower()
            concept_scores[phrase] = concept_scores.get(phrase, 0) + 5
            
        # Relationship phrases are also valuable
        for phrase in relation_phrases:
            phrase = phrase.lower()
            concept_scores[phrase] = concept_scores.get(phrase, 0) + 4
            
        # Technical terms by capitalization
        for term in technical_terms:
            term = term.lower()
            concept_scores[term] = concept_scores.get(term, 0) + 3
            
        # Quoted terms are explicitly highlighted
        for term in quoted_terms:
            term = term.lower()
            concept_scores[term] = concept_scores.get(term, 0) + 3
            
        # Bullet points often indicate key concepts
        for point in bullet_points:
            point = point.lower()
            concept_scores[point] = concept_scores.get(point, 0) + 3
            
        # Standalone words get lower weight
        for word in standalone_words:
            concept_scores[word] = concept_scores.get(word, 0) + 1
            
        # Filter out common words again at the end
        for word in list(concept_scores.keys()):
            if word in common_words or len(word) < 3:
                del concept_scores[word]
                
        # Get the most relevant concepts based on our scoring
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, score in sorted_concepts[:n]]
        
    def extract_context_sentences(self, context, concept, max_sentences=3):
        """
        Extract relevant sentences from context that mention the concept
        This helps generate more contextually relevant questions
        """
        # Split text into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Find sentences that contain the concept
        relevant_sentences = []
        concept_lower = concept.lower()
        
        for sentence in sentences:
            if concept_lower in sentence.lower():
                relevant_sentences.append(sentence.strip())
                if len(relevant_sentences) >= max_sentences:
                    break
                    
        return relevant_sentences if relevant_sentences else [random.choice(sentences).strip()]
        
    def find_related_concepts(self, context, main_concept, concepts, n=2):
        """Find concepts that are mentioned near the main concept in the context"""
        if len(concepts) <= 1:
            return ["related concepts"]
            
        main_concept_lower = main_concept.lower()
        related = []
        
        # Extract a window of text around main concept occurrences
        matches = list(re.finditer(r'\b' + re.escape(main_concept_lower) + r'\b', context.lower()))
        
        if not matches:
            # If exact match not found, find approximate position
            for i, concept in enumerate(concepts):
                if concept != main_concept and concept not in related:
                    related.append(concept)
                    if len(related) >= n:
                        break
            return related
            
        # Look for other concepts that appear close to the main concept
        for match in matches:
            start = max(0, match.start() - 100)
            end = min(len(context), match.end() + 100)
            window = context[start:end].lower()
            
            for concept in concepts:
                if concept != main_concept and concept.lower() in window and concept not in related:
                    related.append(concept)
                    if len(related) >= n:
                        break
            
            if len(related) >= n:
                break
                
        # If we couldn't find enough related concepts, add any other available concepts
        if len(related) < n:
            for concept in concepts:
                if concept != main_concept and concept not in related:
                    related.append(concept)
                    if len(related) >= n:
                        break
                    
        return related
        
    def generate(self, context, domain, question_type, num_questions, difficulty, **kwargs):
        """Generate questions based on context and parameters"""
        if not context or len(context) < 20:
            return ["Please provide more context to generate meaningful questions."]
            
        # Extract key concepts using our improved context analysis
        concepts = self.extract_key_phrases(context)
        if not concepts:
            concepts = ["technical concept", domain.lower(), "the topic"]
            
        # Get templates for requested question type
        templates = self.templates.get(question_type, self.templates["Multiple Choice"])
        
        # Generate questions
        questions = []
        used_concepts = set()  # Track used concepts to ensure variety
        
        for _ in range(num_questions):
            # Prioritize unused concepts for variety
            available_concepts = [c for c in concepts if c not in used_concepts]
            if not available_concepts:
                available_concepts = concepts  # If all used, reset
                
            # Select a concept and template
            concept = random.choice(available_concepts)
            used_concepts.add(concept)
            template = random.choice(templates)
            
            # Find related concepts for this question
            related_concepts = self.find_related_concepts(context, concept, concepts)
            
            # Extract relevant sentences about this concept to inform answer generation
            relevant_sentences = self.extract_context_sentences(context, concept)
            
            # Generate appropriate question type
            try:
                if question_type == "Multiple Choice":
                    question = self._generate_mcq(template, concept, domain, concepts, relevant_sentences, difficulty)
                elif question_type == "Short Answer":
                    question = self._generate_short_answer(template, concept, domain, related_concepts, relevant_sentences, difficulty)
                elif question_type == "True/False":
                    question = self._generate_true_false(template, concept, domain, related_concepts, relevant_sentences, difficulty)
                elif question_type == "Fill in the Blank":
                    question = self._generate_fill_blank(template, concept, domain, relevant_sentences, difficulty)
                else:
                    question = self._generate_mcq(template, concept, domain, concepts, relevant_sentences, difficulty)
            except Exception as e:
                logger.error(f"Error generating question: {str(e)}")
                # Fallback to a simple question if there's an error
                question = f"Q: What is the purpose of {concept} in {domain}?\n\nAnswer: Please refer to the context for details about {concept}."
                
            questions.append(question)
            
        return questions
        
    def _generate_mcq(self, template, concept, domain, concepts, relevant_sentences, difficulty):
        """Generate a multiple-choice question based on relevant sentences from context"""
        # Format the question using the template
        question_text = template.format(concept=concept.title(), domain=domain)
        
        # Generate the correct answer based on actual content
        correct_answer = self._extract_answer_from_context(concept, relevant_sentences, domain, difficulty)
        
        # Create wrong but plausible answers
        wrong_answers = self._generate_wrong_answers_from_context(concept, domain, concepts, relevant_sentences, 3, difficulty)
        
        # Combine and shuffle options
        options = [correct_answer] + wrong_answers
        random.shuffle(options)
        
        # Determine which option is correct
        correct_index = options.index(correct_answer)
        correct_letter = "ABCD"[correct_index]
        
        # Format final question
        final_question = f"Q: {question_text}\n\n"
        for i, option in enumerate(options):
            final_question += f"{chr(65 + i)}. {option}\n"
        final_question += f"\nAnswer: {correct_letter}"
        
        return final_question
        
    def _extract_answer_from_context(self, concept, relevant_sentences, domain, difficulty):
        """Extract the correct answer for MCQs directly from the context"""
        if not relevant_sentences:
            # Fallback if no relevant sentences found
            return f"A key concept in {domain} used for solving specific problems"
            
        # Choose the most relevant sentence
        main_sentence = relevant_sentences[0]
        
        # Extract a meaningful definition or description from the sentence
        # Look for patterns like "X is Y" or "X does Y"
        concept_lower = concept.lower()
        sentence_lower = main_sentence.lower()
        
        # Try to find a definition part after the concept
        if concept_lower in sentence_lower:
            start_idx = sentence_lower.find(concept_lower) + len(concept_lower)
            definition_part = main_sentence[start_idx:].strip()
            
            # Look for verbs or connecting words to start the definition
            connection_words = ["is", "are", "refers to", "means", "represents", "provides", "helps", "allows", "enables"]
            
            for word in connection_words:
                if word in definition_part.lower():
                    definition_idx = definition_part.lower().find(word)
                    definition = definition_part[definition_idx:].strip()
                    # Clean up the definition
                    definition = definition.strip(".,;:()[] ")
                    if definition and len(definition) > 10:
                        return definition.capitalize()
        
        # If no clear definition found, create a synthesized answer from the sentence
        words = main_sentence.split()
        if len(words) > 8:
            # Take a reasonable chunk of the sentence that's likely to contain useful information
            snippet = " ".join(words[max(0, min(len(words)//2, 6)):min(len(words), 14)])
            return f"{snippet.capitalize()}"
        
        return main_sentence.capitalize()
        
    def _generate_short_answer(self, template, concept, domain, related_concepts, relevant_sentences, difficulty):
        """Generate a short-answer question based on context"""
        # Format the question
        question_text = template.format(
            concept=concept.title(), 
            domain=domain,
            related_concept=related_concepts[0].title() if related_concepts else "implementation"
        )
        
        # Generate an appropriate answer based on the context and difficulty
        if relevant_sentences:
            # Combine relevant sentences into a coherent answer
            if difficulty == "Easy":
                # Simple answer using the first relevant sentence
                answer = relevant_sentences[0].strip()
            elif difficulty == "Medium":
                # Combine 1-2 sentences for a more complete answer
                answer = " ".join(relevant_sentences[:min(2, len(relevant_sentences))])
            else:  # Hard or Expert
                # More comprehensive answer using all relevant sentences
                answer = " ".join(relevant_sentences)
                
                # Add synthesis for expert level
                if difficulty == "Expert" and len(relevant_sentences) > 1:
                    answer += f" This makes {concept.title()} particularly valuable in {domain} contexts where complexity needs to be managed effectively."
        else:
            # Fallback if no relevant sentences found
            if difficulty == "Easy":
                answer = f"{concept.title()} is a fundamental concept in {domain} that helps with organizing and structuring code."
            elif difficulty == "Medium":
                answer = f"{concept.title()} provides a structured approach to solving common problems in {domain} by offering standardized solutions that can be adapted to specific contexts."
            elif difficulty == "Hard":
                answer = f"{concept.title()} offers several advantages including improved code organization, better maintainability, and enhanced scalability, particularly in complex {domain} projects."
            else:  # Expert
                answer = f"{concept.title()} represents an advanced approach in {domain} that balances theoretical underpinnings with practical implementations, allowing for optimized trade-offs between competing factors like performance, maintainability, and flexibility."
            
        return f"Q: {question_text}\n\nAnswer: {answer}"
        
    def _generate_true_false(self, template, concept, domain, related_concepts, relevant_sentences, difficulty):
        """Generate a true/false question based on context"""
        # Determine if we'll create a true or false statement
        is_true = random.choice([True, True, False])  # Slightly bias toward true for better quality
        
        # Get related concept to work with
        related_concept = related_concepts[0] if related_concepts else "alternative approaches"
        
        # Define purposes and benefits based on domain and difficulty
        if domain == "Software Engineering":
            purposes = ["organizing code", "solving design problems", "improving maintainability"]
            benefits = ["code reusability", "easier maintenance", "better organization"]
        elif domain == "Machine Learning":
            purposes = ["pattern recognition", "predictive modeling", "data analysis"]
            benefits = ["improved accuracy", "faster training", "better generalization"]
        else:
            purposes = ["solving domain problems", "improving efficiency", "standardizing approaches"]
            benefits = ["better organization", "improved productivity", "reduced complexity"]
        
        # Select appropriate purpose and benefit
        selected_purpose = random.choice(purposes)
        selected_benefit = random.choice(benefits)
        
        # Create statement based on template and whether it should be true or false
        if relevant_sentences:
            # Use the content of relevant sentences to create a more accurate statement
            base_sentence = relevant_sentences[0]
            
            if "is" in base_sentence and concept.lower() in base_sentence.lower():
                # Try to extract the relationship between concept and its definition
                parts = base_sentence.lower().split(concept.lower(), 1)
                if len(parts) > 1 and len(parts[1]) > 5:
                    after_concept = parts[1].strip()
                    
                    if is_true:
                        statement = f"{concept.title()}{after_concept}"
                    else:
                        # Create a false statement by negating or replacing key parts
                        negations = ["not", "rarely", "seldom", "never"]
                        if "is" in after_concept:
                            statement = f"{concept.title()} {after_concept.replace('is', 'is ' + random.choice(negations))}"
                        else:
                            statement = f"{concept.title()} is {random.choice(negations)} used for{after_concept}"
                else:
                    statement = template.format(
                        concept=concept.title(),
                        domain=domain,
                        purpose=selected_purpose,
                        benefit=selected_benefit,
                        related_concept=related_concept.title(),
                        alternative=related_concept.title()
                    )
            else:
                # If we can't easily derive a statement, use the template
                statement = template.format(
                    concept=concept.title(),
                    domain=domain,
                    purpose=selected_purpose if is_true else f"exclusively {random.choice(purposes)}", 
                    benefit=selected_benefit if is_true else f"solely {random.choice(benefits)}",
                    related_concept=related_concept.title(),
                    alternative=related_concept.title()
                )
        else:
            # Use the template if we don't have relevant sentences
            statement = template.format(
                concept=concept.title(),
                domain=domain,
                purpose=selected_purpose if is_true else f"exclusively {random.choice(purposes)}", 
                benefit=selected_benefit if is_true else f"solely {random.choice(benefits)}",
                related_concept=related_concept.title(),
                alternative=related_concept.title()
            )
        
        # Add explanation based on the truth value
        answer = "TRUE" if is_true else "FALSE"
        
        if relevant_sentences and len(relevant_sentences) > 1:
            explanation = relevant_sentences[1] if is_true else f"According to the context, {concept.title()} actually {random.choice(['relates to', 'is used for', 'helps with'])} {selected_purpose}."
        else:
            explanation = f"Based on the provided information, {concept.title()} " + (
                f"is indeed associated with {selected_purpose} in {domain}." if is_true else
                f"serves a different purpose than stated."
            )
        
        return f"True or False: {statement}\n\nAnswer: {answer}. {explanation}"
        
    def _generate_fill_blank(self, template, concept, domain, relevant_sentences, difficulty):
        """Generate a fill-in-the-blank question based on context"""
        # Identify possible blank completions from the context
        if relevant_sentences:
            main_sentence = relevant_sentences[0]
            
            # Look for verbs or actions associated with the concept
            concept_lower = concept.lower()
            if concept_lower in main_sentence.lower():
                idx = main_sentence.lower().find(concept_lower)
                
                # Check what comes after the concept name
                after_concept = main_sentence[idx + len(concept_lower):].strip()
                
                # Try to find a key phrase to use as the answer
                phrases = [
                    "used for", "helps with", "enables", "facilitates", 
                    "improves", "solves", "addresses", "manages",
                    "organizes", "creates", "defines", "implements"
                ]
                
                for phrase in phrases:
                    if phrase in after_concept.lower():
                        phrase_idx = after_concept.lower().find(phrase)
                        # Find what comes after the phrase
                        start = phrase_idx + len(phrase)
                        end = after_concept.find(".") if "." in after_concept[start:] else len(after_concept)
                        target = after_concept[start:end].strip()
                        
                        if target and len(target) > 3:
                            break
                else:
                    # If no specific phrase found, look for a noun phrase
                    words = after_concept.split()
                    if len(words) >= 3:
                        target = " ".join(words[:3]).strip(".,;:() ")
                    else:
                        target = after_concept.strip(".,;:() ")
            else:
                # If concept isn't directly in the sentence, look for key actions
                words = main_sentence.split()
                verbs = ["used", "helps", "enables", "improves", "creates", "defines"]
                
                for i, word in enumerate(words):
                    if word.lower() in verbs and i < len(words) - 2:
                        target = " ".join(words[i+1:i+4]).strip(".,;:() ")
                        break
                else:
                    # Default to a reasonable chunk of the sentence
                    if len(words) > 5:
                        target = " ".join(words[2:5]).strip(".,;:() ")
                    else:
                        target = "improving solution design"
        else:
            # Fallback options if no relevant context
            possible_targets = {
                "Software Engineering": ["improving code organization", "reducing complexity", "solving design problems"],
                "Machine Learning": ["recognizing patterns", "making predictions", "analyzing data"],
                "Web Development": ["creating interactive interfaces", "managing user data", "optimizing performance"],
                "Databases": ["storing data efficiently", "retrieving information", "maintaining data integrity"]
            }
            
            domain_targets = possible_targets.get(domain, ["solving technical problems"])
            target = random.choice(domain_targets)
        
        # Create the question with the blank
        # Use format with only the parameters that are actually in the template
        # to avoid KeyError with alternative parameter
        if "{alternative}" in template:
            question_text = template.format(
                concept=concept.title(),
                domain=domain,
                alternative="other approaches"
            )
        else:
            question_text = template.format(
                concept=concept.title(),
                domain=domain
            )
        
        # Replace the blank placeholder with actual underscores
        question_text = question_text.replace("_______", "______")
        
        return f"Q: {question_text}\n\nAnswer: {target}"
        
    def _generate_wrong_answers_from_context(self, concept, domain, concepts, relevant_sentences, count, difficulty):
        """Generate wrong but plausible answers for multiple-choice questions using context"""
        # Generate a set of wrong answers that are related to the domain but incorrect for the concept
        wrong_answers = []
        
        # Use other concepts from the context as the basis for wrong answers
        other_concepts = [c for c in concepts if c.lower() != concept.lower()][:count+2]
        
        # Create wrong answers from other concepts
        if other_concepts and len(other_concepts) >= count:
            # If we have enough other concepts, create answers based on them
            for i in range(min(count, len(other_concepts))):
                other = other_concepts[i]
                
                # Create a plausible but incorrect answer
                if difficulty == "Easy":
                    # Simple wrong answers for easy difficulty
                    wrong_answers.append(f"A concept related to {other} but not to {concept}")
                elif difficulty == "Medium":
                    # More sophisticated wrong answers for medium difficulty
                    wrong_answers.append(f"A technique for working with {other} in {domain} projects")
                else:  # Hard or Expert
                    # More nuanced and plausible wrong answers for higher difficulties
                    wrong_answers.append(f"An approach that combines elements of {concept} and {other} to solve different problems in {domain}")
        
        # If we need more wrong answers, use generic templates
        generic_wrong_templates = [
            f"A tool for debugging {domain} issues unrelated to {concept}",
            f"An outdated approach replaced by {concept} in modern {domain}",
            f"A theoretical concept with limited practical applications in {domain}",
            f"A specialized technique only used in advanced {domain} scenarios",
            f"A common misunderstanding of how {concept} actually works in {domain}"
        ]
        
        # Add generic wrong answers if needed
        while len(wrong_answers) < count:
            template = random.choice(generic_wrong_templates)
            if template not in wrong_answers:
                wrong_answers.append(template)
        
        return wrong_answers[:count]
        
class QuestionEvaluator:
    """Simple evaluator for generated questions"""
    
    def __init__(self):
        self.technical_terms = {
            "Software Engineering": [
                "design pattern", "architecture", "agile", "scrum", "waterfall", "testing",
                "requirements", "deployment", "inheritance", "polymorphism", "encapsulation"
            ],
            "Machine Learning": [
                "neural network", "supervised", "unsupervised", "deep learning", "backpropagation",
                "gradient descent", "overfitting", "underfitting", "regression", "classification"
            ]
        }
        
    def evaluate_relevance(self, question, context, threshold=0.3):
        """Evaluate relevance between question and context"""
        # Simple word overlap approach
        question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
        context_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', context.lower()))
        
        # Remove common words
        common_words = set(["the", "and", "or", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about", "as"])
        question_words = question_words - common_words
        context_words = context_words - common_words
        
        if not question_words or not context_words:
            return 0.5  # Default mid-relevance
            
        # Calculate Jaccard similarity 
        intersection = question_words.intersection(context_words)
        union = question_words.union(context_words)
        
        similarity = len(intersection) / len(union)
        
        # Check for phrases to boost relevance
        phrases_boost = 0
        for i in range(min(len(intersection), 3)):
            # Look for 2-3 word phrases from intersection
            phrase_pattern = ' '.join(random.sample(list(intersection), 2))
            if phrase_pattern in context.lower() and phrase_pattern in question.lower():
                phrases_boost += 0.1
                
        final_score = min(1.0, similarity + phrases_boost)
        return max(final_score, 0.1)  # Ensure minimum score
        
    def evaluate_complexity(self, questions):
        """Evaluate complexity of questions"""
        if not questions:
            return 0.5
            
        scores = []
        for question in questions:
            # Simple heuristics for complexity
            length_score = min(1.0, len(question) / 500)  # Longer questions trend complex
            
            # Check for technical terms
            tech_term_count = 0
            for domain, terms in self.technical_terms.items():
                for term in terms:
                    if term in question.lower():
                        tech_term_count += 1
            tech_score = min(1.0, tech_term_count / 5)
            
            # Check for complex structures
            structure_score = 0
            if "A." in question and "B." in question and "C." in question:  # MCQ
                structure_score += 0.3
            if "Answer:" in question and len(question.split("Answer:")[1]) > 100:  # Detailed answer
                structure_score += 0.4
                
            final_score = (length_score + tech_score + structure_score) / 3
            scores.append(final_score)
            
        return sum(scores) / len(scores)
        
    def evaluate_diversity(self, questions):
        """Evaluate diversity among questions"""
        if not questions or len(questions) < 2:
            return 1.0
            
        # Simple signature approach 
        signatures = []
        for question in questions:
            # Create a signature from first 10 words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())[:10]
            signatures.append(' '.join(words))
            
        # Calculate pairwise similarities
        similarity_sum = 0
        count = 0
        
        for i in range(len(signatures)):
            for j in range(i+1, len(signatures)):
                words1 = set(signatures[i].split())
                words2 = set(signatures[j].split())
                
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
        diversity = 1.0 - avg_similarity
        
        return diversity

def main():
    st.title("🧠 Technical Question Generator")
    st.markdown("""
    This application uses prompt tuning techniques to generate high-quality 
    technical questions for educational assessments.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Configuration")
        
        domain = st.selectbox(
            "Knowledge Domain",
            ["Software Engineering", "Machine Learning", "Web Development", "Databases"]
        )
        
        question_type = st.selectbox(
            "Question Type",
            ["Multiple Choice", "Short Answer", "True/False", "Fill in the Blank"]
        )
        
        num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=3)
        
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard", "Expert"]
        )
        
        with st.expander("Advanced Options"):
            temperature = st.slider("Creativity Level", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                                 help="Higher values produce more creative but potentially less focused questions")
            max_length = st.slider("Max Length", min_value=100, max_value=1000, value=500, step=50,
                                help="Maximum length of the generated questions")
            
    with col1:
        st.subheader("Context & Topic")
        
        sample_contexts = get_sample_prompts(domain)
        use_sample = st.checkbox("Use Sample Context", value=False)
        
        if use_sample and sample_contexts:
            sample_keys = list(sample_contexts.keys())
            if sample_keys:
                selected_sample = st.selectbox("Select Sample", sample_keys)
                context = st.text_area("Context (Technical content to generate questions about)", 
                                    sample_contexts[selected_sample], height=200)
            else:
                context = st.text_area("Context (Technical content to generate questions about)", 
                                    "", height=200, 
                                    placeholder="Enter the technical content you want to generate questions about...")
        else:
            context = st.text_area("Context (Technical content to generate questions about)", 
                                 "", height=200, 
                                 placeholder="Enter the technical content you want to generate questions about...")
        
        topic = st.text_input("Specific Topic (Optional)", 
                            placeholder="Optional: narrow down to a specific topic within the context")
        
        generate_button = st.button("Generate Questions")
        
        if generate_button:
            if not context:
                st.error("Please provide context information to generate questions.")
            else:
                try:
                    with st.spinner("Generating high-quality technical questions..."):
                        # Initialize generator and evaluator
                        generator = QuestionGenerator()
                        evaluator = QuestionEvaluator()
                        
                        # Generate questions
                        generation_params = {
                            "domain": domain,
                            "question_type": question_type,
                            "num_questions": num_questions,
                            "difficulty": difficulty,
                            "temperature": temperature,
                            "max_length": max_length,
                            "topic": topic
                        }
                        
                        questions = generator.generate(context, **generation_params)
                        
                        # Evaluate questions
                        relevance_scores = []
                        for q in questions:
                            relevance = evaluator.evaluate_relevance(q, context)
                            relevance_scores.append(relevance)
                        
                        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
                        
                    st.success(f"Generated {len(questions)} questions successfully!")
                    
                    # Display generated questions
                    st.subheader("Generated Questions")
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Average Relevance", f"{avg_relevance:.2f}/1.0")
                    with metrics_col2:
                        st.metric("Complexity Score", f"{evaluator.evaluate_complexity(questions):.2f}/1.0")
                    with metrics_col3:
                        st.metric("Diversity Score", f"{evaluator.evaluate_diversity(questions):.2f}/1.0")
                    
                    for i, question in enumerate(questions):
                        with st.expander(f"Question {i+1} - Relevance: {relevance_scores[i]:.2f}", expanded=True):
                            st.markdown(question)
                            
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                if st.button(f"Regenerate Question {i+1}"):
                                    with st.spinner("Regenerating question..."):
                                        # Generate a single new question
                                        new_question = generator.generate(context, **{**generation_params, "num_questions": 1})[0]
                                        questions[i] = new_question
                                        st.rerun()
                            
                except Exception as e:
                    logger.error(f"Error generating questions: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()