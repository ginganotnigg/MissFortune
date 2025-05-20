import re
import random
from prompt_tuning_service import PromptTuningService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerator:
    """
    Class for generating technical questions using the PromptTuningService
    """
    
    def __init__(self, prompt_tuning_service=None):
        """
        Initialize the question generator
        
        Args:
            prompt_tuning_service: Instance of PromptTuningService or None to create a new one
        """
        self.service = prompt_tuning_service or PromptTuningService()
        
        # Maintain question history to avoid repetition
        self.question_history = set()
        
        # Templates for post-processing
        self.mcq_pattern = re.compile(r'Q:|Question:|MCQ:')
        self.answer_pattern = re.compile(r'Answer:|Correct Answer:|A:|Ans:')
    
    def generate(self, context, domain="Software Engineering", question_type="Multiple Choice", 
                num_questions=3, difficulty="Medium", temperature=0.7, max_length=500, 
                top_p=0.9, topic=None):
        """
        Generate multiple technical questions based on the given context
        
        Args:
            context: Technical content context
            domain: Knowledge domain
            question_type: Type of questions to generate
            num_questions: Number of questions to generate
            difficulty: Difficulty level
            temperature: Sampling temperature
            max_length: Maximum length of generated text
            top_p: Nucleus sampling parameter
            topic: Optional specific topic
            
        Returns:
            List of generated questions
        """
        questions = []
        
        # Generate questions in series to ensure diversity
        for i in range(num_questions):
            # Adjust temperature slightly for each question to increase diversity
            adjusted_temp = min(1.0, temperature + (i * 0.05))
            
            single_question = self.generate_single(
                context, 
                domain, 
                question_type, 
                difficulty, 
                adjusted_temp, 
                max_length, 
                top_p, 
                topic
            )
            
            questions.append(single_question)
            
        return questions
    
    def generate_single(self, context, domain="Software Engineering", question_type="Multiple Choice", 
                      difficulty="Medium", temperature=0.7, max_length=500, top_p=0.9, topic=None):
        """
        Generate a single technical question based on the given context
        
        Args:
            context: Technical content context
            domain: Knowledge domain
            question_type: Type of question to generate
            difficulty: Difficulty level
            temperature: Sampling temperature
            max_length: Maximum length of generated text
            top_p: Nucleus sampling parameter
            topic: Optional specific topic
            
        Returns:
            A generated question string
        """
        # Create optimized prompt
        prompt = self.service.create_question_prompt(
            context, domain, question_type, difficulty, topic
        )
        
        # Generate text
        generated_texts = self.service.generate_from_prompt(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1
        )
        
        if not generated_texts:
            raise ValueError("Failed to generate question")
            
        # Post-process the generated question
        question = self.post_process_question(generated_texts[0], question_type)
        
        # Check if question is too similar to history and regenerate if needed
        attempt = 0
        while self._is_similar_to_history(question) and attempt < 3:
            logger.info(f"Question too similar to history, regenerating (attempt {attempt+1})")
            generated_texts = self.service.generate_from_prompt(
                prompt,
                max_length=max_length,
                temperature=temperature + 0.1,  # Increase temperature to promote diversity
                top_p=top_p,
                num_return_sequences=1
            )
            question = self.post_process_question(generated_texts[0], question_type)
            attempt += 1
        
        # Add to history
        self.question_history.add(self._get_question_signature(question))
        
        return question
    
    def post_process_question(self, raw_text, question_type):
        """
        Post-process the generated question text to ensure correct formatting
        
        Args:
            raw_text: Raw generated text
            question_type: Type of question
            
        Returns:
            Formatted question string
        """
        # Remove any leading whitespace or newlines
        processed = raw_text.strip()
        
        # Remove any prefixes like "Q:" or "Question:" that the model might have generated
        processed = self.mcq_pattern.sub('', processed, 1).strip()
        
        # Format differently based on question type
        if question_type == "Multiple Choice":
            # Ensure options are properly formatted
            if 'A.' not in processed and 'A)' not in processed:
                # Try to format options if they're not properly formatted
                processed = self._format_mcq_options(processed)
            
            # Ensure there's an answer indication
            if not self.answer_pattern.search(processed):
                processed += "\n\nAnswer: [Not provided - review required]"
        
        elif question_type == "Short Answer":
            # Make sure there's a clear separation between question and answer
            if "Answer:" not in processed:
                parts = processed.split('\n\n', 1)
                if len(parts) > 1:
                    processed = f"{parts[0]}\n\nAnswer: {parts[1]}"
        
        # Add a prefix based on question type for clarity
        if question_type == "Multiple Choice":
            if not processed.startswith("Q:"):
                processed = f"Q: {processed}"
        elif question_type == "Short Answer":
            if not processed.startswith("Q:"):
                processed = f"Q: {processed}"
        elif question_type == "True/False":
            if not processed.startswith("True or False:"):
                processed = f"True or False: {processed}"
        elif question_type == "Fill in the Blank":
            if "_____" not in processed:
                processed = self._ensure_blank(processed)
        
        return processed
    
    def _format_mcq_options(self, text):
        """Format text to ensure proper MCQ options"""
        lines = text.split('\n')
        result = []
        
        # Keep the question part
        question_part = []
        option_part = []
        answer_part = []
        
        in_options = False
        in_answer = False
        
        for line in lines:
            if in_answer or self.answer_pattern.search(line):
                in_answer = True
                answer_part.append(line)
            elif in_options or re.match(r'^[A-E][\.\)]', line) or re.match(r'^[0-9][\.\)]', line):
                in_options = True
                # Format options consistently
                if re.match(r'^[A-E][\.\)]', line) or re.match(r'^[0-9][\.\)]', line):
                    # Extract the option letter/number and convert to A, B, C, D format
                    option_match = re.match(r'^([A-E0-9])[\.\)]', line)
                    if option_match:
                        option_num = option_match.group(1)
                        if option_num.isdigit():
                            option_letters = 'ABCDE'
                            if int(option_num) <= 5:
                                option_letter = option_letters[int(option_num)-1]
                                line = re.sub(r'^[0-9][\.\)]', f"{option_letter}.", line)
                option_part.append(line)
            else:
                question_part.append(line)
        
        # Combine parts
        result = question_part
        if option_part:
            result.extend(option_part)
        if answer_part:
            result.extend(answer_part)
        else:
            # If no answer part, try to infer from options
            result.append("\nAnswer: [Not explicitly provided]")
        
        return '\n'.join(result)
    
    def _ensure_blank(self, text):
        """Ensure fill-in-the-blank questions have blanks"""
        if "_____" not in text:
            # Try to identify sentence structure and insert a blank
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 1:
                # Choose a random sentence (not the first or last)
                if len(sentences) > 2:
                    sentence_idx = random.randint(1, len(sentences)-2)
                else:
                    sentence_idx = 0
                
                # Split the sentence into words
                words = sentences[sentence_idx].split()
                if len(words) > 3:
                    # Choose a random word to replace (not first or last)
                    word_idx = random.randint(1, len(words)-2)
                    blank_word = words[word_idx]
                    words[word_idx] = "_____"
                    sentences[sentence_idx] = ' '.join(words)
                    
                    # Reconstruct the text
                    filled_text = ' '.join(sentences)
                    return f"{filled_text}\n\nAnswer: {blank_word}"
            
            # If we couldn't properly insert a blank, mark it for review
            return f"{text}\n\n[This question needs to be reformatted as fill-in-the-blank]"
        return text
    
    def _get_question_signature(self, question):
        """Get a signature for the question to check for duplicates"""
        # Remove formatting, punctuation, and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', '', question.lower())
        words = cleaned.split()
        
        # Use first 10 words as signature if available
        if len(words) > 10:
            return ' '.join(words[:10])
        return ' '.join(words)
    
    def _is_similar_to_history(self, question):
        """Check if the question is too similar to previously generated ones"""
        signature = self._get_question_signature(question)
        
        # Check against history
        for hist_sig in self.question_history:
            # If more than 70% similar, consider it too similar
            similarity = self._calculate_similarity(signature, hist_sig)
            if similarity > 0.7:
                return True
        
        return False
    
    def _calculate_similarity(self, text1, text2):
        """Calculate the similarity between two text signatures"""
        # Simple word overlap ratio
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
