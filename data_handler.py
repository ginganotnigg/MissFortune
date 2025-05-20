import os
import json
import logging
from typing import List, Dict, Optional, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles data preparation and management for prompt tuning and question generation
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the data handler
        
        Args:
            data_dir: Directory for storing data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Define paths for sample data files
        self.sample_files = {
            "Multiple Choice": os.path.join(data_dir, "mcq_samples.jsonl"),
            "Short Answer": os.path.join(data_dir, "short_answer_samples.jsonl"),
            "True/False": os.path.join(data_dir, "true_false_samples.jsonl"),
            "Fill in the Blank": os.path.join(data_dir, "fill_blank_samples.jsonl")
        }
        
    def prepare_training_data(self, raw_data: List[Dict[str, str]], 
                            output_file: str = "training_data.jsonl") -> str:
        """
        Prepare training data for prompt tuning
        
        Args:
            raw_data: List of dictionaries with prompt/completion pairs
            output_file: Name of the output JSONL file
            
        Returns:
            Path to the prepared training data file
        """
        output_path = os.path.join(self.data_dir, output_file)
        
        with open(output_path, 'w') as f:
            for item in raw_data:
                f.write(json.dumps(item) + '\n')
                
        return output_path
        
    def load_training_data(self, file_path: Optional[str] = None, 
                         question_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load training data from a JSONL file
        
        Args:
            file_path: Path to the JSONL file, or None to use question type to determine file
            question_type: Type of questions to load data for (Multiple Choice, Short Answer, etc.)
            
        Returns:
            List of dictionaries with question data
        """
        # If question type is provided, use the appropriate sample file
        if question_type and not file_path:
            file_path = self.sample_files.get(question_type)
            if not file_path:
                logger.warning(f"No sample file found for question type: {question_type}")
                return []
        
        # If no file path is specified and no question type, use the default
        if file_path is None:
            file_path = os.path.join(self.data_dir, "training_data.jsonl")
            
        if not os.path.exists(file_path):
            logger.warning(f"Training data file not found: {file_path}")
            return []
            
        data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return []
                    
        return data
        
    def load_sample_questions(self, question_type: Optional[str] = None, 
                            max_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Load sample questions from the appropriate files
        
        Args:
            question_type: Type of questions to load (or None to load all types)
            max_samples: Maximum number of samples to return per type
            
        Returns:
            List of sample question dictionaries
        """
        samples = []
        
        if question_type:
            # Load samples for a specific question type
            samples = self.load_training_data(question_type=question_type)
            # Limit the number of samples
            samples = samples[:max_samples]
        else:
            # Load samples for all question types
            for q_type in self.sample_files:
                type_samples = self.load_training_data(question_type=q_type)
                # Limit the number of samples per type
                samples.extend(type_samples[:max_samples])
                
        return samples
    
    def get_context_relevant_samples(self, context: str, question_type: str,
                                  max_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Find sample questions relevant to the given context
        
        Args:
            context: The context to find relevant questions for
            question_type: Type of questions to find
            max_samples: Maximum number of samples to return
            
        Returns:
            List of relevant sample question dictionaries
        """
        # Load samples for the specified question type
        all_samples = self.load_sample_questions(question_type)
        if not all_samples:
            return []
            
        # Extract keywords from the context
        context_lower = context.lower()
        keywords = set()
        
        # Simple keyword extraction (could be improved with NLP)
        for word in context_lower.split():
            if len(word) > 4 and word.isalpha():  # Simple filter for meaningful words
                keywords.add(word)
        
        # Score samples based on relevance to the context keywords
        scored_samples = []
        for sample in all_samples:
            sample_context = sample.get("context", "").lower()
            
            # Calculate a simple relevance score based on keyword matches
            score = 0
            for keyword in keywords:
                if keyword in sample_context:
                    score += 1
                    
            if score > 0:  # Only include samples with some relevance
                scored_samples.append((score, sample))
                
        # Sort by relevance score (highest first) and take top samples
        scored_samples.sort(reverse=True, key=lambda x: x[0])
        relevant_samples = [sample for score, sample in scored_samples[:max_samples]]
        
        return relevant_samples
        
    def generate_sample_training_data(self, num_samples: int = 10) -> List[Dict[str, str]]:
        """
        Generate sample training data for different question types
        
        Args:
            num_samples: Number of sample records to generate
            
        Returns:
            List of dictionaries with prompt/completion pairs
        """
        samples = []
        
        # Use our actual sample files to generate better training data
        all_question_samples = self.load_sample_questions(max_samples=num_samples)
        
        for sample in all_question_samples:
            context = sample.get("context", "")
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            q_type = sample.get("type", "Multiple Choice")
            
            # Format as prompt/completion pairs
            prompt = f"Generate a {q_type} question about the following context:\n\n{context}"
            
            # Format the completion based on question type
            if q_type == "Multiple Choice":
                options = sample.get("options", [])
                if options:
                    completion = f"Q: {question}\n"
                    for i, option in enumerate(options):
                        completion += f"{chr(65 + i)}. {option}\n"
                    completion += f"\nAnswer: {answer}"
                else:
                    completion = f"Q: {question}\nAnswer: {answer}"
            else:
                completion = f"Q: {question}\nAnswer: {answer}"
                
            samples.append({
                "prompt": prompt,
                "completion": completion
            })
            
        return samples
        
    def save_generated_questions(self, questions: List[str], 
                               file_name: str = "generated_questions.txt") -> str:
        """
        Save generated questions to a file
        
        Args:
            questions: List of generated questions
            file_name: Name of the output file
            
        Returns:
            Path to the saved file
        """
        output_path = os.path.join(self.data_dir, file_name)
        
        with open(output_path, 'w') as f:
            for question in questions:
                f.write(question + '\n\n')
                
        return output_path
        
    def export_to_csv(self, questions: List[Dict[str, Any]], 
                    file_name: str = "questions_export.csv") -> str:
        """
        Export questions to CSV format
        
        Args:
            questions: List of question dictionaries
            file_name: Name of the output CSV file
            
        Returns:
            Path to the saved CSV file
        """
        import csv
        
        output_path = os.path.join(self.data_dir, file_name)
        
        with open(output_path, 'w', newline='') as f:
            if not questions:
                f.write('No questions generated')
                return output_path
                
            # Extract fields from the first question
            fieldnames = list(questions[0].keys())
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for question in questions:
                writer.writerow(question)
                
        return output_path