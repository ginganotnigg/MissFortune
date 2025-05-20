def get_sample_prompts(domain="Software Engineering"):
    """
    Get sample context prompts for different domains
    
    Args:
        domain: Knowledge domain
        
    Returns:
        Dictionary of sample contexts for the specified domain, or default samples if domain not found
    """
    # Default samples for any domain not explicitly defined
    default_samples = {
        "Basic Concepts": """This is a sample context for generating questions about basic concepts in the field.
You can enter your own technical content here to generate custom questions.
The system will analyze the content and create relevant technical questions based on it.""",
        
        "Advanced Topics": """This sample context can be used to create more advanced questions.
Replace this text with your own technical content related to advanced topics in your field.
The question generator will analyze the content and create appropriate questions."""
    }
    
    # Domain-specific samples
    domain_samples = {
        "Software Engineering": {
            "Design Patterns": """Design patterns are typical solutions to common problems in software design. Each pattern is like a blueprint that you can customize to solve a particular design problem in your code.

Types of Design Patterns:
1. Creational Patterns: These patterns provide various object creation mechanisms, which increase flexibility and reuse of existing code.
   - Singleton: Ensures a class has only one instance and provides a global point of access to it.
   - Factory Method: Creates objects without specifying the exact class to create.
   - Abstract Factory: Creates families of related objects without specifying their concrete classes.
   - Builder: Constructs complex objects step by step.
   - Prototype: Creates new objects by copying existing objects.

2. Structural Patterns: These patterns explain how to assemble objects and classes into larger structures while keeping these structures flexible and efficient.
   - Adapter: Allows objects with incompatible interfaces to collaborate.
   - Bridge: Separates an abstraction from its implementation so that the two can vary independently.
   - Composite: Composes objects into tree structures to represent part-whole hierarchies.
   - Decorator: Attaches additional responsibilities to an object dynamically.
   - Facade: Provides a simplified interface to a complex subsystem.
   - Flyweight: Reduces the cost of creating and manipulating a large number of similar objects.
   - Proxy: Provides a substitute or placeholder for another object.

3. Behavioral Patterns: These patterns are concerned with algorithms and the assignment of responsibilities between objects.
   - Chain of Responsibility: Passes a request along a chain of handlers.
   - Command: Turns a request into a stand-alone object that contains all information about the request.
   - Iterator: Provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
   - Mediator: Reduces coupling between classes by introducing a mediator object.
   - Memento: Captures and externalizes an object's internal state.
   - Observer: Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
   - State: Allows an object to alter its behavior when its internal state changes.
   - Strategy: Defines a family of algorithms, encapsulates each one, and makes them interchangeable.
   - Template Method: Defines the skeleton of an algorithm in an operation, deferring some steps to subclasses.
   - Visitor: Separates an algorithm from an object structure on which it operates.""",
            
            "SOLID Principles": """SOLID is an acronym for five design principles intended to make software designs more maintainable, flexible, and understandable.

1. Single Responsibility Principle (SRP):
   - A class should have only one reason to change, meaning it should have only one responsibility.
   - When a class handles multiple responsibilities, it becomes coupled, making it more difficult to maintain.
   - Example: A class that handles both user data persistence and user authentication violates SRP.

2. Open/Closed Principle (OCP):
   - Software entities (classes, modules, functions) should be open for extension but closed for modification.
   - This means you should be able to add new functionality without changing existing code.
   - Using interfaces and abstract classes helps follow this principle.
   - Example: Using strategy pattern to add new sorting algorithms without modifying the client code.

3. Liskov Substitution Principle (LSP):
   - Objects of a superclass should be replaceable with objects of a subclass without affecting the correctness of the program.
   - Subtypes must be substitutable for their base types.
   - Example: If a function expects a Bird that can fly, passing a Penguin (which can't fly) would violate LSP.

4. Interface Segregation Principle (ISP):
   - Clients should not be forced to depend on methods they do not use.
   - Many client-specific interfaces are better than one general-purpose interface.
   - Example: Instead of one large interface with multiple methods, create smaller, more specific interfaces.

5. Dependency Inversion Principle (DIP):
   - High-level modules should not depend on low-level modules. Both should depend on abstractions.
   - Abstractions should not depend on details. Details should depend on abstractions.
   - Example: Using dependency injection to decouple high-level components from low-level components.

Benefits of SOLID Principles:
- Improved maintainability and readability
- Reduced complexity
- Better testability
- Increased flexibility and extensibility
- Reduced fragility of the codebase"""
        },
        
        "Machine Learning": {
            "Neural Networks": """Neural networks are computing systems inspired by the biological neural networks in animal brains. They consist of artificial neurons organized in layers, capable of learning patterns from data.

Key Components of Neural Networks:

1. Neurons (or Nodes):
   - Basic computational units that take inputs, apply an activation function, and produce an output.
   - Each neuron receives inputs, multiplies them by weights, sums them up, adds a bias, and applies an activation function.

2. Layers:
   - Input Layer: Receives the raw input features.
   - Hidden Layers: Intermediate layers between input and output. Deep neural networks have multiple hidden layers.
   - Output Layer: Produces the final prediction or classification.

3. Weights and Biases:
   - Weights: Parameters that determine the strength of connections between neurons.
   - Biases: Additional parameters that allow shifting the activation function.
   - These parameters are learned during training.

4. Activation Functions:
   - Introduce non-linearity, allowing networks to learn complex patterns.
   - Common functions: ReLU (Rectified Linear Unit), Sigmoid, Tanh, Softmax.
   - Different functions serve different purposes (e.g., Softmax for multi-class classification).""",
            
            "Supervised Learning": """Supervised Learning is a machine learning approach where models learn from labeled training data to predict outputs for unseen data. The algorithm learns a mapping from inputs to outputs based on example input-output pairs.

Key Concepts in Supervised Learning:

1. Training Data:
   - Consists of input features (X) and target variables (Y).
   - Features are characteristics or attributes used to make predictions.
   - Labels are the outputs or correct answers that the model aims to predict.

2. Model Training:
   - The algorithm learns patterns from training data by minimizing a loss function.
   - Models adjust parameters to reduce the difference between predictions and actual values.
   - Training continues until convergence or a specified number of iterations.

3. Model Evaluation:
   - Assessing performance on unseen data (validation/test sets).
   - Common metrics: accuracy, precision, recall, F1-score, RMSE, R².
   - Techniques: cross-validation, holdout validation."""
        }
    }
    
    # Return the samples for the requested domain or the default samples if not found
    if domain in domain_samples:
        return domain_samples[domain]
    else:
        return default_samples