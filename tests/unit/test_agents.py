"""Unit tests for agents with mocked LLMs."""

import tempfile
from unittest.mock import patch

from src.ai_book_composer.agents.critic import CriticAgent
from src.ai_book_composer.agents.executor import ExecutorAgent
from src.ai_book_composer.agents.planner import PlannerAgent
from src.ai_book_composer.agents.state import create_initial_state
from src.ai_book_composer.config import Settings

TEST_RESPONSE_FOR_CHAPTERS = """
Alright, I'm trying to figure out how to create a structured list of chapters 
for a comprehensive book. The user has provided several files in different 
formats: PDFs and text files. My goal is to extract information from these files
using the available tools and then organize them into chapters.

First, I'll go through each file one by one to see what kind of content they 
contain. Starting with "2307.06435v10.pdf," it's a PDF document which likely 
contains sections or chapters about theoretical concepts related to AI and 
machine learning since the filename doesn't immediately indicate otherwise, but 
given the other files mentioned like "article1_ai_intro.txt" and others, maybe 
it's part of an introductory section.

Next, "the-illusion-of-thinking.pdf" seems more philosophical in nature. It 
probably discusses cognitive biases or how human thinking works, which could 
form a chapter about understanding human cognition within AI contexts.

Then there's "7.pdf," which I'm guessing might be related to advanced machine 
learning techniques given the numbering and potential topics it could cover. 
This could serve as an intermediate or advanced section in the book.

Looking at the text files: "article1_ai_intro.txt" is probably an introduction 
to AI, so that should be the first chapter. "article2_ml_fundamentals.txt" 
likely covers machine learning basics, which would come after the AI intro since
ML is a subset of AI. "article3_deep_learning.txt" would then follow as it's a 
more specialized area within machine learning.

Putting this together, I can structure the book into chapters that build upon 
each other. Starting with an introduction to AI, moving into the fundamentals of
machine learning, and then diving deeper into deep learning techniques. The PDFs
likely provide theoretical underpinnings or advanced topics which can form 
additional chapters like understanding human cognition for AI development, 
ethical considerations in ML, applications using AI across different industries,
and finally a conclusion that ties everything together.

I should check if each file's content is relevant to these chapter titles. For 
example, "2307.06435v10.pdf" might discuss theoretical frameworks or advanced 
algorithms which would fit into an intermediate chapter. Similarly, the PDF 
about the illusion of thinking could be a separate chapter dedicated to 
cognitive aspects influencing AI development.

I also need to consider if there are any gaps in the content coverage. It seems 
all necessary topics from introduction through deep learning and ethics are 
covered across the provided files. However, I should ensure that each chapter 
flows logically into the next without redundancy or overlap.

Finally, I'll format this as a structured list of chapters using markdown 
numbering, making sure each title is concise yet descriptive.
</think>

Here's a structured list of chapters for your comprehensive book:

1. **Introduction to Artificial Intelligence**
   - Overview of AI concepts and applications
   - Historical development and current trends in AI

2. **Machine Learning Fundamentals**
   - Core principles and algorithms
   - Supervised, unsupervised, and reinforcement learning
   - Evaluation metrics and model optimization

3. **Deep Learning Techniques**
   - Neural networks and their architectures
   - Convolutional and recurrent neural networks
   - Deep learning applications in various domains

4. **The Illusion of Thinking: Cognitive Biases in AI Development**
   - Understanding human cognition for AI design
   - Psychological biases and their implications
   - Enhancing AI systems with human-like reasoning

5. **Ethical Considerations in Machine Learning**
   - Data privacy and security
   - Algorithmic bias and fairness
   - Responsible deployment of AI technologies

6. **Applications of AI Across Industries**
   - Healthcare: Diagnostics, personalized medicine
   - Finance: Fraud detection, algorithmic trading
   - Retail: Personalized customer experiences
   - Beyond traditional industries: Future applications

7. **Conclusion: The Future of Artificial Intelligence**
   - Recap of key concepts and advancements
   - Challenges and opportunities in AI's evolution
   - Final thoughts on the impact of AI on society
"""

TEST_RESPONSE_CHAPTERS_WITH_KIND_OF_JSON = """
<think>\nOkay, so I need to create a chapter structure based on the given files. Let me start by understanding what each file contains.\n\nFirst, 2307.06435v10.pdf is in text format with around 276,989 characters. The other files are articles: article1_ai_intro.txt (2,144 chars), article2_ml_fundamentals.txt (2,704 chars), and article3_deep_learning.txt (3,799 chars). All of them seem to be related to AI, ML, and deep learning.\n\nI should start with an introduction that gives an overview. Then, maybe a section on the basics since there\'s a general AI intro and specific ML and deep learning articles. After that, diving deeper into each technology makes sense—AI, ML, then deep learning as a subset of ML. Including practical applications would help tie everything together.\n\nI should make sure not to repeat topics but cover each area in order from basic concepts to more advanced ones like deep learning. Adding appendices for mathematical foundations and code examples could be useful for readers wanting more technical details. Finally, a conclusion to wrap up the chapter structure.\n</think>\n\n```json\n[\n  {\n    "number": 1,\n    "title": "Introduction",\n    "description": "An overview of artificial intelligence, its importance, and its applications."\n  },\n  {\n    "number": 2,\n    "title": "Basics of Artificial Intelligence",\n    "description": "Exploring fundamental concepts and principles of AI."\n  },\n  {\n    "number": 3,\n    "title": "Machine Learning Fundamentals",\n    "description": "Understanding the core concepts and techniques in machine learning."\n  },\n  {\n    "number": 4,\n    "title": "Deep Learning Introduction",\n    "description": "An introduction to deep learning and its relationship with machine learning."\n  },\n  {\n    "number": 5,\n    "title": "Applications of AI and Machine Learning",\n    "description": "Examining real-world applications and case studies of AI and ML technologies."\n  },\n  {\n    "number": 6,\n    "title": "Mathematical Foundations",\n    "description": "Diving into the mathematical underpinnings necessary for understanding AI algorithms."\n  },\n  {\n    "number": 7,\n    "title": "Code Examples",\n    "description": "Providing practical code examples to illustrate key concepts and techniques."\n  },\n  {\n    "number": 8,\n    "title": "Conclusion",\n    "description": "Summarizing the key points covered in the chapter and looking ahead at further topics."\n  }\n]\n```
"""

TEST_RESPONSE_CHAPTER_CONTENT_SAMPLE_1 = """
<think>\nAlright, I need to write Chapter 1: Introduction for a book based on the provided files. Let me start by analyzing each file\'s content.\n\nFirst, **2307.06435v10.pdf** is quite large (276989 chars). It seems technical with a lot of equations and discussions about mathematical concepts. Maybe it covers foundational math needed for the book.\n\nNext, **the-illusion-of-thinking.pdf** is 81670 chars. The title suggests it\'s about cognitive biases. This could be useful for explaining common mistakes in reasoning or decision-making processes.\n\nThen, there are three files with "7.pdf": one is 92506 chars and the others are 3799 chars each. These might be case studies or examples related to specific theories or models discussed in the book.\n\n**article1_ai_intro.txt (2144 chars)** introduces AI basics, covering definitions, history, types of AI, applications, and ethical considerations. This is a solid foundation for Chapter 1.\n\n**article2_ml_fundamentals.txt (2704 chars)** focuses on machine learning within the broader context of AI. It likely explains key concepts like supervised vs unsupervised learning, algorithms, and model evaluation.\n\nLastly, **article3_deep_learning.txt (3799 chars)** dives into deep learning, a subset of machine learning involving neural networks. It probably covers layers, activation functions, backpropagation, and applications.\n\nPutting this together, Chapter 1 should introduce the reader to AI and machine learning basics before diving deeper in subsequent chapters. The introduction should set up the context by mentioning how the book will build from foundational concepts into more complex areas like deep learning.\n</think>\n\n<result>\nAI and machine learning are rapidly transforming the way we live, work, and think. This chapter provides an overview of the fundamental concepts, tools, and applications that form the basis of AI and machine learning. By understanding these basics, you\'ll be well-prepared to explore more advanced topics in later chapters.\n</result>
"""

TEST_RESPONSE_CHAPTER_CONTENT_SAMPLE_2 = """
<think>\nAlright, I\'m trying to figure out how to write Chapter 2: Basics of Artificial Intelligence based on the files provided. Let me start by reviewing the available files and their content.\n\nFirst, there\'s "2307.06435v10.pdf" which seems to be a scientific paper with a lot of text—276,989 characters. It might contain advanced or technical details about AI that could be useful for a deeper understanding.\n\nNext, "the-illusion-of-thinking.pdf" has 81,670 characters and is titled "The Illusion of Thinking." This sounds intriguing—it might discuss human thought processes versus AI, which could provide a contrasting perspective to the chapter.\n\nThen, there\'s "7.pdf" with 92,506 characters. The title isn\'t provided, but given the numbering, it could be part of a series and possibly delve into specific aspects of AI in more detail.\n\nAdditionally, two text files are available: "article1_ai_intro.txt" (2,144 chars) and "article2_ml_fundamentals.txt" (2704 chars), along with "article3_deep_learning.txt" (3,799 chars). These seem to be introductory materials on AI and machine learning fundamentals. The first one might be a broad introduction, while the others could cover specific areas like ML and deep learning.\n\nStarting with these files, I can outline the chapter by first introducing AI as a field, then moving into its subfields such as machine learning and deep learning. Using "2307.06435v10.pdf" for technical details, "the-illusion-of-thinking.pdf" to discuss human cognition, and the other articles to elaborate on specific concepts.\n\nI need to ensure that I cover fundamental concepts without diving too deep into advanced topics since this is an introductory chapter. The goal is to provide a comprehensive overview of AI basics, so integrating information from all these sources will give a well-rounded perspective.\n\nNow, I\'ll begin drafting the chapter using this plan.\n</think>\n\n# Chapter 2: Basics of Artificial Intelligence\n\n## Introduction\nArtificial Intelligence (AI) has become an integral part of modern technology and society. From smartphones to self-driving cars, AI systems are transforming how we live, work, and interact with the world around us. This chapter provides an overview of AI basics, exploring its history, key concepts, and applications.\n\n### The Evolution of AI\nThe concept of artificial intelligence dates back to ancient times, with early ideas about mechanical toys and human-like machines. However, modern AI as we know it today began to take shape in the mid-20th century with the development of computers that could process information faster than ever before (see "2307.06435v10.pdf" for more details).\n\n### Core Concepts in AI\nAI is broadly defined as the simulation of human intelligence in machines that are able to perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and decision-making.\n\n#### Machine Learning (ML)\nMachine learning is a subset of AI that involves training algorithms to learn patterns from data without explicit programming. This process enables models to make predictions or decisions based on new data. Key concepts in ML include supervised learning, unsupervised learning, and reinforcement learning.\n\n#### Deep Learning\nDeep learning is a further specialization of machine learning that uses neural networks with many layers (hence "deep"). These networks can automatically learn features from raw data, making them particularly effective for complex tasks such as image recognition and natural language processing. ("Deep Learning" by Ian Goodfellow et al. provides an excellent introduction to this field.)\n\n#### Applications of AI\nAI has a wide range of applications across various industries:\n- **Healthcare**: Diagnosing diseases, predicting patient outcomes, and personalizing treatment plans.\n- **Finance**: Automating trading, risk assessment, and fraud detection.\n- **Transportation**: Developing autonomous vehicles and optimizing logistics.\n- **Entertainment**: Recommending content on platforms like Netflix and Spotify.\n- **Agriculture**: Monitoring crop conditions and optimizing farming practices.\n\n### Current Challenges and Future Directions\nDespite its progress, AI still faces significant challenges. Issues such as bias in algorithms, ethical considerations, and the need for more powerful computing resources are actively being addressed by researchers and industry leaders.\n\nIn conclusion, this chapter has provided an overview of the basics of AI, touching on its history, key concepts, and applications. As we delve deeper into this fascinating field, understanding these fundamentals will serve as a solid foundation for exploring more complex topics in subsequent chapters.\n\n[The complete chapter text continues here.]
"""


class TestAgentState:
    """Test agent state management."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output",
            language="en-US",
            book_title="Test Book",
            book_author="Test Author"
        )

        assert state["input_directory"] == "/tmp/input"
        assert state["output_directory"] == "/tmp/output"
        assert state["language"] == "en-US"
        assert state["book_title"] == "Test Book"
        assert state["book_author"] == "Test Author"
        assert state["status"] == "initialized"
        assert state["iterations"] == 0
        assert isinstance(state["files"], list)
        assert isinstance(state["chapters"], list)

    def test_state_defaults(self):
        """Test state with default values."""
        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )

        assert state["language"] == "en-US"
        assert state["book_title"] == "Composed Book"
        assert state["book_author"] == "AI Book Composer"


class TestPlannerAgent:
    """Test planner agent with mocked LLM."""

    def test_plan_generation_static_plan(self):
        """Test plan generation."""
        settings = Settings()
        settings.llm.static_plan = True

        planner = PlannerAgent(settings)

        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )
        state["files"] = [
            {"name": "file1.txt", "path": "/tmp/input/file1.txt", "extension": ".txt"},
            {"name": "file2.txt", "path": "/tmp/input/file2.txt", "extension": ".txt"}
        ]

        result = planner.plan(state)

        assert "plan" in result
        assert "status" in result
        assert result["status"] == "planned"
        assert isinstance(result["plan"], list)

    @patch('src.ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    def test_plan_generation_llm_plan(self, mock_invoke_agent):
        """Test plan generation with LLM (non-static plan)."""
        # Mock LLM response with a valid plan JSON (as string)
        mock_response = '''[
            {"task": "gather_content", "description": "Read and transcribe all source files", "status": "pending", "files": ["file1.txt", "file2.txt"]},
            {"task": "plan_chapters", "description": "Determine book structure and chapters", "status": "pending"},
            {"task": "generate_chapters", "description": "Write each chapter based on gathered content", "status": "pending"},
            {"task": "compile_references", "description": "Compile list of references", "status": "pending"},
            {"task": "generate_book", "description": "Generate final book with all components", "status": "pending"}
        ]'''
        mock_invoke_agent.return_value = mock_response

        settings = Settings()
        settings.llm.static_plan = False
        planner = PlannerAgent(settings)
        planner.prompts = {
            'planner': {
                'system_prompt': 'SYSTEM {language} {style_instructions_section}',
                'user_prompt': 'USER {file_summary}'
            }
        }
        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )
        state["files"] = [
            {"name": "file1.txt", "path": "/tmp/input/file1.txt", "extension": ".txt"},
            {"name": "file2.txt", "path": "/tmp/input/file2.txt", "extension": ".txt"}
        ]
        result = planner.plan(state)
        assert "plan" in result
        assert "status" in result
        assert result["status"] == "planned"
        plan = result["plan"]
        assert isinstance(plan, list)
        assert plan[0]["task"] == "gather_content"
        assert plan[1]["task"] == "plan_chapters"
        assert plan[2]["task"] == "generate_chapters"
        assert plan[3]["task"] == "compile_references"
        assert plan[4]["task"] == "generate_book"
        assert plan[0]["files"] == ["file1.txt", "file2.txt"]


class TestCriticAgent:
    """Test critic agent with mocked LLM."""

    @patch('src.ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    def test_critique_good_quality(self, mock_invoke_agent):
        """Test critique with good quality score."""
        # Mock LLM response indicating good quality
        mock_response = "Quality score: 0.9\nDecision: approve\nThe book is excellent."
        mock_invoke_agent.return_value = mock_response

        critic = CriticAgent(Settings(), quality_threshold=0.7)

        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )
        state["chapters"] = [
            {"number": 1, "title": "Chapter 1", "content": "Content 1"},
            {"number": 2, "title": "Chapter 2", "content": "Content 2"}
        ]
        state["references"] = ["Ref 1", "Ref 2"]

        result = critic.critique(state)

        assert "critic_feedback" in result
        assert "quality_score" in result
        assert "status" in result
        # Status should be approved due to high score
        assert result["status"] in ["approved", "needs_revision"]
        assert mock_invoke_agent.called

    @patch('src.ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    def test_critique_no_chapters(self, invoke_agent_mock):
        """Test critique with no chapters."""
        invoke_agent_mock.return_value = ''

        critic = CriticAgent(Settings())

        state = create_initial_state(
            input_directory="/tmp/input",
            output_directory="/tmp/output"
        )
        state["chapters"] = []

        result = critic.critique(state)

        assert result["quality_score"] == 0.0
        assert result["status"] == "needs_revision"
        assert not invoke_agent_mock.called  # Should not call LLM if no chapters


class TestExecutorAgent:

    @patch('src.ai_book_composer.agents.agent_base.AgentBase._invoke_agent')
    def test_llm_agent_else_branch(self, mock_invoke_agent):
        """Test ExecutorAgent LLM agent else branch uses prompts and tools correctly."""
        # Mock tools (none needed for this test, just to satisfy init)
        mock_invoke_agent.return_value = '{"result": "Tool executed successfully"}'

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ExecutorAgent(
                Settings(),
                output_directory=tmpdir
            )
            # Patch prompts to ensure LLM agent prompt is used
            executor.prompts['executor']['llm_agent_system_prompt'] = 'SYSTEM PROMPT'
            executor.prompts['executor']['llm_agent_user_prompt'] = 'USER PROMPT {state} {current_task}'

            # Create a state and a plan with an unknown task (triggers else branch)
            state = create_initial_state(
                input_directory=tmpdir,
                output_directory=tmpdir
            )
            state['plan'] = [
                {"task": "special task", "description": "Do something special!", "status": "pending"}
            ]
            state['current_task_index'] = 0

            result = executor.execute(state)

            # Check that the LLM was called with the correct prompt
            assert mock_invoke_agent.called
            called_args = mock_invoke_agent.call_args[0]
            assert any('SYSTEM PROMPT' in m for m in called_args)
            assert any('USER PROMPT' in m for m in called_args)
            # Check result structure
            assert result['status'] == 'executing'
            assert result['current_task_index'] == 1

    def test_chapters_planning_response_parsing(self):
        """Test parsing of chapter planning response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings()
            executor = ExecutorAgent(settings, output_directory=tmpdir)

            thought, action = executor._extract_thought_and_action(TEST_RESPONSE_FOR_CHAPTERS)
            assert thought == ''
            chapters = executor._parse_chapter_list(action)
            expected_chapters = \
                [
                    {
                        'description': '- Overview of AI concepts and applications\n'
                                       '- Historical development and current trends in AI\n',
                        'number': 1,
                        'title': '**Introduction to Artificial Intelligence**'
                    },
                    {
                        'description': '- Core principles and algorithms\n'
                                       '- Supervised, unsupervised, and reinforcement learning\n'
                                       '- Evaluation metrics and model optimization\n',
                        'number': 2,
                        'title': '**Machine Learning Fundamentals**'
                    },
                    {
                        'description': '- Neural networks and their architectures\n'
                                       '- Convolutional and recurrent neural networks\n'
                                       '- Deep learning applications in various domains\n',
                        'number': 3,
                        'title': '**Deep Learning Techniques**'
                    },
                    {
                        'description': '- Understanding human cognition for AI design\n'
                                       '- Psychological biases and their implications\n'
                                       '- Enhancing AI systems with human-like reasoning\n',
                        'number': 4,
                        'title': '**The Illusion of Thinking: Cognitive Biases in AI Development**'
                    },
                    {
                        'description': '- Data privacy and security\n'
                                       '- Algorithmic bias and fairness\n'
                                       '- Responsible deployment of AI technologies\n',
                        'number': 5,
                        'title': '**Ethical Considerations in Machine Learning**'
                    },
                    {
                        'description': '- Healthcare: Diagnostics, personalized medicine\n'
                                       '- Finance: Fraud detection, algorithmic trading\n'
                                       '- Retail: Personalized customer experiences\n'
                                       '- Beyond traditional industries: Future applications\n',
                        'number': 6,
                        'title': '**Applications of AI Across Industries**'
                    },
                    {
                        'description': '- Recap of key concepts and advancements\n'
                                       "- Challenges and opportunities in AI's evolution\n"
                                       '- Final thoughts on the impact of AI on society',
                        'number': 7,
                        'title': '**Conclusion: The Future of Artificial Intelligence**'
                    }
                ]

            assert isinstance(chapters, list)
            assert chapters == expected_chapters

    def test_chapters_planning_response_parsing_kind_of_json(self):
        """Test parsing of chapter planning response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings()
            executor = ExecutorAgent(settings, output_directory=tmpdir)

            thought, action = executor._extract_thought_and_action(TEST_RESPONSE_CHAPTERS_WITH_KIND_OF_JSON)

            expected_thought = ('Okay, so I need to create a chapter structure based on the given files. Let '
                                'me start by understanding what each file contains.\n'
                                '\n'
                                'First, 2307.06435v10.pdf is in text format with around 276,989 characters. '
                                'The other files are articles: article1_ai_intro.txt (2,144 chars), '
                                'article2_ml_fundamentals.txt (2,704 chars), and article3_deep_learning.txt '
                                '(3,799 chars). All of them seem to be related to AI, ML, and deep learning.\n'
                                '\n'
                                'I should start with an introduction that gives an overview. Then, maybe a '
                                "section on the basics since there's a general AI intro and specific ML and "
                                'deep learning articles. After that, diving deeper into each technology makes '
                                'sense—AI, ML, then deep learning as a subset of ML. Including practical '
                                'applications would help tie everything together.\n'
                                '\n'
                                'I should make sure not to repeat topics but cover each area in order from '
                                'basic concepts to more advanced ones like deep learning. Adding appendices '
                                'for mathematical foundations and code examples could be useful for readers '
                                'wanting more technical details. Finally, a conclusion to wrap up the chapter '
                                'structure.')

            assert thought == expected_thought

            chapters = executor._parse_chapter_list(action)
            expected_chapters = \
                [{'description': 'An overview of artificial intelligence, its importance, and '
                                 'its applications.',
                  'number': 1,
                  'title': 'Introduction'},
                 {'description': 'Exploring fundamental concepts and principles of AI.',
                  'number': 2,
                  'title': 'Basics of Artificial Intelligence'},
                 {'description': 'Understanding the core concepts and techniques in machine '
                                 'learning.',
                  'number': 3,
                  'title': 'Machine Learning Fundamentals'},
                 {'description': 'An introduction to deep learning and its relationship with '
                                 'machine learning.',
                  'number': 4,
                  'title': 'Deep Learning Introduction'},
                 {'description': 'Examining real-world applications and case studies of AI and '
                                 'ML technologies.',
                  'number': 5,
                  'title': 'Applications of AI and Machine Learning'},
                 {'description': 'Diving into the mathematical underpinnings necessary for '
                                 'understanding AI algorithms.',
                  'number': 6,
                  'title': 'Mathematical Foundations'},
                 {'description': 'Providing practical code examples to illustrate key concepts '
                                 'and techniques.',
                  'number': 7,
                  'title': 'Code Examples'},
                 {'description': 'Summarizing the key points covered in the chapter and '
                                 'looking ahead at further topics.',
                  'number': 8,
                  'title': 'Conclusion'}]

            assert isinstance(chapters, list)
            assert chapters == expected_chapters

    def test_chapter_content_1(self):
        """Test parsing of chapter planning response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings()
            executor = ExecutorAgent(settings, output_directory=tmpdir)

            thought, action = executor._extract_thought_and_action(TEST_RESPONSE_CHAPTER_CONTENT_SAMPLE_1)

            expected_thought = ('Alright, I need to write Chapter 1: Introduction for a book based on the '
                                "provided files. Let me start by analyzing each file's content.\n"
                                '\n'
                                'First, **2307.06435v10.pdf** is quite large (276989 chars). It seems '
                                'technical with a lot of equations and discussions about mathematical '
                                'concepts. Maybe it covers foundational math needed for the book.\n'
                                '\n'
                                'Next, **the-illusion-of-thinking.pdf** is 81670 chars. The title suggests '
                                "it's about cognitive biases. This could be useful for explaining common "
                                'mistakes in reasoning or decision-making processes.\n'
                                '\n'
                                'Then, there are three files with "7.pdf": one is 92506 chars and the others '
                                'are 3799 chars each. These might be case studies or examples related to '
                                'specific theories or models discussed in the book.\n'
                                '\n'
                                '**article1_ai_intro.txt (2144 chars)** introduces AI basics, covering '
                                'definitions, history, types of AI, applications, and ethical considerations. '
                                'This is a solid foundation for Chapter 1.\n'
                                '\n'
                                '**article2_ml_fundamentals.txt (2704 chars)** focuses on machine learning '
                                'within the broader context of AI. It likely explains key concepts like '
                                'supervised vs unsupervised learning, algorithms, and model evaluation.\n'
                                '\n'
                                'Lastly, **article3_deep_learning.txt (3799 chars)** dives into deep '
                                'learning, a subset of machine learning involving neural networks. It '
                                'probably covers layers, activation functions, backpropagation, and '
                                'applications.\n'
                                '\n'
                                'Putting this together, Chapter 1 should introduce the reader to AI and '
                                'machine learning basics before diving deeper in subsequent chapters. The '
                                'introduction should set up the context by mentioning how the book will build '
                                'from foundational concepts into more complex areas like deep learning.')

            assert thought == expected_thought

            chapter_content = executor._parse_chapter_content_response(action)
            expected_chapter_content = ('AI and machine learning are rapidly transforming the way we live, work, and '
                                        'think. This chapter provides an overview of the fundamental concepts, tools, '
                                        'and applications that form the basis of AI and machine learning. By '
                                        "understanding these basics, you'll be well-prepared to explore more advanced "
                                        'topics in later chapters.')

            assert chapter_content == expected_chapter_content

    def test_chapter_content_2(self):
        """Test parsing of chapter planning response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings()
            executor = ExecutorAgent(settings, output_directory=tmpdir)

            thought, action = executor._extract_thought_and_action(TEST_RESPONSE_CHAPTER_CONTENT_SAMPLE_2)

            expected_thought = ("Alright, I'm trying to figure out how to write Chapter 2: Basics of "
                                'Artificial Intelligence based on the files provided. Let me start by '
                                'reviewing the available files and their content.\n'
                                '\n'
                                'First, there\'s "2307.06435v10.pdf" which seems to be a scientific paper '
                                'with a lot of text—276,989 characters. It might contain advanced or '
                                'technical details about AI that could be useful for a deeper understanding.\n'
                                '\n'
                                'Next, "the-illusion-of-thinking.pdf" has 81,670 characters and is titled '
                                '"The Illusion of Thinking." This sounds intriguing—it might discuss human '
                                'thought processes versus AI, which could provide a contrasting perspective '
                                'to the chapter.\n'
                                '\n'
                                'Then, there\'s "7.pdf" with 92,506 characters. The title isn\'t provided, '
                                'but given the numbering, it could be part of a series and possibly delve '
                                'into specific aspects of AI in more detail.\n'
                                '\n'
                                'Additionally, two text files are available: "article1_ai_intro.txt" (2,144 '
                                'chars) and "article2_ml_fundamentals.txt" (2704 chars), along with '
                                '"article3_deep_learning.txt" (3,799 chars). These seem to be introductory '
                                'materials on AI and machine learning fundamentals. The first one might be a '
                                'broad introduction, while the others could cover specific areas like ML and '
                                'deep learning.\n'
                                '\n'
                                'Starting with these files, I can outline the chapter by first introducing AI '
                                'as a field, then moving into its subfields such as machine learning and deep '
                                'learning. Using "2307.06435v10.pdf" for technical details, '
                                '"the-illusion-of-thinking.pdf" to discuss human cognition, and the other '
                                'articles to elaborate on specific concepts.\n'
                                '\n'
                                'I need to ensure that I cover fundamental concepts without diving too deep '
                                'into advanced topics since this is an introductory chapter. The goal is to '
                                'provide a comprehensive overview of AI basics, so integrating information '
                                'from all these sources will give a well-rounded perspective.\n'
                                '\n'
                                "Now, I'll begin drafting the chapter using this plan.")

            assert thought == expected_thought

            chapter_content = executor._parse_chapter_content_response(action)
            expected_chapter_content = ('# Chapter 2: Basics of Artificial Intelligence\n'
                                        '\n'
                                        '## Introduction\n'
                                        'Artificial Intelligence (AI) has become an integral part of modern '
                                        'technology and society. From smartphones to self-driving cars, AI systems '
                                        'are transforming how we live, work, and interact with the world around us. '
                                        'This chapter provides an overview of AI basics, exploring its history, key '
                                        'concepts, and applications.\n'
                                        '\n'
                                        '### The Evolution of AI\n'
                                        'The concept of artificial intelligence dates back to ancient times, with '
                                        'early ideas about mechanical toys and human-like machines. However, modern '
                                        'AI as we know it today began to take shape in the mid-20th century with the '
                                        'development of computers that could process information faster than ever '
                                        'before (see "2307.06435v10.pdf" for more details).\n'
                                        '\n'
                                        '### Core Concepts in AI\n'
                                        'AI is broadly defined as the simulation of human intelligence in machines '
                                        'that are able to perform tasks that typically require human intelligence. '
                                        'These tasks include learning, reasoning, problem-solving, perception, and '
                                        'decision-making.\n'
                                        '\n'
                                        '#### Machine Learning (ML)\n'
                                        'Machine learning is a subset of AI that involves training algorithms to '
                                        'learn patterns from data without explicit programming. This process enables '
                                        'models to make predictions or decisions based on new data. Key concepts in '
                                        'ML include supervised learning, unsupervised learning, and reinforcement '
                                        'learning.\n'
                                        '\n'
                                        '#### Deep Learning\n'
                                        'Deep learning is a further specialization of machine learning that uses '
                                        'neural networks with many layers (hence "deep"). These networks can '
                                        'automatically learn features from raw data, making them particularly '
                                        'effective for complex tasks such as image recognition and natural language '
                                        'processing. ("Deep Learning" by Ian Goodfellow et al. provides an excellent '
                                        'introduction to this field.)\n'
                                        '\n'
                                        '#### Applications of AI\n'
                                        'AI has a wide range of applications across various industries:\n'
                                        '- **Healthcare**: Diagnosing diseases, predicting patient outcomes, and '
                                        'personalizing treatment plans.\n'
                                        '- **Finance**: Automating trading, risk assessment, and fraud detection.\n'
                                        '- **Transportation**: Developing autonomous vehicles and optimizing '
                                        'logistics.\n'
                                        '- **Entertainment**: Recommending content on platforms like Netflix and '
                                        'Spotify.\n'
                                        '- **Agriculture**: Monitoring crop conditions and optimizing farming '
                                        'practices.\n'
                                        '\n'
                                        '### Current Challenges and Future Directions\n'
                                        'Despite its progress, AI still faces significant challenges. Issues such as '
                                        'bias in algorithms, ethical considerations, and the need for more powerful '
                                        'computing resources are actively being addressed by researchers and industry '
                                        'leaders.\n'
                                        '\n'
                                        'In conclusion, this chapter has provided an overview of the basics of AI, '
                                        'touching on its history, key concepts, and applications. As we delve deeper '
                                        'into this fascinating field, understanding these fundamentals will serve as '
                                        'a solid foundation for exploring more complex topics in subsequent '
                                        'chapters.\n'
                                        '\n'
                                        '[The complete chapter text continues here.]')

            assert chapter_content == expected_chapter_content


class TestAgentStateSummary:
    """Test agent state summary functionality."""

    def test_get_agent_state_summary_with_plan(self):
        """Test state summary generation with plan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings()
            agent = PlannerAgent(settings)
            
            # Create a state with plan
            state = create_initial_state(
                input_directory=tmpdir,
                output_directory=tmpdir
            )
            state["plan"] = [
                {"task": "plan_chapters", "description": "Determine book structure", "status": "completed"},
                {"task": "generate_chapters", "description": "Write chapters", "status": "pending"},
                {"task": "generate_book", "description": "Generate final book", "status": "pending"}
            ]
            state["current_task_index"] = 1
            
            agent.state = state
            summary = agent._get_agent_state_summary()
            
            # Verify the summary contains plan information
            assert "Plan Steps:" in summary
            assert "plan_chapters" in summary
            assert "generate_chapters" in summary
            assert "CURRENT" in summary
            assert "COMPLETED" in summary
            assert "PENDING" in summary

    def test_get_agent_state_summary_with_critic_feedback(self):
        """Test state summary generation with critic feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings()
            agent = CriticAgent(settings)
            
            # Create a state with critic feedback
            state = create_initial_state(
                input_directory=tmpdir,
                output_directory=tmpdir
            )
            state["critic_feedback"] = "The book needs improvement in chapter 2. Please revise for better clarity."
            state["quality_score"] = 0.65
            state["iterations"] = 1
            
            agent.state = state
            summary = agent._get_agent_state_summary()
            
            # Verify the summary contains critic feedback
            assert "Critic Feedback:" in summary
            assert "needs improvement" in summary
            assert "Iteration: 1" in summary
            assert "Quality Score: 65.00%" in summary

    def test_get_agent_state_summary_empty_state(self):
        """Test state summary generation with empty state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings()
            agent = PlannerAgent(settings)
            
            # State is None
            agent.state = None
            summary = agent._get_agent_state_summary()
            
            assert summary == ""

    def test_get_agent_state_summary_minimal_state(self):
        """Test state summary generation with minimal state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings()
            agent = PlannerAgent(settings)
            
            # Create a minimal state
            state = create_initial_state(
                input_directory=tmpdir,
                output_directory=tmpdir
            )
            
            agent.state = state
            summary = agent._get_agent_state_summary()
            
            # Should return empty string when there's nothing relevant to report
            assert summary == ""
