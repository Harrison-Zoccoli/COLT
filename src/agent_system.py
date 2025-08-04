"""
Multi-Agent Legal Intake System

Boilerplate code with placeholder functions for each agent.
Uses OpenAI models with the ability to swap models per agent.
"""
try:
    from .openai_processor import process_openai_request, process_openai_request_streaming, ModelParams, setup_openai_client
except ImportError:
    from openai_processor import process_openai_request, process_openai_request_streaming, ModelParams, setup_openai_client

from typing import Dict, Any, Optional
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import random
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a separate logger for clean conversation output
conversation_logger = logging.getLogger("conversation")
conversation_handler = logging.StreamHandler()
conversation_handler.setFormatter(logging.Formatter('%(message)s'))
conversation_logger.addHandler(conversation_handler)
conversation_logger.setLevel(logging.INFO)
conversation_logger.propagate = False  # Don't send to root logger

# Global debug mode flag
DEBUG_MODE = False

def set_debug_mode(enabled: bool = True):
    """Enable/disable verbose debug logging."""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    
    if enabled:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔧 Debug mode ENABLED - showing all logs")
    else:
        logging.getLogger().setLevel(logging.WARNING)  # Hide INFO logs
        conversation_logger.info("🤫 Debug mode DISABLED - showing only conversation")

class GlobalMemoryBoard:
    """Shared memory board for agent coordination."""
    
    def __init__(self, questionnaire_file: Optional[str] = None, buck_data: Optional[Dict[str, Any]] = None):
        """Initialize the global memory board."""
        #########################################################
        # SEPARATE caller type history tracking (not in self.data)
        self.caller_type_history = []  # Track sequence of caller type classifications
        #########################################################
        # load buck data if provided
        self.buck_data = buck_data or {}

        # segrate buck data into lawfirm_info, caller_prompt_info, caller_non_prompt_info
        # if self.buck_data:
        self.lawfirm_info = self.buck_data.get("lawfirm_info", {})
        self.caller_prompt_info = self.buck_data.get("caller_prompt_info", {})
        self.caller_non_prompt_info = self.buck_data.get("caller_non_prompt_info", {})
        self.agent_info = self.buck_data.get("agent_info", {})
        self.new_client_questionnaire = self.buck_data.get("new_client_questionnaire", {})
        self.existing_client_questionnaire = self.buck_data.get("existing_client_questionnaire", {})
        self.existing_client = self.buck_data.get("existing_client")
        if self.agent_info:
            self.luna_config = self.agent_info.get("luna_config", {})
            self.agent_0_config = self.agent_info.get("agent_0_config", {})
            self.agent_1_config = self.agent_info.get("agent_1_config", {})
            self.agent_2_config = self.agent_info.get("agent_2_config", {})
            # self.agent_3_config = self.agent_info.get("agent_3_config", {})
            # self.agent_4_config = self.agent_info.get("agent_4_config", {})
            # self.agent_5_config = self.agent_info.get("agent_5_config", {})
        #########################################################

        # Load questionnaire from file if provided
        questionnaire = {}
        if self.existing_client:
            questionnaire = self.existing_client_questionnaire
        else:
            questionnaire = self.new_client_questionnaire

        # if questionnaire_file:
        #     try:
        #         with open(questionnaire_file, 'r') as f:
        #             questionnaire = json.load(f)
        #     except Exception as e:
        #         logger.error(f"Failed to load questionnaire file {questionnaire_file}: {e}")
        #         questionnaire = {"caller_type": "unknown", "questions": []}

        self.data = {
            "transcript": [],                                                         # List of conversation messages
            # "unproessed_transcript": "",                                            # Full user interaction history
            "caller_type": "existing" if self.existing_client else None,              # HARDCODED: 'existing' if existing_client=True, otherwise set by Agent0
            "questionnaire": questionnaire,                                           # JSON form loaded from file
            "next_question": None,                                                    # Suggested next question to ask agent 1
            "is_complete": False,                                                     # Whether form is complete (set by Agent2)
            # "missing_fields": [],                                                   # List of fields still needed (Agent2)
            "qualification": {                                                        # Qualification results (set by Agent2)
                "qualification_status": "incomplete",  # qualified|not_qualified|incomplete
                "confidence": None,  # No assessment yet - will be 0.0-1.0 after Agent2 runs
                "reasoning": "",
                "red_flags": [],
                "strengths": [],
                "should_continue": True,
                "next_action": "continue",
                "completion_percentage": 0,
                "missing_questions": []
            },
            "session_id": None                # Session identifier (set during initialization)
        }

    def update(self, key: str, value: Any) -> None:
        """Update a value in the memory board."""
        self.data[key] = value
        logger.info(f"Memory board updated: {key}")
    
    def get(self, key: str) -> Any:
        """Get a value from the memory board."""
        return self.data.get(key)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all data from the memory board."""
        return self.data.copy()
    
    def load_questionnaire(self, questionnaire_file: str) -> None:
        """Load a new questionnaire from file."""
        try:
            with open(questionnaire_file, 'r') as f:
                questionnaire = json.load(f)
                self.data["questionnaire"] = questionnaire
                logger.info(f"Loaded questionnaire from {questionnaire_file}")
        except Exception as e:
            logger.error(f"Failed to load questionnaire file {questionnaire_file}: {e}")
    
    def add_caller_type_to_history(self, caller_type: str) -> None:
        """Add caller type to history sequence."""
        self.caller_type_history.append(caller_type)
        logger.info(f"📝 Added to caller type history: {caller_type}")
        logger.info(f"📝 Full history: {self.caller_type_history}")
    
    def get_majority_of_last_three(self, history: list) -> Optional[str]:
        """Get majority of last 3 items in history."""
        if len(history) < 3:
            return None
        
        last_three = history[-3:]
        count_new = last_three.count("new")
        count_existing = last_three.count("existing")
        count_unknown = last_three.count("unknown")
        
        if count_new > count_existing and count_new > count_unknown:
            return "new"
        elif count_existing > count_new and count_existing > count_unknown:
            return "existing"
        else:
            return "unknown"
    
    def should_load_questionnaire(self, new_caller_type: str) -> bool:
        """Check if questionnaire should be loaded based on majority logic."""
        # Add new caller type to history
        self.add_caller_type_to_history(new_caller_type)
        
        # If we don't have enough history yet, load questionnaire
        if len(self.caller_type_history) < 3:
            logger.info(f"📊 Not enough history ({len(self.caller_type_history)}) - loading questionnaire")
            return True
        
        # Get majority of current last 3
        current_majority = self.get_majority_of_last_three(self.caller_type_history)
        
        # Get majority of previous last 3 (if we have at least 4 items)
        if len(self.caller_type_history) >= 4:
            previous_majority = self.get_majority_of_last_three(self.caller_type_history[:-1])
        else:
            previous_majority = None
        
        # Check if majority changed
        if current_majority != previous_majority:
            logger.info(f"📊 Majority changed: {previous_majority} -> {current_majority}")
            logger.info(f"📊 History: {self.caller_type_history}")
            return True
        else:
            logger.info(f"🔒 Majority unchanged ({current_majority}) - NOT loading questionnaire")
            logger.info(f"📊 History: {self.caller_type_history}")
            return False

    # =================================================================
    # GETTER FUNCTIONS FOR ALL CONFIGURATIONS
    # =================================================================
    
    def get_buck_data(self) -> Dict[str, Any]:
        """Get buck data configuration."""
        return getattr(self, 'buck_data', {})
    
    def get_lawfirm_info(self) -> Dict[str, Any]:
        """Get lawfirm info configuration."""
        return getattr(self, 'lawfirm_info', {})
    
    def get_caller_prompt_info(self) -> Dict[str, Any]:
        """Get caller prompt info configuration."""
        return getattr(self, 'caller_prompt_info', {})
    
    def get_caller_non_prompt_info(self) -> Dict[str, Any]:
        """Get caller non-prompt info configuration."""
        return getattr(self, 'caller_non_prompt_info', {})
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent info configuration."""
        return getattr(self, 'agent_info', {})
    
    # Client configurations
    def get_existing_client(self) -> Dict[str, Any]:
        """Get existing client configuration."""
        return getattr(self, 'existing_client')
    
    # Questionnaire configurations
    def get_new_client_questionnaire(self) -> Dict[str, Any]:
        """Get new client questionnaire configuration."""
        return getattr(self, 'new_client_questionnaire', {})
    
    def get_existing_client_questionnaire(self) -> Dict[str, Any]:
        """Get existing client questionnaire configuration."""
        return getattr(self, 'existing_client_questionnaire', {})
    
    # Agent configurations
    def get_luna_config(self) -> Dict[str, Any]:
        """Get Luna agent configuration."""
        return getattr(self, 'luna_config', {})
    
    def get_agent_0_config(self) -> Dict[str, Any]:
        """Get Agent 0 (classifier) configuration."""
        return getattr(self, 'agent_0_config', {})
    
    def get_agent_1_config(self) -> Dict[str, Any]:
        """Get Agent 1 (form filler) configuration."""
        return getattr(self, 'agent_1_config', {})
    
    def get_agent_2_config(self) -> Dict[str, Any]:
        """Get Agent 2 (coordinator) configuration."""
        return getattr(self, 'agent_2_config', {})
    
    # def get_agent_3_config(self) -> Dict[str, Any]:
    #     """Get Agent 3 configuration."""
    #     return getattr(self, 'agent_3_config', {})
    
    # def get_agent_4_config(self) -> Dict[str, Any]:
    #     """Get Agent 4 configuration."""
    #     return getattr(self, 'agent_4_config', {})
    
    # def get_agent_5_config(self) -> Dict[str, Any]:
    #     """Get Agent 5 configuration."""
    #     return getattr(self, 'agent_5_config', {})
    
    # =================================================================
    # UTILITY FUNCTIONS FOR ALL CONFIGS
    # =================================================================
    
    def get_all_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all agent configurations in one dictionary."""
        return {
            'luna_config': self.get_luna_config(),
            'agent_0_config': self.get_agent_0_config(),
            'agent_1_config': self.get_agent_1_config(),
            'agent_2_config': self.get_agent_2_config(),
            # 'agent_3_config': self.get_agent_3_config(),
            # 'agent_4_config': self.get_agent_4_config(),
            # 'agent_5_config': self.get_agent_5_config()
        }
    
    def get_all_client_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all client configurations in one dictionary."""
        return {
            'existing_client': self.get_existing_client()
        }
    
    def get_all_questionnaire_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all questionnaire configurations in one dictionary."""
        return {
            'new_client_questionnaire': self.get_new_client_questionnaire(),
            'existing_client_questionnaire': self.get_existing_client_questionnaire()
        }
    
    def get_all_info_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all info configurations in one dictionary."""
        return {
            'buck_data': self.get_buck_data(),
            'lawfirm_info': self.get_lawfirm_info(),
            'caller_prompt_info': self.get_caller_prompt_info(),
            'caller_non_prompt_info': self.get_caller_non_prompt_info(),
            'agent_info': self.get_agent_info()
        }
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations in one comprehensive dictionary."""
        return {
            **self.get_all_info_configs(),
            **self.get_all_client_configs(),
            **self.get_all_questionnaire_configs(),
            **self.get_all_agent_configs()
        }

def load_buck_data_from_file(file_path: str = "src/sample_buck_data.json") -> Dict[str, Any]:
    """
    Load buck_data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing buck_data
        
    Returns:
        Dict[str, Any]: The loaded buck_data
    """
    try:
        with open(file_path, 'r') as f:
            buck_data = json.load(f)
        logger.info(f"Successfully loaded buck_data from {file_path}")
        return buck_data
    except FileNotFoundError:
        logger.error(f"Buck data file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in buck data file {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading buck data from {file_path}: {e}")
        return {}

# Global memory board instance - start with default new client questionnaire
# memory_board = GlobalMemoryBoard("new_client_questionaire.json")
buck_data = load_buck_data_from_file("src/sample_buck_data.json")
memory_board = GlobalMemoryBoard(buck_data=buck_data)



# def initialize_system_with_buck_data(buck_data_file: str = "sample_buck_data.json") -> None:
#     """
#     Initialize the system with buck_data loaded from a JSON file.
    
#     Args:
#         buck_data_file (str): Path to the JSON file containing buck_data
#     """
#     global memory_board
    
#     # Load buck_data from file
#     buck_data = load_buck_data_from_file(buck_data_file)
    
#     if buck_data:
#         # Create new memory board with loaded buck_data
#         memory_board = GlobalMemoryBoard(buck_data=buck_data)
#         logger.info("✅ System initialized with buck_data from JSON file")
#         logger.info(f"🏢 Law firm: {memory_board.get_lawfirm_info().get('firm_name', 'Unknown')}")
#         logger.info(f"👤 Existing client: {memory_board.get_existing_client()}")
#         logger.info(f"📝 New client questionnaire: {len(memory_board.get_new_client_questionnaire().get('questions', []))} questions")
#         logger.info(f"📋 Existing client questionnaire: {len(memory_board.get_existing_client_questionnaire().get('questions', []))} questions")
#     else:
#         logger.warning("❌ Failed to load buck_data, using default configuration")
#         memory_board = GlobalMemoryBoard()

def luna_streaming(
    input_text: str, 
    model: str = "gpt-4",
    params: Optional[ModelParams] = None
):
    """
    Luna - Real-time streaming conversational agent that talks to the user.
    
    Args:
        input_text (str): User's input text
        model (str): OpenAI model to use
        params (Optional[ModelParams]): Model parameters
    
    Yields:
        str: Individual characters/tokens as they stream in real-time
    """
    if params is None:
        params = ModelParams(temperature=0.7, max_tokens=500)
    
    # Add user input to transcript
    transcript = memory_board.get("transcript") or []
    transcript.append({"role": "user", "content": input_text})
    memory_board.update("transcript", transcript)
    
    # Get full memory context for Luna
    memory_data = memory_board.get_all()
    is_complete = memory_data.get("is_complete", False)
    qualification = memory_data.get("qualification", {})
    next_question = memory_data.get("next_question")
    
    #########################################################
    # Luna's system prompt with memory context
    # system_prompt = f"""
    # You are Luna, an empathetic intake specialist for {lawfirm_info}.
    # if caller is an existing client then caller prompt info will be available
    # if caller is an new client then caller prompt info will be empty
    # - Caller Prompt Info: {caller_prompt_info}
    
    # You're like a caring Californian mother who's handled thousands of these cases.
    # This is a phone conversation - your responses will be spoken aloud.
    
    # CONVERSATION STYLE:
    # - Keep responses to 3 sentences maximum
    # - Ask ONE question at a time (never multiple questions)
    # - Be colloquial and warm, like talking to a friend
    # - Start simple responses with "okay," "yeah," or "alright" for basic facts
    # - Show genuine empathy but don't overstate it - react naturally like someone just told you they're sad
    # - Don't just restate their pain - respond with brief, genuine concern like "I hope you're doing alright. That's terrible."
    
    # CURRENT CONTEXT:
    # - Intake Complete: {is_complete}
    # - Qualification Status: {qualification.get('qualification_status', 'incomplete')}
    # - Next Suggested Question: {next_question}
    
    # FIRM CONFIDENCE:
    # - Reassure them: "Our firm has handled thousands of cases like yours"
    # - Make them feel they're in the right place: "You made the right call contacting us"
    # - Show confidence: "We know exactly what we're doing with these situations"
    
    #      CONVERSATION FLOW:
    #  1. If is_complete is FALSE and next_action is continue: 
    #     - Use next_question as guidance but make it conversational
    #     - If multiple questions are unanswered, ask ONE clever question that covers multiple areas
    #     - Be slightly skeptical - ask for clarification if something seems vague or unclear
    #     - If qualified but incomplete: "This looks like a strong case, but I need a few more details for our attorneys

    #  2.  if is_complete is FALSE and next_action is complete and qualification_status is "qualified":
    #     - Express confidence in their case: "This sounds like exactly the kind of case we handle successfully"
    #     - Offer consultation: "Let's get you scheduled with one of our attorneys"
    #     - Ask for their preferred day/time for consultation
    #     - Be encouraging about their case prospects

    #  3.  if is_complete is FALSE and next_action is complete and qualification_status is "not_qualified":
    #     - keep on asking missing questions until all questions are answered or is_complete becomes true
        
    #  4. If is_complete is TRUE and qualification_status is "qualified":
    #     - Express confidence in their case: "This sounds like exactly the kind of case we handle successfully"
    #     - Offer consultation: "Let's get you scheduled with one of our attorneys"
    #     - Ask for their preferred day/time for consultation
    #     - Be encouraging about their case prospects
    
    #  5. If is_complete is TRUE and qualification_status is "not_qualified":
    #    - Stay empathetic: "I understand this is frustrating"
    #    - Explain that based on the information provided, this doesn't appear to be a personal injury case
    #    - Ask if there are injury aspects they haven't mentioned
    #    - Give them a chance to clarify before suggesting other resources
    
    #  6. If qualification_status is "existing": Handle like an existing client calling about their case
    
    # Remember: You're human-like, caring, but efficient. Make them feel heard and in good hands.
    # """
    #########################################################

    #########################################################
    # USE THIS FOR THE SYSTEM PROMPT
    #########################################################
    # Get law firm information for Luna
    # Available variables for system_prompt and full_context templates: 
    # {lawfirm_info} - always available
    # {caller_prompt_info} - only when existing_client is True
    # {existing_client} - boolean flag
    lawfirm_info = memory_board.get_lawfirm_info()
    
    # Get existing client information for Luna if existing client is True
    existing_client = memory_board.get_existing_client()
    caller_prompt_info = {}
    if existing_client:
        caller_prompt_info = memory_board.get_caller_prompt_info()
    #########################################################
    
    # Get system prompt from memory board
    #########################################################
    system_prompt_template = memory_board.get_luna_config().get("system_prompt", "")
    qualification_status = qualification.get('qualification_status', 'incomplete')
    system_prompt = system_prompt_template.format(
        lawfirm_info=lawfirm_info,
        caller_prompt_info=caller_prompt_info,
        is_complete=is_complete,
        qualification_status=qualification_status,
        next_question=next_question
    )
    #########################################################

    #########################################################
    # # Prepare full context for Luna
    # full_context = f"""
    # Current Memory State:
    # {json.dumps(memory_data, indent=2)}
    
    # Latest User Input: {input_text}
    
    # Respond appropriately based on the context and current status.
    # """
    #########################################################

    # Get full context from memory board
    #########################################################
    full_context_template = memory_board.get_luna_config().get("full_context", "")
    memory_data_json = json.dumps(memory_data, indent=2)
    full_context = full_context_template.format(
        memory_data_json=memory_data_json,
        input_text=input_text
    )
    #########################################################

    try:
        # Retry with exponential backoff for Luna
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Process request using OpenAI streaming with full context
                full_response = ""
                
                for chunk in process_openai_request_streaming(
                    prompt=full_context,
                    system_prompt=system_prompt,
                    model_name=model,
                    params=params
                ):
                    full_response += chunk
                    yield chunk  # Stream the chunk to the user
                
                # Add Luna's complete response to transcript after streaming is done
                transcript = memory_board.get("transcript") or []
                transcript.append({"role": "assistant", "content": full_response})
                memory_board.update("transcript", transcript)
                
                logger.info("Luna processed user input successfully")
                return  # End the generator
                
            except Exception as e:
                # If this is the last attempt, yield fallback message
                if attempt == max_retries - 1:
                    logger.error(f"Luna error after {max_retries} attempts: {e}")
                    yield "I apologize, but I'm having trouble processing your request right now. Could you please try again?"
                    return
                
                # Calculate exponential backoff with jitter
                base_delay = 2 ** attempt  # 1s, 2s, 4s
                jitter = random.uniform(0.1, 0.3)  # Add 10-30% jitter
                delay = base_delay + jitter
                
                logger.warning(f"Luna request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
    except Exception as e:
        logger.error(f"Luna error: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Could you please try again?"

# dont care for now
#==============================================================
async def luna_streaming_async(
    input_text: str, 
    model: str = "gpt-4",
    params: Optional[ModelParams] = None
):
    """
    Luna - Async version for TTS integration.
    Real-time streaming conversational agent that talks to the user.
    
    Args:
        input_text (str): User's input text
        model (str): OpenAI model to use
        params (Optional[ModelParams]): Model parameters
    
    Yields:
        str: Streaming response chunks
    """
    if params is None:
        params = ModelParams(temperature=0.7, max_tokens=500)
    
    # Get current memory state for context
    memory_data = memory_board.get_all()
    
    system_prompt = """
    You are Luna, a warm, professional legal intake specialist for a personal injury law firm.
    You are the human face of the firm - empathetic, knowledgeable, and trustworthy.
    
    Your personality:
    - Warm and caring but professional 
    - Confident in your expertise
    - Slightly maternal/protective tone
    - Use "dear", "honey" occasionally but not excessively
    - Show genuine concern for their situation
    - Be encouraging about their case prospects when appropriate
    
    Key communication principles:
    - Listen actively and acknowledge their pain/frustration
    - Make them feel they're in the right place: "You made the right call contacting us"
    - Show confidence: "We know exactly what we're doing with these situations"
    
         CONVERSATION FLOW:
     1. If is_complete is FALSE: 
        - Use next_question as guidance but make it conversational
        - If multiple questions are unanswered, ask ONE clever question that covers multiple areas
        - Be slightly skeptical - ask for clarification if something seems vague or unclear
        - If qualified but incomplete: "This looks like a strong case, but I need a few more details for our attorneys"
     
     2. If is_complete is TRUE and qualification_status is "qualified":
        - Express confidence in their case: "This sounds like exactly the kind of case we handle successfully"
        - Offer consultation: "Let's get you scheduled with one of our attorneys"
        - Ask for their preferred day/time for consultation
        - Be encouraging about their case prospects
    
    3. If is_complete is TRUE and qualification_status is "not_qualified":
       - Stay empathetic: "I understand this is frustrating"
       - Explain that based on the information provided, this doesn't appear to be a personal injury case
       - Ask if there are injury aspects they haven't mentioned
       - Give them a chance to clarify before suggesting other resources
    
    4. If qualification_status is "existing": Handle like an existing client calling about their case
    
    Remember: You're human-like, caring, but efficient. Make them feel heard and in good hands.
    """
    
    # Prepare full context for Luna
    full_context = f"""
    Current Memory State:
    {json.dumps(memory_data, indent=2)}
    
    Latest User Input: {input_text}
    
    Respond appropriately based on the context and current status.
    """
    
    try:
        # Retry with exponential backoff for Luna
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Process request using OpenAI streaming with full context
                full_response = ""
                
                # Convert sync generator to async generator
                for chunk in process_openai_request_streaming(
                    prompt=full_context,
                    system_prompt=system_prompt,
                    model_name=model,
                    params=params
                ):
                    full_response += chunk
                    yield chunk  # Stream the chunk to the user
                    await asyncio.sleep(0.01)  # Small delay for async cooperation
                
                # Add Luna's complete response to transcript after streaming is done
                transcript = memory_board.get("transcript") or []
                transcript.append({"role": "assistant", "content": full_response})
                memory_board.update("transcript", transcript)
                
                logger.info("Luna processed user input successfully")
                return  # End the generator
                
            except Exception as e:
                # If this is the last attempt, yield fallback message
                if attempt == max_retries - 1:
                    logger.error(f"Luna error after {max_retries} attempts: {e}")
                    yield "I apologize, but I'm having trouble processing your request right now. Could you please try again?"
                    return
                
                # Calculate exponential backoff with jitter
                base_delay = 2 ** attempt  # 1s, 2s, 4s
                jitter = random.uniform(0.1, 0.3)  # Add 10-30% jitter
                delay = base_delay + jitter
                
                logger.warning(f"Luna request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
    except Exception as e:
        logger.error(f"Luna error: {e}")
        yield "I apologize, but I'm having trouble processing your request right now. Could you please try again?"
#==============================================================

def agent_0_classifier(
    input_text: str,
    model: str = "gpt-3.5-turbo",
    params: Optional[ModelParams] = None
) -> Dict[str, Any]:
    """
    Agent 0 - Classifies caller type and determines conversation path.
    
    Args:
        input_text (str): Input text to classify
        model (str): OpenAI model to use
        params (Optional[ModelParams]): Model parameters
    
    Returns:
        Dict[str, Any]: Classification results
    """
    if params is None:
        params = ModelParams(temperature=0.3, max_tokens=200)
    
    #########################################################
    # system_prompt = """
    # You are a client classification specialist. Your ONLY job is to determine if this is a NEW client or EXISTING client.

    # EXISTING CLIENT indicators (must be explicitly mentioned):
    # - "my case" / "my attorney" / "my lawyer" 
    # - "I'm calling about my case"
    # - "update on my case" / "checking on my case"
    # - "I already have a case with you"
    # - "I'm an existing client"
    # - References case numbers, file numbers, or attorney names
    # - "following up on..." with legal context

    # NEW CLIENT indicators:
    # - Describes a recent incident/accident
    # - "I need a lawyer" / "should I call a lawyer"
    # - "I was in an accident" / "I was injured" 
    # - Asking if they have a case
    # - First time describing their situation
    # - "someone told me to call a lawyer"

    # CRITICAL RULES:
    # - Default to "new" unless EXPLICITLY stated they are existing
    # - Having lots of details about an incident does NOT make them existing
    # - Being prepared or organized does NOT make them existing
    # - Knowing legal terms does NOT make them existing
    # - ONLY mark as "existing" if they clearly reference an ongoing case with this firm

    # Examples:
    # - "I was in a car accident and have photos" → NEW (describing incident)
    # - "I'm calling about my car accident case" → EXISTING (references ongoing case)
    # - "I have all my insurance information ready" → NEW (being prepared ≠ existing)
    # - "My attorney said to call about my case" → EXISTING (references ongoing case)

    # Respond in JSON format:
    # {
    #     "caller_type": "new" or "existing", 
    #     "confidence": 0.0-1.0,
    #     "reasoning": "specific evidence from transcript that supports this classification"
    # }
    # """
    #########################################################

    #########################################################
    # USE THIS FOR THE SYSTEM PROMPT
    #########################################################
    # Get law firm information for Agent 0
    # Available variables for system_prompt and full_context templates: {lawfirm_info}
    lawfirm_info = memory_board.get_lawfirm_info()
    #########################################################

    # Get system prompt from memory board
    #########################################################
    system_prompt_template = memory_board.get_agent_0_config().get("system_prompt", "")
    system_prompt = system_prompt_template  # Agent 0 doesn't need variable substitution
    #########################################################

    # Check if existing client is hardcoded
    existing_client = memory_board.get_existing_client()
    if existing_client:
        # Hardcode classification to "existing" when existing_client is True
        hardcoded_classification = {
            "caller_type": "existing",
            "confidence": 1.0,
            "reasoning": "Hardcoded as existing client based on existing_client flag"
        }
        memory_board.update("caller_type", "existing")
        logger.info("Agent 0 hardcoded classification to 'existing' based on existing_client flag")
        return hardcoded_classification
    
    # Get full transcript from memory board for classification
    transcript = memory_board.get("transcript") or []

    #########################################################
    # full_context = f"Full conversation transcript:\n{json.dumps(transcript)}\n\nLatest input: {input_text}"
    #########################################################

    # Get full context from memory board
    #########################################################
    full_context_template = memory_board.get_agent_0_config().get("full_context", "")
    transcript_json = json.dumps(transcript)
    full_context = full_context_template.format(
        transcript_json=transcript_json,
        input_text=input_text
    )
    #########################################################

    # Retry with exponential backoff for Agent 0
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = process_openai_request(
                prompt=full_context,
                system_prompt=system_prompt,
                model_name=model,
                params=params
            )
            
            # Parse JSON response
            classification = json.loads(response["content"])
            
            # SMART MAJORITY-BASED QUESTIONNAIRE LOADING
            caller_type = classification.get("caller_type")
            
            # Check if we should load questionnaire based on majority logic
            should_load = memory_board.should_load_questionnaire(caller_type)
            
            if should_load:
                # Update caller type and load questionnaire
                memory_board.update("caller_type", caller_type)
                
                # Load appropriate questionnaire based on caller type
                if caller_type == "new":
                    logger.info("🔄 Loading NEW client questionnaire (majority changed)")
                    # memory_board.load_questionnaire("new_client_questionaire.json")
                    memory_board.update("questionnaire", memory_board.get_new_client_questionnaire())
                elif caller_type == "existing":
                    logger.info("🔄 Loading EXISTING client questionnaire (majority changed)")
                    # memory_board.load_questionnaire("existing_client_questionaire.json")
                    memory_board.update("questionnaire", memory_board.get_existing_client_questionnaire())
                else:
                    logger.warning(f"Unknown caller type: {caller_type}, keeping current questionnaire")
            else:
                # Don't update caller_type or load questionnaire - preserve existing state
                logger.info(f"🔒 Questionnaire preserved! Current caller type: {memory_board.get('caller_type')}")
                logger.info(f"🔒 Classified as: {caller_type} but majority unchanged")
            
            logger.info(f"Agent 0 classified caller: {classification}")
            return classification
            
        except Exception as e:
            # If this is the last attempt, return default classification
            if attempt == max_retries - 1:
                logger.error(f"Agent 0 error after {max_retries} attempts: {e}")
                return {"caller_type": "unknown", "confidence": 0.0, "reasoning": "no specific evidence from transcript"}
            
            # Calculate exponential backoff with jitter
            base_delay = 2 ** attempt  # 1s, 2s, 4s
            jitter = random.uniform(0.1, 0.3)  # Add 10-30% jitter
            delay = base_delay + jitter
            
            logger.warning(f"Agent 0 request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
            time.sleep(delay)
    
    # This should never be reached due to the return in the last attempt
    return {"caller_type": "unknown", "confidence": 0.0, "reasoning": "no specific evidence from transcript"}


def agent_1_form_filler(
    input_text: str,
    model: str = "gpt-3.5-turbo",
    params: Optional[ModelParams] = None
) -> Dict[str, Any]:
    """
    Agent 1 - Fills out questionnaire forms based on conversation.
    
    Args:
        input_text (str): Input text to extract information from
        model (str): OpenAI model to use
        params (Optional[ModelParams]): Model parameters
    
    Returns:
        Dict[str, Any]: Extracted form data
    """
    if params is None:
        params = ModelParams(temperature=0.2, max_tokens=300)
    
    #########################################################
    # system_prompt = """
    # You are a form-filling agent for legal intake questionnaire for the following law firm: {lawfirm_info}
    #
    
    # Your job:
    # 1. Review the questionnaire questions and their current answers
    # 2. Check the conversation transcript for new information
    # 3. ONLY update/add answers for questions where you found relevant info in the transcript
    # 4. Suggest the next question to ask
    
    # IMPORTANT RULES:
    # - Include ALL questions that you can answer from the transcript (both new and existing)
    # - This ensures consistent questionnaire state across multiple runs
    # - Do NOT include questions you cannot answer from the transcript
    # - Use proper JSON formatting with escaped quotes
    
    # Return EXACTLY this JSON format (no extra text, no markdown):
    # {
    #     "answers": [
    #         {"question_number": 1, "answer": "answer from transcript"},
    #         {"question_number": 3, "answer": "another answer from transcript"}
    #     ],
    #     "next_question": "What specific question should Luna ask next?"
    # }
    
    # JSON Rules:
    # - Use double quotes for all strings
    # - Escape any quotes inside answers with backslash
    # - No line breaks inside JSON strings
    # - No trailing commas
    # - Keep answers concise and factual
    
    # Additional Rules:
    # - Only include answers you can confidently extract from the transcript
    # - Skip questions that can't be answered from the conversation
    # - next_question should be a smart, natural question that can gather multiple missing pieces
    # - Combine multiple questionnaire items into one natural question when possible
    # - If all questions answered, set next_question to null
    # """
    #########################################################

    #########################################################
    # USE THIS FOR THE SYSTEM PROMPT
    #########################################################
    # Get law firm information for Agent 1
    # Available variables for system_prompt and full_context templates: {lawfirm_info}
    lawfirm_info = memory_board.get_lawfirm_info()
    #########################################################
    
    # Get system prompt from memory board
    #########################################################
    system_prompt_template = memory_board.get_agent_1_config().get("system_prompt", "")
    system_prompt = system_prompt_template.format(
        lawfirm_info=lawfirm_info
    )
    #########################################################

    try:
        # Get full transcript and current questionnaire from memory board
        transcript = memory_board.get("transcript") or []
        current_questionnaire = memory_board.get("questionnaire") or {}
        
        # Prepare clear context for Agent 1
        questions = current_questionnaire.get("questions", [])
        answered_count = sum(1 for q in questions if q.get("answer", "").strip())
        
        # Debug the counting
        logger.info(f"Agent 1 context - Total questions: {len(questions)}")
        logger.info(f"Agent 1 context - Answered count: {answered_count}")
        for i, q in enumerate(questions):
            answer = q.get("answer", "").strip()
            logger.info(f"Question {q.get('question_number', i+1)}: {'ANSWERED' if answer else 'EMPTY'} - '{answer[:50]}...' if len(answer) > 50 else answer")
        
        #########################################################
        # full_context = """QUESTIONNAIRE REVIEW TASK:

# Current questionnaire status: {answered_count}/{len(questions)} questions answered

# QUESTIONNAIRE STRUCTURE:
# {json.dumps(current_questionnaire, indent=2)}

# CONVERSATION TRANSCRIPT:
# {json.dumps(transcript, indent=2)}

# LATEST USER INPUT:
# {input_text}

# INSTRUCTIONS:
# - Review the transcript for information that can answer unanswered questions
# - Check if any existing answers can be improved with new information
# - Only return questions you want to UPDATE or ADD answers for
# - Do not return questions that are already well-answered unless you have better info
# """
        #########################################################

        # Get full context from memory board
        #########################################################
        full_context_template = memory_board.get_agent_1_config().get("full_context", "")
        current_questionnaire_json = json.dumps(current_questionnaire, indent=2)
        transcript_json = json.dumps(transcript, indent=2)
        full_context = full_context_template.format(
            answered_count=answered_count,
            len_questions=len(questions),
            current_questionnaire_json=current_questionnaire_json,
            transcript_json=transcript_json,
            input_text=input_text
        )
        #########################################################

        
        # Retry with exponential backoff for Agent 1
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = process_openai_request(
                    prompt=full_context,
                    system_prompt=system_prompt,
                    model_name=model,
                    params=params
                )
                logger.info(f"Agent 1 response: {response}")
                # Parse response data
                response_data = json.loads(response["content"])
                
                # Update questionnaire answers in memory board
                current_questionnaire = memory_board.get("questionnaire") or {}
                logger.info(f"Current questionnaire before update: {json.dumps(current_questionnaire, indent=2)}")
                
                # Update ONLY the questions that the LLM specifically mentioned
                if "answers" in response_data and response_data["answers"]:
                    questions = current_questionnaire.get("questions", [])
                    logger.info(f"Found {len(questions)} questions in questionnaire")
                    logger.info(f"Processing {len(response_data['answers'])} LLM-specified answers")
                    
                    # Only update questions that the LLM specifically returned
                    for answer_data in response_data["answers"]:
                        question_num = answer_data["question_number"]
                        answer_text = answer_data["answer"]
                        logger.info(f"LLM wants to update question {question_num}: {answer_text}")
                        
                        # Find and update ONLY this specific question
                        question_found = False
                        for question in questions:
                            if question.get("question_number") == question_num:
                                old_answer = question.get("answer", "")
                                question["answer"] = answer_text
                                question_found = True
                                logger.info(f"Updated question {question_num}: '{old_answer}' -> '{answer_text}'")
                                break
                        
                        if not question_found:
                            logger.warning(f"Question number {question_num} not found in questionnaire")
                    
                    logger.info("Only LLM-specified questions were updated, others preserved")
                else:
                    logger.info("No answers to update from LLM response")
                
                # Update next question suggestion (simple string)
                if "next_question" in response_data:
                    memory_board.update("next_question", response_data["next_question"])
                
                memory_board.update("questionnaire", current_questionnaire)
                logger.info(f"Updated questionnaire in memory board: {json.dumps(current_questionnaire, indent=2)}")
                
                logger.info("Agent 1 updated questionnaire answers")
                logger.info(f"Agent 1 response: {response_data}")
                return response_data
                
            except Exception as e:
                # If this is the last attempt, return current state to preserve questionnaire
                if attempt == max_retries - 1:
                    logger.error(f"Agent 1 error after {max_retries} attempts: {e}")
                    # Return current questionnaire state to avoid resetting progress
                    current_questionnaire = memory_board.get("questionnaire") or {}
                    return {
                        "answers": [],  # No new answers extracted due to error
                        "next_question": memory_board.get("next_question"),  # Keep current next question
                        "error": "Failed to extract answers from conversation",
                        "preserved_state": True
                    }
                
                # Calculate exponential backoff with jitter
                base_delay = 2 ** attempt  # 1s, 2s, 4s
                jitter = random.uniform(0.1, 0.3)  # Add 10-30% jitter
                delay = base_delay + jitter
                
                logger.warning(f"Agent 1 request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        # This should never be reached due to the return in the last attempt
        current_questionnaire = memory_board.get("questionnaire") or {}
        return {
            "answers": [],
            "next_question": memory_board.get("next_question"),
            "error": "Failed to extract answers from conversation",
            "preserved_state": True
        }
        
    except Exception as e:
        logger.error(f"Agent 1 error: {e}")
        # Preserve questionnaire state even on unexpected errors
        current_questionnaire = memory_board.get("questionnaire") or {}
        return {
            "answers": [],
            "next_question": memory_board.get("next_question"),
            "error": "Failed to extract answers from conversation",
            "preserved_state": True
        }


def agent_2_coordinator(
    input_text: str,
    model: str = "gpt-3.5-turbo",
    params: Optional[ModelParams] = None
) -> Dict[str, Any]:
    """
    Agent 2 - Checks questionnaire completion and evaluates candidate qualification.
    
    Args:
        input_text (str): Current conversation context
        model (str): OpenAI model to use
        params (Optional[ModelParams]): Model parameters
    
    Returns:
        Dict[str, Any]: Completion status and qualification decision
    """
    if params is None:
        params = ModelParams(temperature=0.2, max_tokens=300)
    
    # Get current questionnaire state
    questionnaire = memory_board.get("questionnaire") or {}
    questions = questionnaire.get("questions", [])
    
    # Check if all questions are answered
    unanswered_questions = []
    total_questions = len(questions)
    answered_questions = 0
    
    for question in questions:
        answer = question.get("answer", "").strip()
        if answer:
            answered_questions += 1
        else:
            unanswered_questions.append(question.get("question_number", 0))
    
    completion_percentage = int((answered_questions / total_questions * 100)) if total_questions > 0 else 0
    is_complete = len(unanswered_questions) == 0
    
    if is_complete:
        memory_board.update("is_complete", True)

    logger.info(f"Questionnaire status: {answered_questions}/{total_questions} answered ({completion_percentage}%)")
    
    # Check if caller type is existing - if so, hardcode qualification but still do completion checks
    caller_type = memory_board.get("caller_type")
    if caller_type == "existing":
        # Hardcode qualification as qualified for existing clients, but still check completion threshold
        completion_threshold = 85  # Need 85%+ completion before booking consultation
        
        if completion_percentage >= completion_threshold:
            # Qualified AND sufficiently complete - ready for consultation booking
            hardcoded_qualification = {
                "qualification_status": "qualified",
                "confidence": 1.0,
                "reasoning": "Existing clients are pre-qualified, questionnaire sufficiently complete",
                "red_flags": [],
                "strengths": ["Existing client", "Questionnaire complete"],
                "should_continue": False,
                "next_action": "complete",
                "completion_percentage": completion_percentage,
                "missing_questions": unanswered_questions
            }
            # is_complete should only be True when ALL questions are answered (100%)
            memory_board.update("is_complete", is_complete)
            logger.info(f"Existing client qualified and {completion_percentage}% complete - ready for consultation")
        else:
            # Qualified but need more completion - continue gathering information
            hardcoded_qualification = {
                "qualification_status": "qualified",
                "confidence": 1.0,
                "reasoning": "Existing clients are pre-qualified, but need more questionnaire completion",
                "red_flags": [],
                "strengths": ["Existing client"],
                "should_continue": True,
                "next_action": "continue",
                "completion_percentage": completion_percentage,
                "missing_questions": unanswered_questions
            }
            memory_board.update("is_complete", is_complete)
            logger.info(f"Existing client qualified but only {completion_percentage}% complete - continuing intake")
        
        # Update global memory with hardcoded values
        memory_board.update("should_continue", hardcoded_qualification["should_continue"])
        memory_board.update("qualification", hardcoded_qualification)
        
        logger.info("Agent 2 hardcoded qualification as 'qualified' for existing client")
        return hardcoded_qualification
    
    # ALWAYS run qualification check - even with incomplete questionnaire
    # This helps catch cases the firm doesn't handle early
    
    # Evaluate qualification based on available information (complete or partial)

    #########################################################
    # system_prompt = f"""
    # You are a legal case qualification expert for the following law firm: {lawfirm_info}
    
    # CRITICAL: Base your decision ONLY on:
    # 1. What the caller actually said in the transcript
    # 2. The questionnaire answers provided
    
    # DO NOT use your general knowledge or make assumptions beyond what is explicitly stated.
    
    # QUESTIONNAIRE STATUS: {answered_questions}/{total_questions} answered ({completion_percentage}%)
    
    # STRICT QUALIFICATION RULES:
    # - If less than 50% of questions answered, ALWAYS return "incomplete"
    # - ONLY return "not_qualified" if transcript EXPLICITLY rules out personal injury
    # - If transcript mentions ANY injury, assault, accident, or harm, return "incomplete" or "qualified"
    # - Every potential client is valuable - err on the side of keeping them engaged
    # - Examples to CONTINUE: immigration + assault, employment + injury, any mention of harm
    
    # Consider ONLY these factors from the actual answers:
    # 1. Severity of injury (medical treatment, ongoing pain)
    # 2. Clear liability (other party at fault)
    # 3. Insurance coverage available
    # 4. Damages (medical bills, lost wages, property damage)
    # 5. Timeline (statute of limitations concerns)
    # 6. Evidence available (witnesses, photos, police reports)
    
    # Respond in JSON format:
    # {{
    #     "qualification_status": "qualified" or "not_qualified" or "incomplete",
    #     "confidence": 0.0-1.0,
    #     "reasoning": "Brief explanation based ONLY on transcript/answers",
    #     "red_flags": ["list", "of", "concerns"],
    #     "strengths": ["list", "of", "positive", "factors"],
    #     "should_continue": true or false
    # }}
    # """
    #########################################################

    #########################################################
    # USE THIS FOR THE SYSTEM PROMPT
    #########################################################
    # Get law firm information for Agent 2
    # Available variables for system_prompt and full_context templates: {lawfirm_info}
    lawfirm_info = memory_board.get_lawfirm_info()
    #########################################################
    
    # Get system prompt from memory board
    #########################################################
    system_prompt_template = memory_board.get_agent_2_config().get("system_prompt", "")
    system_prompt = system_prompt_template.format(
        lawfirm_info=lawfirm_info,
        answered_questions=answered_questions,
        total_questions=total_questions,
        completion_percentage=completion_percentage
    )
    #########################################################

    try:
        # Prepare context with all questionnaire data
        #########################################################
        # full_context = f"""
        # Completed questionnaire for qualification evaluation:
        # {json.dumps(questionnaire, indent=2)}
        
        # Latest input: {input_text}
        
        # Evaluate qualification for legal consultation.
        # """
        #########################################################

        # Get full context from memory board
        #########################################################
        full_context_template = memory_board.get_agent_2_config().get("full_context", "")
        questionnaire_json = json.dumps(questionnaire, indent=2)
        full_context = full_context_template.format(
            questionnaire_json=questionnaire_json,
            input_text=input_text
        )
        #########################################################

        # Retry with exponential backoff for Agent 2
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = process_openai_request(
                    prompt=full_context,
                    system_prompt=system_prompt,
                    model_name=model,
                    params=params
                )
                
                qualification = json.loads(response["content"])
                
                # Determine next action based on qualification status
                qualification_status = qualification.get("qualification_status", "incomplete")
                should_continue = qualification.get("should_continue", True)
                
                # Update global memory with should_continue value
                memory_board.update("should_continue", should_continue)
                
                # Determine completion threshold - most questions should be answered
                completion_threshold = 85  # Need 85%+ completion before booking consultation
                
                if qualification_status == "not_qualified":
                    # Early disqualification - stop asking questions
                    qualification.update({
                        "next_action": "complete",
                        "completion_percentage": completion_percentage,
                        "missing_questions": unanswered_questions
                    })
                    memory_board.update("is_complete", True)
                elif qualification_status == "qualified" and completion_percentage >= completion_threshold:
                    # Qualified AND sufficiently complete - ready for consultation booking
                    qualification.update({
                        "next_action": "complete", 
                        "completion_percentage": completion_percentage,
                        "missing_questions": unanswered_questions
                    })
                    memory_board.update("is_complete", True)
                    logger.info(f"Case qualified and {completion_percentage}% complete - ready for consultation")
                else:
                    # Continue gathering information (not qualified, incomplete, or qualified but need more info)
                    qualification.update({
                        "next_action": "continue",
                        "completion_percentage": completion_percentage,
                        "missing_questions": unanswered_questions
                    })
                    memory_board.update("is_complete", False)
                    
                    if qualification_status == "qualified":
                        logger.info(f"Case qualified but only {completion_percentage}% complete - continuing intake")
                
                # Store qualification results in global memory
                memory_board.update("qualification", qualification)
                
                logger.info(f"Agent 2 qualification: {qualification_status} ({qualification.get('confidence', 0):.2f} confidence) - Action: {qualification['next_action']}")
                return qualification
                
            except Exception as e:
                # If this is the last attempt, return continue status
                if attempt == max_retries - 1:
                    logger.error(f"Agent 2 error after {max_retries} attempts: {e}")
                    return {
                        "next_action": "continue",
                        "completion_percentage": completion_percentage,
                        "missing_questions": [],
                        "qualification_status": "error",
                        "suggested_question": "keep the conversation on there is some error in proccessing so keep the conversation on in the meantime?"
                    }
                
                # Calculate exponential backoff with jitter
                base_delay = 2 ** attempt  # 1s, 2s, 4s
                jitter = random.uniform(0.1, 0.3)  # Add 10-30% jitter
                delay = base_delay + jitter
                
                logger.warning(f"Agent 2 request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        # This should never be reached due to the return in the last attempt
        return {
            "next_action": "continue",
            "completion_percentage": completion_percentage,
            "missing_questions": [],
            "qualification_status": "error",
            "suggested_question": "keep the conversation on there is some error in proccessing so keep the conversation on in the meantime?"
        }
        
    except Exception as e:
        logger.error(f"Agent 2 error: {e}")
        return {
            "next_action": "continue",
            "completion_percentage": completion_percentage,
            "missing_questions": [],
            "qualification_status": "error",
            "suggested_question": "keep the conversation on there is some error in proccessing so keep the conversation on in the meantime?"
        }

## checked until here
def process_parallel_agents(input_text: str) -> Dict[str, Any]:
    """
    Process input through multiple agents with conditional execution.
    - Agent 0 always runs (classifier)
    - Agent 1 & 2 run for both NEW and EXISTING clients
    - Agent 2 uses hardcoded qualification for existing clients but still checks completion
    
    Args:
        input_text (str): User input to process
    
    Returns:
        Dict[str, Any]: Combined results from all agents
    """
    results = {}
    
    # Always run Agent 0 first to classify caller type
    results["classification"] = agent_0_classifier(input_text)
    
    # Get caller type from classification result
    caller_type = results["classification"].get("caller_type", "unknown")
    
    if caller_type == "new":
        # NEW CLIENT: Run Agent 1 and Agent 2 for questionnaire processing
        logger.info("New client detected - running questionnaire agents")
        
        # Run Agent 1 (form filler)
        results["form_data"] = agent_1_form_filler(input_text)
        
        # Run Agent 2 after Agent 1 completes
        results["coordination"] = agent_2_coordinator(input_text)
        
        logger.info("New client processing completed (Agent 0 -> Agent 1 -> Agent 2)")
        
    elif caller_type == "existing":
        # EXISTING CLIENT: Run Agent 1 and Agent 2 (hardcoded qualification but completion checks)
        logger.info("Existing client detected - running questionnaire agents")
        
        # Run Agent 1 (form filler) for existing clients too
        results["form_data"] = agent_1_form_filler(input_text)
        
        # Run Agent 2 after Agent 1 completes (hardcoded qualification but completion checks)
        results["coordination"] = agent_2_coordinator(input_text)
        
        logger.info("Existing client processing completed (Agent 0 -> Agent 1 -> Agent 2)")
        
    else:
        # UNKNOWN CALLER TYPE: Run agents as fallback
        logger.warning(f"Unknown caller type '{caller_type}' - running all agents as fallback")
        
        results["form_data"] = agent_1_form_filler(input_text)
        results["coordination"] = agent_2_coordinator(input_text)
        
        logger.info("Unknown caller processing completed (Agent 0 -> Agent 1 -> Agent 2)")
    
    return results


def main_system_streaming(user_input: str):
    """
    Main system function - coordinates all agents and streams Luna's response in real-time.
    Luna starts streaming immediately while agents run in background.
    
    Args:
        user_input (str): User's input text
    
    Yields:
        str: Individual characters/tokens as they stream in real-time
    """
    try:

        #########################################################
        # APPROACH 1: BLOCKING - Wait for agents to finish before Luna streams
        # Pro: Agents definitely complete before Luna responds
        # Con: Slower response time (700ms -> 1500ms), user waits longer
        # Use case: When you need agents to finish before responding
        #########################################################
        # # Start agents in background (non-blocking)
        # with ThreadPoolExecutor(max_workers=1) as executor:
        #     # Submit agent processing as background task
        #     future_agents = executor.submit(process_parallel_agents, user_input)
            
        #     # print("Luna starts strezaming immediately (doesn't wait for agents): ", user_input)
        #     # Luna starts streaming immediately (doesn't wait for agents)
        #     for chunk in luna_streaming(user_input):
        #         yield chunk
            
        #     # Agents should be done by now, but wait if needed
        #     agent_results = future_agents.result()
        
        #########################################################
        # APPROACH 2: NON-BLOCKING - Luna streams immediately, agents finish in background
        # Pro: Fastest response time (700ms), Luna streams immediately
        # Con: Agents might not finish before function returns, potential race conditions
        # Use case: When you want immediate response and agents can finish later
        # CURRENT APPROACH: Chosen for best user experience
        #########################################################
        # Start agents in background (non-blocking) using daemon thread
        import threading
        
        def run_agents():
            """Background task to run agents with error handling."""
            try:
                process_parallel_agents(user_input)
            except Exception as e:
                logger.error(f"Background agent processing error: {e}")
        
        agent_thread = threading.Thread(
            target=run_agents,
            daemon=False  # Let thread finish processing even if main function exits ## important ##
        )
        agent_thread.start()
        
        # Luna streams immediately (doesn't wait for agents at all)
        for chunk in luna_streaming(user_input):
            yield chunk
        #########################################################

        #########################################################
        # APPROACH 3: SEMI-BLOCKING - Luna streams immediately, but wait for agents before function exits
        # Pro: Luna streams immediately + agents guaranteed to finish
        # Con: Function doesn't return until agents complete, blocks next request
        # Use case: When you want immediate streaming but need agents to complete before handling next request
        #########################################################
        # agent_thread.join()  # Uncomment to wait for agents to finish before function exits
        #########################################################

    except Exception as e:
        logger.error(f"Main system error: {e}")
        yield "I'm sorry, there was an error processing your request. Please try again."


def main_system(user_input: str) -> str:
    """
    Main system function - coordinates all agents and returns Luna's response.
    (Non-streaming version for backward compatibility)
    
    Args:
        user_input (str): User's input text
    
    Returns:
        str: Luna's complete response
    """
    try:
        # Collect all streaming chunks into a single response
        response_chunks = []
        for chunk in main_system_streaming(user_input):
            response_chunks.append(chunk)
        
        return "".join(response_chunks)
        
    except Exception as e:
        logger.error(f"Main system error: {e}")
        return "I'm sorry, there was an error processing your request. Please try again."


def initialize_system() -> None:
    """Initialize the system - setup OpenAI client and memory board."""
    try:
        setup_openai_client()
        memory_board.update("session_id", f"session_{int(time.time())}")
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"System initialization error: {e}")
        raise

# Example usage and testing
def example_usage() -> None:
    """Example demonstrating how to use the agent system."""
    
    # Initialize system
    initialize_system()
    
    # Example conversation
    test_inputs = [
        "Hi, I was in a car accident last week and I'm injured",
        "It happened on Main Street. The other driver ran a red light",
        "I have back pain and my car is totaled. What should I do?"
    ]
    
    print("=== Legal Intake System Demo ===\n")
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"User {i}: {user_input}")
        response = main_system(user_input)
        print(f"Luna: {response}\n")
        print("-" * 50)
    
    # Show final memory board state
    print("\nFinal Memory Board State:")
    print(json.dumps(memory_board.get_all(), indent=2))


if __name__ == "__main__":
    example_usage() 