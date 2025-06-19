#!/usr/bin/env python3
# Azure Foundry Agent Chat with Group Chat Orchestration

import asyncio
import sys
import os
import re
import traceback
import time
import uuid
import json
from typing import List, Dict, Any, Optional, Union, Callable
from functools import lru_cache
from azure.identity import DefaultAzureCredential
from semantic_kernel.agents import Agent, ChatCompletionAgent, GroupChatOrchestration
from semantic_kernel.agents.orchestration.group_chat import BooleanResult, GroupChatManager, MessageResult, StringResult
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
from semantic_kernel.agents.azure_ai.azure_ai_agent_settings import AzureAIAgentSettings
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig

# Copilot Studio imports
from copilot_studio_agent import CopilotAgent
from copilot_studio_agent_thread import CopilotAgentThread
from copilot_studio_channel import CopilotStudioAgentChannel
from copilot_studio_message_content import CopilotMessageContent
from directline_client import DirectLineClient

# Enhanced Agent Selection System
from enhanced_agent_selector_fixed import EnhancedAgentSelector, RoutingDecision

# Add src folder to Python path for Microsoft Graph integration
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Microsoft Graph imports from src folder
try:
    from graph_agent_plugin import MicrosoftGraphPlugin
    from graph_agent import GraphAgent
    GRAPH_INTEGRATION_AVAILABLE = True
    print("✅ Microsoft Graph integration loaded successfully")
except ImportError as e:
    if "rich" in str(e):
        print("⚠️ Microsoft Graph integration needs 'rich' module: pip install rich")
    else:
        print(f"⚠️ Microsoft Graph integration not available: {e}")
    GRAPH_INTEGRATION_AVAILABLE = False

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class AzureFoundryOrchestration:
    """
    Enhanced Azure Foundry Agent Chatbot using Group Chat Orchestration Pattern
    
    This implementation combines your specialized Azure Foundry agent with a general
    chat completion agent, orchestrated through a smart group chat manager.
    """

    def __init__(self, endpoint: str, api_key: str, agent_id: str, model_name: str = "gpt-4o", 
                 bot_secret: str = None):
        """
        Initialize the Azure Foundry Orchestration
        
        Args:
            endpoint: Your Azure Foundry project endpoint URL (can be foundry or cognitive services)
            api_key: Your Azure Foundry API key 
            agent_id: Your specific agent ID to retrieve
            model_name: The model deployment name (default: gpt-4o)
            bot_secret: The Copilot Studio bot secret for DirectLine API
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.agent_id = agent_id
        self.model_name = model_name
        self.bot_secret = bot_secret
        
        # Initialize Azure Chat Completion for the orchestration manager
        self.chat_service = self._init_chat_service()
        
        # Initialize Copilot Studio DirectLine client if bot_secret is provided
        self.directline_client = None
        if bot_secret:
            self.directline_client = DirectLineClient(
                copilot_agent_secret=bot_secret,
                directline_endpoint="https://europe.directline.botframework.com/v3/directline"
            )
        
        # Create agents
        self.agents = []
        self.foundry_agent = None
        self.runtime = None
        
        print(f"✓ Azure Foundry Orchestration initialized")
        print(f"✓ Agent ID: {agent_id}")
        print(f"✓ Model: {model_name}")
        print(f"✓ Endpoint: {endpoint}")
        if bot_secret:
            print(f"✓ Copilot Studio agent enabled")

    def _init_chat_service(self) -> AzureChatCompletion:
        """Initialize the Azure Chat Completion service"""
        try:
            print("🔄 Initializing Azure Chat Completion service...")
            
            # Try multiple approaches for service initialization
            chat_service = None
            
            # Approach 1: Try with new Cognitive Services endpoint
            try:
                cognitive_services_endpoint = "https://aif-e2edemo.cognitiveservices.azure.com/"
                
                chat_service = AzureChatCompletion(
                    endpoint=cognitive_services_endpoint,
                    api_key=self.api_key,
                    deployment_name=self.model_name
                )
                print("✅ Azure Chat Completion service initialized with Cognitive Services endpoint")
                
            except Exception as e1:
                print(f"❌ Cognitive Services endpoint approach failed: {e1}")
                
                # Approach 2: Try with original Azure OpenAI endpoint
                try:
                    foundry_openai_endpoint = "https://aif-e2edemo.openai.azure.com/"
                    
                    chat_service = AzureChatCompletion(
                        endpoint=foundry_openai_endpoint,
                        api_key=self.api_key,
                        deployment_name=self.model_name
                    )
                    print("✅ Azure Chat Completion service initialized with OpenAI endpoint")
                    
                except Exception as e2:
                    print(f"❌ OpenAI endpoint approach failed: {e2}")
                    
                    # Approach 3: Try with DefaultAzureCredential
                    try:
                        print("🔍 Trying Azure credential approach...")
                        credential = DefaultAzureCredential()
                        
                        chat_service = AzureChatCompletion(
                            endpoint=cognitive_services_endpoint,
                            ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token,
                            deployment_name=self.model_name
                        )
                        print("✅ Azure Chat Completion service initialized with Azure credentials")
                        
                    except Exception as e3:
                        print(f"❌ All approaches failed. Last error: {e3}")
                        raise e3
            
            if chat_service is None:
                raise Exception("Failed to initialize chat service with any approach")
                
            return chat_service
            
        except Exception as e:
            print(f"❌ Failed to initialize chat service: {e}")
            print("⚠️ This might be due to authentication or endpoint configuration issues")
            raise

    def create_general_agent(self) -> ChatCompletionAgent:
        """Create a general conversation agent with synthesis capabilities"""
        return ChatCompletionAgent(
            name="GeneralAssistant",
            description="A helpful general-purpose AI assistant and synthesis coordinator",
            instructions=(
                "You are a helpful, friendly AI assistant with dual capabilities:\n\n"
                "🔹 **General Assistance Mode**: Engage in general conversation, "
                "answer questions, and provide assistance on a wide range of topics. "
                "When in group discussions, you provide accessible, clear explanations and "
                "help facilitate productive conversations. You complement specialized agents "
                "by offering broader perspective and clarifying complex topics for users.\n\n"
                
                "🔹 **Synthesis Mode**: When you detect that multiple agents have completed "
                "related tasks (indicated by task completion context), act as a synthesis coordinator. "
                "Analyze all the individual agent responses, identify connections and complementary insights, "
                "and provide a comprehensive, integrated answer that combines the best of all responses.\n\n"
                
                "**Synthesis Guidelines:**\n"
                "• Look for completed tasks in the conversation history\n"
                "• Identify how different agent responses complement each other\n"
                "• Create connections between separate pieces of advice or information\n"
                "• Provide a cohesive, integrated final answer\n"
                "• Highlight key insights from each specialist agent\n\n"
                
                "**Response Length Guidelines:**\n"
                "• Keep responses concise and focused (aim for 200-500 words)\n"
                "• For synthesis: provide structured summary with key points\n"
                "• For general questions: provide direct, helpful answers\n"
                "• Use bullet points and clear formatting for readability\n"
                "• End with actionable next steps or conclusions\n\n"
                
                "Example synthesis approach:\n"
                "- Career Agent provided job search tips → Extract key strategies\n"
                "- Philosophy response about life meaning → Extract key insights\n"
                "- Combine both: Show how career fulfillment connects to life purpose\n"
                "- Provide integrated guidance that addresses both aspects holistically"
            ),
            service=self.chat_service,
        )

    async def create_transaction_analysis_agent(self):
        """Create a Transaction Analysis specialist agent using actual Azure Foundry Agent"""
        try:
            # Import required classes for Azure AI Agent integration
            from azure.identity import DefaultAzureCredential
            from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
            from semantic_kernel.agents.azure_ai.azure_ai_agent_settings import AzureAIAgentSettings
            from semantic_kernel.agents import AzureAIAgentThread
            
            print("🔄 Creating Transaction Analysis Agent with direct Azure Foundry integration...")
            
            # Create Azure AI Agent settings for the transaction analysis agent
            ai_agent_settings = AzureAIAgentSettings(
                endpoint=self.endpoint,
                model_deployment_name=self.model_name
            )

            # Initialize with DefaultAzureCredential and get existing agent
            credential = DefaultAzureCredential()
            
            # Create client for the Azure AI Agent
            client = AzureAIAgent.create_client(
                credential=credential,
                endpoint=ai_agent_settings.endpoint,
                api_version=ai_agent_settings.api_version,
            )
            
            # Get the existing agent definition using the agent_id
            agent_definition = await client.agents.get_agent(self.agent_id)
            
            # Fix the agent name to match the required pattern
            if hasattr(agent_definition, 'name') and agent_definition.name:
                import re
                agent_definition.name = re.sub(r'[^0-9A-Za-z_-]', '_', agent_definition.name)
            
            # Create the actual Azure AI Agent
            azure_ai_agent = AzureAIAgent(
                client=client,
                definition=agent_definition,
            )
            
            # Ensure the agent has a description for GroupChatOrchestration
            if not hasattr(azure_ai_agent, 'description') or not azure_ai_agent.description:
                azure_ai_agent.description = "Your role is to process structured transaction data provided by the user."
            
            # Store the actual agent name for selection logic
            self.transaction_agent_name = azure_ai_agent.name or "TransactionAgent"
            
            print(f"✅ TransactionAgent '{self.transaction_agent_name}' ready")
            
            # Return the actual Azure AI Agent (not a wrapper)
            return azure_ai_agent
            
        except Exception as e:
            print(f"❌ Failed to create TransactionAgent: {e}")
            print("🔄 Falling back to ChatCompletionAgent...")
            
            # Fallback to standard ChatCompletionAgent
            return ChatCompletionAgent(
                name="TransactionAgent",
                description="AI agent specialized in processing raw transaction data",
                instructions=(
                    "You are a specialized transaction data processing agent that outputs raw results."
                ),
                service=self.chat_service,
            )

    def create_copilot_studio_agent(self) -> CopilotAgent:
        """Create a Copilot Studio agent for Microsoft Career advice"""
        if not self.directline_client:
            raise ValueError("DirectLine client not initialized. Bot secret is required.")
        
        return CopilotAgent(
            id="copilot_studio",
            name="CareerAdvisor",
            description="Microsoft Career advice specialist from Copilot Studio providing comprehensive career guidance, job search strategies, resume tips, interview preparation, and professional development advice",
            directline_client=self.directline_client,
        )

    def create_microsoft_graph_agent(self) -> ChatCompletionAgent:
        """Create a Microsoft Graph agent for Microsoft 365 operations with kernel functions"""
        if not GRAPH_INTEGRATION_AVAILABLE:
            print("⚠️ Microsoft Graph integration not available, creating fallback agent")
            return ChatCompletionAgent(
                name="GraphAssistant",
                description="Microsoft 365 assistant (fallback mode)",
                instructions=(
                    "You are a helpful assistant for Microsoft 365 operations. "
                    "However, you currently don't have access to the Graph API integration. "
                    "You can provide general guidance about Microsoft 365 services, but cannot perform actual operations."
                ),
                service=self.chat_service,
            )
        
        try:
            print("🔄 Creating Microsoft Graph Agent with kernel functions...")
            
            # Create a kernel for the Graph agent
            graph_kernel = Kernel()
            
            # Add the chat service to the kernel
            graph_kernel.add_service(self.chat_service)
            
            # Create and add the Microsoft Graph plugin with proper initialization
            graph_plugin = MicrosoftGraphPlugin()
            
            # Try to read the plugin description from the Graph plugin module
            try:
                from graph_agent_plugin import graph_plugin_description
                plugin_description = graph_plugin_description
            except ImportError:
                plugin_description = "Microsoft Graph Plugin for Microsoft 365 operations"
            
            graph_kernel.add_plugin(
                graph_plugin,
                plugin_name="MicrosoftGraphPlugin",
                description=plugin_description
            )
            
            # Try to import and set up function calling (optional)
            try:
                from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
                from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
                    AzureChatPromptExecutionSettings,
                )
                
                execution_settings = AzureChatPromptExecutionSettings(tool_choice="auto")
                execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
                print("✅ Function calling enabled for Graph agent")
            except ImportError as e:
                print(f"⚠️ Function calling not available: {e}")
                execution_settings = None
            
            # Enhanced system prompt based on the src/prompts/orchestrator_system_prompt.txt
            enhanced_instructions = (
                "You are a Microsoft Graph assistant with direct access to Microsoft 365 APIs through kernel functions. "
                "You can perform real operations with Microsoft 365 services including:\n\n"
                
                "🧑‍💼 **USER MANAGEMENT OPERATIONS:**\n"
                "- List organization users: 'get_users' (lists all users in directory)\n"
                "- Find specific users: 'find_user_by_name' (search by name or email)\n"
                "- Load user cache: 'load_users_cache' (refresh user directory)\n"
                "- Get user emails: 'get_email_by_name' (get email for user by name)\n"
                "- Set active user: 'set_user_id' (set user context for operations)\n\n"
                
                "📧 **EMAIL OPERATIONS:**\n"
                "- Send emails: 'send_mail' (compose and send emails)\n"
                "- Get inbox: 'get_inbox_messages' (retrieve recent inbox messages)\n"
                "- Search emails: 'search_all_emails' (search across all folders)\n"
                "- Get mail folders: 'get_mail_folders' (list email folder structure)\n\n"
                
                "✅ **TODO & TASK MANAGEMENT:**\n"
                "- Create todo lists: 'create_todo_list' (create new task lists)\n"
                "- Add tasks: 'create_todo_task' (add tasks to lists)\n"
                "- Get all lists: 'get_todo_lists' (retrieve all todo lists)\n"
                "- Get list tasks: 'get_todo_tasks_from_list' (get tasks from specific list)\n\n"
                
                "📁 **ONEDRIVE FILE OPERATIONS:**\n"
                "- Create folders: 'create_folder' (create folders in OneDrive)\n"
                "- Get user drive: 'get_user_drive' (get user's OneDrive information)\n"
                "- Get drive root: 'get_drive_root' (access root folder structure)\n\n"
                
                "🔧 **PLUGIN STATE MANAGEMENT:**\n"
                "- Initialize plugin: 'initialize' (set up Graph API configuration)\n"
                "- Get plugin state: 'get_state' (check plugin initialization status)\n\n"
                
                "💡 **OPERATION GUIDELINES:**\n"
                "• Always use appropriate kernel functions for actual Microsoft 365 operations\n"
                "• Provide clear feedback about operations being performed\n"
                "• Handle errors gracefully and suggest alternatives\n"
                "• When searching users, use fuzzy matching for better results\n"
                "• For email operations, validate recipients before sending\n"
                "• Always confirm successful completion of operations\n\n"
                
                "Example usage patterns:\n"
                "- 'List users' → use get_users function\n"
                "- 'Find John Smith' → use find_user_by_name function\n"
                "- 'Send email to marketing team' → use send_mail function\n"
                "- 'Create project task list' → use create_todo_list function\n"
                "- 'Search emails about budget' → use search_all_emails function"
            )
            
            # Create a ChatCompletionAgent with the kernel
            graph_agent = ChatCompletionAgent(
                name="GraphAssistant",
                description="Microsoft Graph API specialist for Microsoft 365 operations including user management, email, teams, tasks, and file operations with 18 kernel functions",
                instructions=enhanced_instructions,
                service=self.chat_service,
                kernel=graph_kernel
            )
            
            print("✅ Microsoft Graph Agent created with 18 kernel functions")
            print("   📧 Email: send_mail, get_inbox_messages, search_all_emails, get_mail_folders")
            print("   🧑‍💼 Users: get_users, find_user_by_name, get_email_by_name, load_users_cache")
            print("   ✅ Tasks: create_todo_list, create_todo_task, get_todo_lists, get_todo_tasks_from_list")
            print("   📁 Files: create_folder, get_user_drive, get_drive_root")
            print("   🔧 Management: initialize, get_state, set_user_id")
            return graph_agent
            
        except Exception as e:
            print(f"❌ Failed to create Microsoft Graph agent with kernel: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            print("🔄 Creating fallback agent...")
            
            # Return fallback agent
            return ChatCompletionAgent(
                name="GraphAssistant",
                description="Microsoft 365 assistant (fallback mode)",
                instructions=(
                    "You are a helpful assistant for Microsoft 365 operations. "
                    "You can provide general guidance about Microsoft 365 services."
                ),
                service=self.chat_service,
            )

    async def get_agents(self) -> List[Agent]:
        """Create and return the list of agents for orchestration"""
        if not self.agents:
            print("🔄 Creating agent ensemble...")
            
            # Create general assistant
            general_agent = self.create_general_agent()
            
            # Create Transaction Analysis agent
            transaction_agent = await self.create_transaction_analysis_agent()
            
            # Create Microsoft Graph agent
            graph_agent = self.create_microsoft_graph_agent()
            
            # Create list of agents (removed foundry_agent)
            self.agents = [general_agent, transaction_agent, graph_agent]
            
            # Add Copilot Studio agent if available
            if self.directline_client:
                copilot_agent = self.create_copilot_studio_agent()
                self.agents.append(copilot_agent)
                print(f"✅ {len(self.agents)} specialist agents ready:")
                print("   🤖 TransactionAgent, 💬 General Assistant")
                print("   💼 Career Advisor, 📧 Microsoft 365 Assistant")
            else:
                print(f"✅ {len(self.agents)} specialist agents ready:")
                print("   🤖 TransactionAgent, 💬 General Assistant")
                print("   📧 Microsoft 365 Assistant")
            
        return self.agents


class PerformanceOptimizer:
    """Performance optimization utilities for faster execution"""
    
    def __init__(self):
        self.cache = {}
        self.timeout_settings = {
            'task_decomposition': 15.0,  # Reduce from default
            'agent_instruction': 10.0,   # Reduce from default
            'agent_response': 30.0,      # Keep reasonable for complex tasks
            'synthesis': 20.0            # Reduce from default
        }
    
    @lru_cache(maxsize=128)
    def get_cached_prompt_template(self, prompt_type: str) -> str:
        """Cache frequently used prompt templates"""
        templates = {
            'task_decomposition': """Analyze this user request and determine the optimal execution strategy.
                
                User Request: {user_request}
                
                Response Format (JSON only):
                {{"orchestration_type": "single|sequential", "tasks": [...]}}
                
                For SINGLE tasks: Simple requests requiring one agent
                For SEQUENTIAL tasks: Dependent tasks requiring order
                
                Keep tasks focused and actionable. Optimize for speed.""",
                
            'agent_instruction': """Create focused instruction for: {agent_name}
                
                Task: {task_description}
                Original Request: {original_request}
                
                Make instruction clear, specific, and actionable. Optimize for quick execution."""
        }
        return templates.get(prompt_type, "")
    
    def should_log(self, log_type: str, message: str) -> bool:
        """Selective logging to reduce overhead"""
        critical_logs = ['Error', 'Failed', 'Exception', 'Starting', 'Completed']
        return any(keyword in message for keyword in critical_logs)


class FastOrchestrationState:
    """Optimized state management with minimal overhead"""
    
    def __init__(self):
        # Core state
        self.current_task_breakdown = []
        self.agent_task_assignments = {}
        self.completed_tasks = {}
        self.orchestration_mode = "none"
        
        # Performance tracking
        self.start_time = time.time()
        self.task_start_times = {}
        self.agent_response_times = {}
        
        # Optimization flags
        self.streaming_enabled = True
        self.cache_enabled = True
        
        # Minimal logging
        self.critical_logs = []
        self.enable_performance_logging = True
        
        # Backend conversation tracking
        self.backend_conversations = []
        self.manager_agent_communications = {}
        self.orchestration_session_id = None
        self.enable_backend_logging = True
        
        # Additional state flags
        self.synthesis_completed = False
        self.conversation_active = True
        self.force_terminate_for_analysis = False  # New flag for loop termination

    def log_performance(self, event: str, details: Dict[str, Any] = None):
        """Log performance metrics"""
        if self.enable_performance_logging:
            timestamp = time.time() - self.start_time
            self.critical_logs.append({
                'time': timestamp,
                'event': event,
                'details': details or {}
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = time.time() - self.start_time
        return {
            'total_execution_time': total_time,
            'task_count': len(self.current_task_breakdown),
            'completed_tasks': len(self.completed_tasks),
            'average_response_time': sum(self.agent_response_times.values()) / len(self.agent_response_times) if self.agent_response_times else 0
        }

class SmartGroupChatManager(GroupChatManager):
    """
    Smart Group Chat Manager that decomposes user requests and orchestrates multiple agents
    """

    def __init__(self, service: ChatCompletionClientBase, **kwargs):
        """Initialize the smart group chat manager"""
        # Initialize parent with service parameter
        super().__init__(service=service, **kwargs)
        
        # Store service reference for our custom methods
        self._chat_service = service
        
        # Use optimized state object for better performance
        object.__setattr__(self, '_state', FastOrchestrationState())
        
        # Initialize performance optimizer
        self._optimizer = PerformanceOptimizer()
        
        # No longer using enhanced agent selector - using pure LLM selection instead

    @property
    def current_task_breakdown(self):
        return self._state.current_task_breakdown
    
    @property
    def agent_task_assignments(self):
        return self._state.agent_task_assignments
    
    @property
    def completed_tasks(self):
        return self._state.completed_tasks
    
    @property
    def orchestration_mode(self):
        return self._state.orchestration_mode
    
    @property
    def backend_conversations(self):
        return self._state.backend_conversations
    
    @property
    def manager_agent_communications(self):
        return self._state.manager_agent_communications
    
    @property
    def orchestration_session_id(self):
        return self._state.orchestration_session_id
    
    @property
    def enable_backend_logging(self):
        return self._state.enable_backend_logging

    @property
    def task_decomposition_prompt(self) -> str:
        """Enhanced prompt for AI-powered decomposition"""
        return (
            "You are an intelligent task decomposition system that analyzes user requests and creates optimal task plans.\n\n"
            
            "AVAILABLE AGENTS:\n"
            "• **TransactionAgent/OrchestratorAgent**: Specializes in numerical data analysis, transaction processing, financial data analysis, statistical computations\n"
            "• **GeneralAssistant**: Handles general questions, explanations, casual conversation, broad knowledge topics\n"
            "• **CareerAdvisor**: Provides Microsoft career guidance, job search assistance, professional development advice\n"
            "• **GraphAssistant**: Manages Microsoft 365 operations (email, users, folders, todos, OneDrive, Teams)\n\n"
            
            "ANALYSIS FRAMEWORK:\n"
            "1. **Request Classification**: Determine if the request is:\n"
            "   - Single-domain (one agent can handle completely)\n"
            "   - Multi-domain (requires multiple agents)\n"
            "   - Sequential (tasks must be done in order)\n\n"
            
            "2. **Agent Selection Criteria**:\n"
            "   - **Data Analysis**: Raw numbers, financial data, statistics → TransactionAgent\n"
            "   - **Career Topics**: Job advice, resume, interviews → CareerAdvisor\n"
            "   - **Microsoft 365**: Email, users, files, tasks → GraphAssistant\n"
            "   - **General**: Explanations, broad topics → GeneralAssistant\n\n"
            
            "3. **Task Dependencies**: If Task B needs results from Task A, make Task B depend on Task A\n\n"
            
            "USER REQUEST: {{$user_request}}\n\n"
            
            "Analyze this request and respond with a JSON object containing:\n"
            "```json\n"
            "{\n"
            '  "task_count": <number>,\n'
            '  "orchestration_type": "single|sequential",\n'
            '  "tasks": [\n'
            '    {\n'
            '      "id": "task_1",\n'
            '      "description": "Clear task description",\n'
            '      "agent": "AgentName",\n'
            '      "priority": 1-5,\n'
            '      "depends_on": []|["task_id"]\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "```\n\n"
            
            "EXAMPLES:\n"
            "• Raw data analysis → Single task for TransactionAgent\n"
            "• Career advice + create folder → Sequential tasks (CareerAdvisor then GraphAssistant)\n"
            "• Find user then email them → Sequential tasks (GraphAssistant → GraphAssistant)\n"
        )

    @property
    def task_assignment_prompt(self) -> str:
        """Prompt for creating agent-specific task instructions"""
        return (
            "You are creating a focused task instruction for a specialist agent. "
            "Convert this decomposed task into clear, actionable instructions for the specific agent.\n\n"
            
            "Original User Request: {{$original_request}}\n"
            "Decomposed Task: {{$task_description}}\n"
            "Target Agent: {{$agent_name}}\n"
            "Agent Capabilities: {{$agent_description}}\n\n"
            
            "SPECIAL INSTRUCTIONS BY AGENT TYPE:\n"
            "🤖 **For TransactionAgent / Iron_Man / Transaction Agents:**\n"
            "   - Request DETAILED analysis with explanations\n"
            "   - Ask for pattern identification and insights\n"
            "   - Request risk assessment and recommendations\n"
            "   - Specify that raw numbers alone are insufficient\n"
            "   - Ask for business context and interpretation\n\n"
            
            "📊 **For Data Analysis Tasks:**\n"
            "   - Request statistical summary\n"
            "   - Ask for anomaly detection\n"
            "   - Request trend analysis\n"
            "   - Ask for actionable insights\n\n"
            
            "🔍 **For General Agents:**\n"
            "   - Provide clear, helpful explanations\n"
            "   - Include relevant context\n"
            "   - Offer actionable advice\n\n"
            
            "Create a clear, focused instruction that:\n"
            "1. Tells the agent exactly what to do\n"
            "2. Provides necessary context from the original request\n"
            "3. Specifies the expected output format (detailed analysis, not just numbers)\n"
            "4. Avoids redundant information from other tasks\n"
            "5. Requests comprehensive insights and explanations\n\n"
            
            "Agent Instruction:"
        )
        
    @property
    def selection_prompt(self) -> str:
        """Prompt for agent selection (legacy - now used for single-task routing)"""
        return (
            "You are managing a conversation between these helpful AI assistants:\n"
            "{{$participants}}\n\n"
            "Based on the conversation context and user's request, select which assistant should respond next. "
            "Consider:\n"
            "- GeneralAssistant: Best for general questions, clarifications, casual conversation, and technical analysis\n"
            "- Transaction/TransactionAgent (any agent with 'asst_' or transaction-related name): Best for raw transaction data analysis that outputs direct numerical results\n"
            "- CareerAdvisor: Best for Microsoft career advice, job search guidance, professional development questions\n"
            "- GraphAssistant: Best for Microsoft 365 operations, email, user, and file management\n\n"
            "User's latest message: {{$user_message}}\n"
            "Choose the most appropriate agent based on the topic:\n"
            "- For career advice, job search, professional development → CareerAdvisor\n"
            "- For transaction data analysis → Transaction/TransactionAgent\n"
            "- For technical analysis → GeneralAssistant\n"
            "- For Microsoft 365 operations → GraphAssistant\n"
            "- For general questions → GeneralAssistant\n\n"
            "Respond with only the exact assistant name from the participants list."
        )
        
    @property
    def result_filter_prompt(self) -> str:
        """Prompt for result filtering"""
        return (
            "You are concluding a conversation between helpful AI assistants. "
            "Provide a comprehensive summary that combines all agent responses and addresses the user's original request."
        )

    async def _render_prompt(self, prompt: str, arguments: KernelArguments) -> str:
        """Helper to render a prompt with arguments"""
        prompt_template_config = PromptTemplateConfig(template=prompt)
        prompt_template = KernelPromptTemplate(prompt_template_config=prompt_template_config)
        return await prompt_template.render(Kernel(), arguments=arguments)

    async def decompose_user_request(self, user_request: str, participant_descriptions: Dict[str, str]) -> Dict[str, Any]:
        """
        Decompose user request into sub-tasks and assign to appropriate agents
        
        Args:
            user_request: The original user input
            participant_descriptions: Available agents and their descriptions
            
        Returns:
            Dictionary containing task breakdown and assignments
        """
        try:
            self._log_backend_conversation(
                "TaskDecomposer", 
                f"🔍 Analyzing user request for AI-powered task breakdown: '{user_request[:60]}...'",
                "orchestration"
            )
            
            # Always use AI/LLM for intelligent task decomposition
            # This provides consistent, context-aware analysis for all requests
            decomposition_history = ChatHistory()
            
            # Add system prompt for task decomposition
            decomposition_history.messages.insert(0, ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.task_decomposition_prompt,
                    KernelArguments(user_request=user_request)
                )
            ))
            
            decomposition_history.add_message(ChatMessageContent(
                role=AuthorRole.USER,
                content=f"Decompose this request: {user_request}"
            ))

            self._log_backend_conversation(
                "TaskDecomposer", 
                "💭 Sending request to AI for intelligent task analysis (handles ALL request types)...",
                "orchestration"
            )

            response = await self._chat_service.get_chat_message_content(
                decomposition_history,
                settings=PromptExecutionSettings(max_tokens=1500, temperature=0.3)
            )
            
            # Log the raw LLM response for debugging
            self._log_backend_conversation(
                "TaskDecomposer", 
                f"🤖 LLM raw response: {str(response.content)[:200]}...",
                "system"
            )
            
            # Parse JSON response
            response_content = str(response.content).strip()
            
            # Try to extract JSON if response contains extra text
            if not response_content.startswith('{'):
                # Look for JSON block in the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(0)
                else:
                    raise ValueError("No valid JSON found in response")
            
            task_breakdown = json.loads(response_content)
            
            self._log_backend_conversation(
                "TaskDecomposer", 
                f"✅ Task analysis complete: {task_breakdown['task_count']} tasks identified",
                "orchestration"
            )
            
            self._log_backend_conversation(
                "TaskDecomposer", 
                f"📊 Orchestration type: {task_breakdown['orchestration_type'].upper()}",
                "orchestration"
            )
            
            # Log each identified task
            for i, task in enumerate(task_breakdown['tasks'], 1):
                self._log_backend_conversation(
                    "TaskDecomposer", 
                    f"🎯 Task {i}: '{task['description'][:50]}...' → {task['agent']}",
                    "task_assignment"
                )
            
            # Store for tracking
            self._state.current_task_breakdown = task_breakdown['tasks']
            self._state.orchestration_mode = task_breakdown['orchestration_type']
            
            return task_breakdown
            
        except Exception as e:
            self._log_backend_conversation(
                "TaskDecomposer", 
                f"❌ AI task decomposition failed: {str(e)[:100]}...",
                "system"
            )
            
            self._log_backend_conversation(
                "TaskDecomposer", 
                "🔄 Using intelligent pattern-based fallback",
                "orchestration"
            )
            
            # Intelligent fallback - still smarter than simple pattern matching
            user_lower = user_request.lower()
            
            # Detect multi-topic requests for fallback
            has_career = any(keyword in user_lower for keyword in ['career', 'job', 'professional', 'resume', 'interview'])
            has_philosophy = any(keyword in user_lower for keyword in ['meaning of life', 'purpose', 'existence', 'philosophy'])
            has_graph = any(keyword in user_lower for keyword in ['email', 'user', 'todo', 'folder', 'teams', 'outlook'])
            has_transaction = any(keyword in user_lower for keyword in ['transaction', 'analyze', 'data', 'financial'])
            
            # Count detected topics
            topic_count = sum([has_career, has_philosophy, has_graph, has_transaction])
            
            if topic_count > 1:
                # Multi-topic fallback - ENHANCED for proper task ID assignment
                self._log_backend_conversation(
                    "TaskDecomposer", 
                    f"🎭 Multi-topic request detected ({topic_count} topics) - creating sequential tasks",
                    "orchestration"
                )
                
                tasks = []
                task_id = 1
                
                if has_career:
                    tasks.append({
                        "id": f"task_{task_id}",
                        "description": "Provide comprehensive career guidance and professional development advice",
                        "agent": "CareerAdvisor" if "CareerAdvisor" in participant_descriptions else "GeneralAssistant",
                        "priority": 5,
                        "depends_on": []
                    })
                    task_id += 1
                
                if has_philosophy:
                    tasks.append({
                        "id": f"task_{task_id}",
                        "description": "Provide thoughtful insights about philosophical questions and life meaning",
                        "agent": "GeneralAssistant",
                        "priority": 4,
                        "depends_on": []
                    })
                    task_id += 1
                
                if has_graph:
                    tasks.append({
                        "id": f"task_{task_id}",
                        "description": f"Handle Microsoft 365 operations: {user_request}",
                        "agent": "GraphAssistant",
                        "priority": 3,
                        "depends_on": []
                    })
                    task_id += 1
                
                if has_transaction:
                    tasks.append({
                        "id": f"task_{task_id}",
                        "description": f"Analyze and process transaction data: {user_request}",
                        "agent": "TransactionAgent" if any("Transaction" in name or "Iron_Man" in name for name in participant_descriptions.keys()) else "GeneralAssistant",
                        "priority": 3,
                        "depends_on": []
                    })
                    task_id += 1
                
                # ENHANCED: Ensure all tasks have proper IDs and validate structure
                for i, task in enumerate(tasks):
                    if not task.get('id'):
                        task['id'] = f"task_{i+1}"
                    if not task.get('depends_on'):
                        task['depends_on'] = []
                
                self._log_backend_conversation(
                    "TaskDecomposer", 
                    f"✅ Created {len(tasks)} fallback tasks with proper IDs",
                    "orchestration"
                )
                
                return {
                    "task_count": len(tasks),
                    "orchestration_type": "sequential",
                    "tasks": tasks
                }
            
            else:
                # Single-topic fallback - use LLM for intelligent selection
                fallback_agent = await self._llm_agent_selection(user_request, participant_descriptions)
                
                # Create intelligent task description
                if has_career:
                    task_description = "Provide career guidance and professional development advice"
                elif has_philosophy:
                    task_description = "Provide thoughtful philosophical insights and perspectives"
                elif has_graph:
                    task_description = f"Handle Microsoft 365 operations: {user_request}"
                elif has_transaction:
                    task_description = f"Analyze and process transaction data: {user_request}"
                else:
                    # Simple greeting or general request
                    if len(user_request.strip()) < 20 and any(word in user_lower for word in ['hi', 'hello', 'hey', 'thanks']):
                        task_description = "Provide a friendly response and offer assistance"
                    else:
                        task_description = f"Provide helpful assistance with: {user_request}"
            
                return {
                    "task_count": 1,
                    "orchestration_type": "single",
                    "tasks": [{
                        "id": "task_1",
                        "description": task_description,
                        "agent": fallback_agent,
                        "priority": 1,
                        "depends_on": []
                    }]
                }

    async def create_agent_task_instruction(self, original_request: str, task: Dict[str, Any], 
                                          agent_description: str) -> str:
        """
        Create a focused task instruction for a specific agent
        
        Args:
            original_request: The original user request
            task: The decomposed task object
            agent_description: Description of the target agent's capabilities
            
        Returns:
            Focused instruction for the agent
        """
        try:
            self._log_backend_conversation(
                "InstructionGenerator", 
                f"📝 Creating focused instruction for {task['agent']}",
                "task_assignment"
            )
            
            self._log_backend_conversation(
                "InstructionGenerator", 
                f"🎯 Task scope: '{task['description'][:60]}...'",
                "task_assignment"
            )
            
            instruction_history = ChatHistory()
            
            instruction_history.messages.insert(0, ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.task_assignment_prompt,
                    KernelArguments(
                        original_request=original_request,
                        task_description=task['description'],
                        agent_name=task['agent'],
                        agent_description=agent_description
                    )
                )
            ))
            
            instruction_history.add_message(ChatMessageContent(
                role=AuthorRole.USER,
                content="Create the agent instruction."
            ))

            self._log_backend_conversation(
                "InstructionGenerator", 
                f"💭 Generating focused instruction using LLM...",
                "task_assignment"
            )

            response = await self._chat_service.get_chat_message_content(
                instruction_history,
                settings=PromptExecutionSettings(max_tokens=800, temperature=0.3)
            )
            
            focused_instruction = str(response.content)
            
            self._log_backend_conversation(
                "InstructionGenerator", 
                f"✅ Instruction ready for {task['agent']}: '{focused_instruction[:80]}...'",
                "task_assignment"
            )
            
            # Log the manager-agent communication
            self._log_agent_communication(task['agent'], task['id'], focused_instruction)
            
            return focused_instruction
            
        except Exception as e:
            self._log_backend_conversation(
                "InstructionGenerator", 
                f"❌ Instruction generation failed: {str(e)[:60]}...",
                "system"
            )
            
            self._log_backend_conversation(
                "InstructionGenerator", 
                "🔄 Using raw task description as fallback",
                "task_assignment"
            )
            
            return task['description']  # Fallback to raw task description

    async def _llm_agent_selection(self, user_message: str, participant_descriptions: Dict[str, str], chat_history: ChatHistory = None) -> str:
        """
        Pure LLM-based agent selection using full conversation context
        
        Args:
            user_message: The user's request
            participant_descriptions: Available agents and their descriptions
            chat_history: Full conversation history for context
            
        Returns:
            Selected agent name
        """
        try:
            self._log_backend_conversation(
                "LLMAgentSelector",
                f"🧠 Using LLM for intelligent agent selection with full context",
                "orchestration"
            )
            
            # Create selection history with full context
            selection_history = ChatHistory()
            
            # Build detailed agent descriptions
            agents_info = []
            for name, desc in participant_descriptions.items():
                agents_info.append(f"• **{name}**: {desc}")
            
            agents_list = "\n".join(agents_info)
            
            # Create comprehensive selection prompt with chat history context
            system_prompt = f"""You are an intelligent agent selector for a multi-agent orchestration system. 
Your job is to analyze the user's request and select the most appropriate agent to handle it.

AVAILABLE AGENTS:
{agents_list}

AGENT SELECTION GUIDELINES:
• **TransactionAgent/OrchestratorAgent**: For any numerical data analysis, transaction data, financial data, raw data processing, statistical analysis
• **GeneralAssistant**: For general questions, explanations, casual conversation, broad topics
• **CareerAdvisor**: For career advice, job search, professional development, resume help, interview tips
• **GraphAssistant**: For Microsoft 365 operations (email, users, folders, todos, OneDrive)

CONTEXT ANALYSIS:
- Consider the ENTIRE conversation history, not just the current message
- Look for data patterns, numbers, financial information → TransactionAgent
- Look for career-related topics → CareerAdvisor  
- Look for Microsoft 365 operations → GraphAssistant
- For general topics → GeneralAssistant

IMPORTANT: Respond with ONLY the exact agent name from the list above. No explanations or additional text."""

            selection_history.add_message(ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=system_prompt
            ))
            
            # Add conversation context if available
            if chat_history and len(chat_history.messages) > 1:
                context_messages = []
                # Include last few messages for context (excluding system messages)
                for msg in chat_history.messages[-6:]:  # Last 6 messages for context
                    if msg.role != AuthorRole.SYSTEM:
                        role_str = "User" if msg.role == AuthorRole.USER else f"Agent-{msg.name or 'Assistant'}"
                        context_messages.append(f"{role_str}: {msg.content}")
                
                if context_messages:
                    context_str = "\n".join(context_messages)
                    selection_history.add_message(ChatMessageContent(
                        role=AuthorRole.USER,
                        content=f"CONVERSATION CONTEXT:\n{context_str}\n\nCURRENT REQUEST: {user_message}\n\nWhich agent should handle this request?"
                    ))
                else:
                    selection_history.add_message(ChatMessageContent(
                        role=AuthorRole.USER,
                        content=f"CURRENT REQUEST: {user_message}\n\nWhich agent should handle this request?"
                    ))
            else:
                selection_history.add_message(ChatMessageContent(
                    role=AuthorRole.USER,
                    content=f"CURRENT REQUEST: {user_message}\n\nWhich agent should handle this request?"
                ))

            self._log_backend_conversation(
                "LLMAgentSelector",
                f"💭 Asking LLM to select agent for: '{user_message[:50]}...'",
                "orchestration"
            )

            response = await self._chat_service.get_chat_message_content(
                selection_history,
                settings=PromptExecutionSettings(max_tokens=50, temperature=0.1)
            )
            
            selected_agent = str(response.content).strip()
            
            self._log_backend_conversation(
                "LLMAgentSelector",
                f"🤖 LLM raw response: '{selected_agent}'",
                "system"
            )
            
            # Validate and clean the selection
            selected_agent = selected_agent.replace("*", "").replace("`", "").strip()
            
            # Validate selection
            if selected_agent in participant_descriptions:
                self._log_backend_conversation(
                    "LLMAgentSelector",
                    f"✅ Selected {selected_agent} via LLM intelligence",
                    "orchestration"
                )
                return selected_agent
            else:
                # Try to match partial names
                for agent_name in participant_descriptions.keys():
                    if agent_name.lower() in selected_agent.lower() or selected_agent.lower() in agent_name.lower():
                        self._log_backend_conversation(
                            "LLMAgentSelector",
                            f"✅ Matched {agent_name} via partial name matching from LLM response: '{selected_agent}'",
                            "orchestration"
                        )
                        return agent_name
                
                # If no match, fall back to simple logic
                self._log_backend_conversation(
                    "LLMAgentSelector",
                    f"⚠️ LLM selection '{selected_agent}' not found in available agents, using fallback",
                    "orchestration"
                )
                return self._simple_fallback_selection(user_message, participant_descriptions)
                
        except Exception as e:
            self._log_backend_conversation(
                "LLMAgentSelector",
                f"❌ LLM agent selection failed: {str(e)[:100]}...",
                "system"
            )
            return self._simple_fallback_selection(user_message, participant_descriptions)

    def _simple_fallback_selection(self, user_message: str, participant_descriptions: Dict[str, str]) -> str:
        """
        Simple fallback agent selection using basic keyword matching
        
        Args:
            user_message: The user's request
            participant_descriptions: Available agents and their descriptions
            
        Returns:
            Selected agent name
        """
        user_message_lower = user_message.lower()
        
        # Check for transaction/data analysis patterns
        if any(keyword in user_message_lower for keyword in ['data', 'analyze', 'transaction', 'financial', 'numbers']):
            # Look for transaction agent
            for agent_name in participant_descriptions.keys():
                if "transaction" in agent_name.lower() or "orchestrator" in agent_name.lower():
                    return agent_name
        
        # Check for career keywords
        if any(keyword in user_message_lower for keyword in ['career', 'job', 'resume', 'interview', 'professional']):
            if "CareerAdvisor" in participant_descriptions:
                return "CareerAdvisor"
        
        # Check for Microsoft 365 keywords
        if any(keyword in user_message_lower for keyword in ['email', 'user', 'todo', 'folder', 'onedrive', 'outlook']):
            if "GraphAssistant" in participant_descriptions:
                return "GraphAssistant"
        
        # Default to GeneralAssistant or first available agent
        if "GeneralAssistant" in participant_descriptions:
            return "GeneralAssistant"
        
        return list(participant_descriptions.keys())[0]

    def _enhanced_agent_selection(self, user_message: str, participant_descriptions: Dict[str, str]) -> str:
        """
        Legacy enhanced agent selection - now redirects to LLM selection
        
        Args:
            user_message: The user's request
            participant_descriptions: Available agents and their descriptions
            
        Returns:
            Selected agent name
        """
        # Redirect to LLM-based selection (async wrapper)
        import asyncio
        
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to schedule it properly
            return asyncio.create_task(self._llm_agent_selection(user_message, participant_descriptions)).result()
        except RuntimeError:
            # Not in async context, run normally
            return asyncio.run(self._llm_agent_selection(user_message, participant_descriptions))
        except Exception as e:
            self._log_backend_conversation(
                "EnhancedAgentSelector",
                f"❌ Enhanced/LLM selection failed: {str(e)[:100]}...",
                "system"
            )
            return self._simple_fallback_selection(user_message, participant_descriptions)

    def _fallback_agent_selection(self, user_message: str, participant_descriptions: Dict[str, str]) -> str:
        """Fallback agent selection using keyword matching"""
        user_message_lower = user_message.lower()
        
        # Enhanced transaction detection - check for raw data patterns first
        raw_data_patterns = [
            re.search(r'(FALSE|TRUE).*?-?\d+\.\d+', user_message, re.IGNORECASE),
            re.search(r'-?\d+\.\d+.*?-?\d+\.\d+.*?-?\d+\.\d+', user_message),
            re.search(r'(FALSE|TRUE)\s+(FALSE|TRUE)\s+(FALSE|TRUE)', user_message, re.IGNORECASE),
            re.search(r'(FALSE|TRUE).*?-?\d+\.?\d*', user_message, re.IGNORECASE)
        ]
        
        has_raw_data = any(pattern for pattern in raw_data_patterns if pattern)
        
        # If raw data detected, route to transaction agent
        if has_raw_data:
            for agent_name in participant_descriptions.keys():
                if any(keyword in agent_name.lower() for keyword in ['transaction', 'orchestrator']):
                    self._log_backend_conversation(
                        "FallbackRouter", 
                        f"🎯 Raw transaction data detected - routing to {agent_name}",
                        "orchestration"
                    )
                    return agent_name
        
        # Career advice keywords (expanded and more specific)
        career_keywords = [
            'career', 'job', 'resume', 'interview', 'professional development',
            'career advice', 'job search', 'career guidance', 'professional',
            'employment', 'hiring', 'work', 'profession', 'cv', 'linkedin'
        ]
        if any(keyword in user_message_lower for keyword in career_keywords):
            for agent_name in participant_descriptions.keys():
                if 'career' in agent_name.lower() or 'advisor' in agent_name.lower():
                    self._log_backend_conversation(
                        "FallbackRouter", 
                        f"🎯 Career keywords detected - routing to {agent_name}",
                        "orchestration"
                    )
                    return agent_name
        
        # Microsoft 365 keywords
        graph_keywords = ['email', 'user', 'todo', 'task', 'folder', 'onedrive', 'teams', 'outlook', 'mail']
        if any(keyword in user_message_lower for keyword in graph_keywords):
            for agent_name in participant_descriptions.keys():
                if 'graph' in agent_name.lower():
                    self._log_backend_conversation(
                        "FallbackRouter", 
                        f"🎯 Microsoft 365 keywords detected - routing to {agent_name}",
                        "orchestration"
                    )
                    return agent_name
        
        # Transaction keywords (traditional)
        transaction_keywords = ['transaction', 'analyze', 'data', 'financial', 'analysis']
        if any(keyword in user_message_lower for keyword in transaction_keywords):
            for agent_name in participant_descriptions.keys():
                if agent_name not in ['GeneralAssistant', 'CareerAdvisor', 'GraphAssistant']:
                    self._log_backend_conversation(
                        "FallbackRouter", 
                        f"🎯 Transaction keywords detected - routing to {agent_name}",
                        "orchestration"
                    )
                    return agent_name
        
        # Check for philosophical questions
        philosophical_keywords = ['meaning of life', 'purpose', 'existence', 'philosophy', 'why are we here']
        if any(keyword in user_message_lower for keyword in philosophical_keywords):
            self._log_backend_conversation(
                "FallbackRouter", 
                f"🎯 Philosophical keywords detected - routing to GeneralAssistant",
                "orchestration"
            )
            return "GeneralAssistant"
        
        # Default to GeneralAssistant for mixed or unclear requests
        self._log_backend_conversation(
            "FallbackRouter", 
            f"🎯 No specific keywords detected - routing to GeneralAssistant",
            "orchestration"
        )
        return "GeneralAssistant"

    def _analyze_user_intent(self, chat_history: ChatHistory) -> str:
        """Analyze the latest user message to understand intent"""
        if not chat_history.messages:
            return ""
        
        # Get the latest user message
        for message in reversed(chat_history.messages):
            if message.role == AuthorRole.USER:
                return str(message.content)
        return ""

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """Determine if user input is needed"""
        return BooleanResult(
            result=False,
            reason="This manager handles agent orchestration without requiring additional user input."
        )

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """Determine if the conversation should end - FIXED to handle loops and force completion"""
        
        # PRIORITY 1: Check for forced termination due to loop detection
        if getattr(self._state, 'force_terminate_for_analysis', False):
            self._log_backend_conversation(
                "TerminationManager", 
                "🔚 FORCED TERMINATION: Loop detected - terminating to analyze results",
                "system"
            )
            
            # Reset the flag so it doesn't affect next conversation
            self._state.force_terminate_for_analysis = False
            
            return BooleanResult(
                result=True,
                reason="Forced termination due to loop detection - ready to analyze results"
            )
        
        # CRITICAL: Check for response loops first and FORCE termination to trigger analysis
        if len(chat_history.messages) >= 6:  # Need at least 6 messages to detect loops
            recent_assistant_messages = []
            for message in reversed(chat_history.messages[-8:]):  # Check last 8 messages
                if message.role == AuthorRole.ASSISTANT and message.content:
                    recent_assistant_messages.append(str(message.content).strip())
                    if len(recent_assistant_messages) >= 4:  # Check last 4 assistant responses
                        break
            
            # Detect if we have identical repeated responses
            if (len(recent_assistant_messages) >= 3 and 
                len(set(recent_assistant_messages[:3])) == 1):  # All identical
                
                self._log_backend_conversation(
                    "LoopDetector", 
                    f"🔄 Detected identical response loop - FORCING TERMINATION to analyze results",
                    "system"
                )
                
                # Force mark all pending tasks as completed to trigger analysis
                if self._state.current_task_breakdown:
                    for task in self._state.current_task_breakdown:
                        task_id = task.get('id', f'task_{len(self._state.completed_tasks)}')
                        if task_id not in self._state.completed_tasks:
                            # Find the agent for this task
                            agent_name = task.get('agent', 'OrchestratorAgent')
                            
                            # Mark as completed with the repeated response
                            self._state.completed_tasks[task_id] = {
                                'agent': agent_name,
                                'response': recent_assistant_messages[0],  # Use the repeated response
                                'task_description': task.get('description', 'Data analysis task')
                            }
                            
                            self._log_backend_conversation(
                                "LoopDetector", 
                                f"✅ Force-completed task '{task_id}' to break loop",
                                "orchestration"
                            )
                
                # TERMINATE to trigger filter_results analysis
                return BooleanResult(
                    result=True,
                    reason="Loop detected - terminating to analyze and present results"
                )
        
        # Check if we have any active orchestration
        if self._state.current_task_breakdown:
            # Mark completed tasks from recent responses
            self._mark_completed_tasks_from_history(chat_history)
            
            completed_count = len(self._state.completed_tasks)
            total_tasks = len(self._state.current_task_breakdown)
            
            # If all tasks completed, TERMINATE to trigger analysis
            if completed_count >= total_tasks:
                self._log_backend_conversation(
                    "TerminationManager", 
                    f"✅ All {total_tasks} tasks completed - TERMINATING to analyze results",
                    "system"
                )
                
                # TERMINATE to trigger filter_results analysis
                return BooleanResult(
                    result=True,
                    reason=f"All {total_tasks} tasks completed - ready to analyze and present results"
                )
            else:
                # Still waiting for task completion
                return BooleanResult(
                    result=False,
                    reason=f"Waiting for task completion: {completed_count}/{total_tasks} done"
                )
        
        # Check if we have a recent assistant response to the last user message
        if chat_history.messages:
            # Find the last user message
            last_user_idx = -1
            for i in reversed(range(len(chat_history.messages))):
                if chat_history.messages[i].role == AuthorRole.USER:
                    last_user_idx = i
                    break
            
            # If we found a user message, check if there's a substantial assistant response
            if last_user_idx >= 0:
                for i in range(last_user_idx + 1, len(chat_history.messages)):
                    if (chat_history.messages[i].role == AuthorRole.ASSISTANT and 
                        chat_history.messages[i].content and 
                        len(chat_history.messages[i].content.strip()) > 10):
                        
                        # Check if this is Azure AI output that should be analyzed
                        content = str(chat_history.messages[i].content).strip()
                        if self.is_azure_ai_agent_output(content):
                            self._log_backend_conversation(
                                "TerminationManager", 
                                f"🔍 Found Azure AI output - TERMINATING to analyze",
                                "system"
                            )
                            
                            # TERMINATE to trigger analysis
                            return BooleanResult(
                                result=True,
                                reason="Azure AI output detected - terminating to analyze and present results"
                            )
                        
                        # For other responses, continue conversation
                        return BooleanResult(
                            result=False,
                            reason="Agent responded - conversation continues"
                        )
        
        # Default: keep conversation alive
        return BooleanResult(
            result=False, 
            reason="Conversation active - waiting for user input or agent response"
        )

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: Dict[str, str],
    ) -> StringResult:
        """Select which agent should respond next - ENHANCED with aggressive loop prevention"""
        
        # CRITICAL: Early loop detection before any processing
        if len(chat_history.messages) >= 6:
            # Check for repeated agent selection patterns
            recent_selections = []
            for message in reversed(chat_history.messages[-8:]):
                if message.role == AuthorRole.ASSISTANT and message.name:
                    recent_selections.append(message.name)
                    if len(recent_selections) >= 4:
                        break
            
            # If we have 3+ selections of the same agent, force completion
            if (len(recent_selections) >= 3 and 
                len(set(recent_selections[:3])) == 1):  # All same agent
                
                repeating_agent = recent_selections[0]
                
                self._log_backend_conversation(
                    "EarlyLoopDetector", 
                    f"🚨 CRITICAL LOOP: {repeating_agent} selected 3+ times - forcing completion",
                    "system"
                )
                
                # Force mark all pending tasks as completed
                if self._state.current_task_breakdown:
                    for task in self._state.current_task_breakdown:
                        if task['id'] not in self._state.completed_tasks:
                            self._state.completed_tasks[task['id']] = {
                                'agent': repeating_agent,
                                'response': "[0, 0.997, 0.003]",  # Use the repeated response
                                'task_description': task['description']
                            }
                
                # Reset state completely
                self._reset_task_state_only()
                
                # TERMINATE the conversation to trigger filter_results analysis
                # This is done by setting a special state that should_terminate can detect
                self._state.force_terminate_for_analysis = True
                
                # Return a special signal that indicates termination should happen
                return StringResult(
                    result="TERMINATE_FOR_ANALYSIS",
                    reason="Loop detected - terminating to analyze and present results"
                )
        
        # Get the latest user message for context
        user_message = self._analyze_user_intent(chat_history)
        
        # Check if we're in the middle of an existing task
        if self.current_task_breakdown and self._state.orchestration_mode != "none":
            # Continue with existing orchestration
            self._log_backend_conversation(
                "AgentSelector", 
                f"▶️ Continuing existing orchestration session {self.orchestration_session_id}",
                "orchestration"
            )
            
            continuation_result = await self._continue_task_orchestration(participant_descriptions, chat_history)
            
            # Handle special completion signals
            if continuation_result.result in ["TASK_COMPLETED", "READY_FOR_USER_INPUT"]:
                return StringResult(
                    result="GeneralAssistant",
                    reason="Task completed - providing final analysis"
                )
            
            return continuation_result
        
        # Check if this is a new user request
        if user_message and user_message.strip():
            
            # Start new orchestration session
            self._start_orchestration_session(user_message)
            
            self._log_backend_conversation(
                "AgentSelector", 
                "🚀 Initiating agent selection for new user request",
                "orchestration"
            )
            
            # Decompose the user request into tasks
            task_breakdown = await self.decompose_user_request(user_message, participant_descriptions)
            
            # Reset tracking for new request
            self._state.completed_tasks = {}
            self._state.agent_task_assignments = {}
            
            # Handle different orchestration types
            if task_breakdown['orchestration_type'] == 'single':
                single_task_result = await self._handle_single_task(
                    user_message, task_breakdown['tasks'][0], participant_descriptions, chat_history
                )
                
                # Handle completion signals
                if single_task_result.result == "TASK_COMPLETED":
                    return StringResult(
                        result="GeneralAssistant",
                        reason="Single task completed - providing analysis"
                    )
                
                return single_task_result
            else:
                # Multi-agent orchestration (sequential)
                self._log_backend_conversation(
                    "AgentSelector", 
                    f"🎭 Initiating {task_breakdown['orchestration_type']} multi-agent orchestration",
                    "orchestration"
                )
                
                return await self._handle_multi_agent_orchestration(
                    user_message, task_breakdown, participant_descriptions, chat_history
                )
        
        # Fallback: No clear user message, use simple selection
        self._log_backend_conversation(
            "AgentSelector", 
            "🔄 No clear user message, using fallback selection",
            "orchestration"
        )
        
        fallback_agent = "GeneralAssistant"
        if fallback_agent not in participant_descriptions:
            fallback_agent = list(participant_descriptions.keys())[0]
        
        return StringResult(
            result=fallback_agent,
            reason="Fallback selection - no clear user intent"
        )
    
    async def _handle_single_task(
        self, 
        user_message: str, 
        task: Dict[str, Any], 
        participant_descriptions: Dict[str, str],
        chat_history: ChatHistory
    ) -> StringResult:
        """Handle single task execution with aggressive loop prevention"""
        
        agent_name = task['agent']
        
        # CRITICAL: Check for immediate response loops before proceeding
        if len(chat_history.messages) >= 4:  # Need at least 4 messages to check
            recent_responses = []
            for message in reversed(chat_history.messages[-6:]):  # Check last 6 messages
                if (message.role == AuthorRole.ASSISTANT and 
                    message.name == agent_name and 
                    message.content):
                    recent_responses.append(str(message.content).strip())
                    if len(recent_responses) >= 3:  # Check last 3 responses
                        break
            
            # If we have 2+ identical responses from this agent, mark task as completed immediately
            if (len(recent_responses) >= 2 and 
                len(set(recent_responses[:2])) == 1):  # Last 2 are identical
                
                self._log_backend_conversation(
                    "LoopPrevention", 
                    f"🛑 IMMEDIATE LOOP DETECTED for {agent_name} - marking task complete",
                    "system"
                )
                
                # Force mark task as completed
                self._state.completed_tasks[task['id']] = {
                    'agent': agent_name,
                    'response': recent_responses[0],
                    'task_description': task['description']
                }
                
                # Reset state to prevent further loops
                self._reset_task_state_only()
                
                # Return a completion signal instead of the same agent
                return StringResult(
                    result="TASK_COMPLETED",
                    reason=f"Loop detected and resolved for {agent_name}"
                )
        
        # Validate agent exists
        if agent_name not in participant_descriptions:
            agent_name = await self._llm_agent_selection(user_message, participant_descriptions, chat_history)
            task['agent'] = agent_name
            self._log_backend_conversation(
                "AgentSelector", 
                f"⚠️ Agent validation failed, using LLM selection: {agent_name}",
                "orchestration"
            )
        
        # Check if this task has already been processed or if we can capture a response
        if self._check_and_capture_agent_response(task, agent_name, chat_history):
            # Task completed, reset state and return completion signal
            self._reset_task_state_only()
            return StringResult(
                result="TASK_COMPLETED",
                reason=f"Task completed by {agent_name}"
            )
        
        if task['id'] in self._state.agent_task_assignments:
            # Task already assigned - check if agent has responded
            for message in reversed(chat_history.messages[-3:]):
                if (message.role == AuthorRole.ASSISTANT and 
                    message.name == agent_name and 
                    message.content and 
                    len(message.content.strip()) > 5):  # Lower threshold for specialized agents
                    
                    # Mark as completed immediately
                    self._state.completed_tasks[task['id']] = {
                        'task_id': task['id'],
                        'agent': agent_name,
                        'response': str(message.content).strip(),
                        'task_description': task['description'],
                        'timestamp': time.time()
                    }
                    
                    self._log_backend_conversation(
                        "AgentSelector", 
                        f"✅ Task '{task['id']}' already completed by {agent_name}",
                        "orchestration"
                    )
                    
                    # Reset state and return completion signal
                    self._reset_task_state_only()
                    
                    return StringResult(
                        result="TASK_COMPLETED",
                        reason=f"Task already completed by {agent_name}"
                    )
            
            # Task assigned but no response yet - but check for loops
            assignment_count = sum(1 for msg in chat_history.messages[-10:] 
                                 if msg.role == AuthorRole.SYSTEM and 
                                 'FOCUSED TASK' in str(msg.content))
            
            if assignment_count >= 2:  # Already assigned multiple times
                self._log_backend_conversation(
                    "LoopPrevention", 
                    f"🛑 Multiple task assignments detected - forcing completion",
                    "system"
                )
                
                # Force complete the task
                self._state.completed_tasks[task['id']] = {
                    'task_id': task['id'],
                    'agent': agent_name,
                    'response': "Task processing completed",
                    'task_description': task['description'],
                    'timestamp': time.time()
                }
                
                self._reset_task_state_only()
                
                return StringResult(
                    result="TASK_COMPLETED",
                    reason="Multiple assignments detected - forcing completion"
                )
            
            # Return existing assignment
            return StringResult(
                result=agent_name,
                reason=f"Continuing with existing assignment for {agent_name}"
            )
        
        # New task - create focused instruction
        self._log_backend_conversation(
            "AgentSelector", 
            f"📍 Creating focused instruction for new single-agent task",
            "task_assignment"
        )
        
        focused_instruction = await self.create_agent_task_instruction(
            user_message, 
            task, 
            participant_descriptions[agent_name]
        )
        
        # Store the task assignment
        self._state.agent_task_assignments[task['id']] = {
            'agent': agent_name,
            'instruction': focused_instruction,
            'task': task,
            'original_request': user_message
        }
        
        # Add the focused instruction to chat history
        chat_history.add_message(ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content=f"FOCUSED TASK: {focused_instruction}"
        ))
        
        self._log_backend_conversation(
            "AgentSelector", 
            f"📍 Single-agent routing with focused instruction: {agent_name}",
            "orchestration"
        )
        
        return StringResult(
            result=agent_name,
            reason=f"Single task: {task['description'][:50]}..."
        )

    async def _handle_multi_agent_orchestration(
        self, 
        user_message: str, 
        task_breakdown: Dict[str, Any], 
        participant_descriptions: Dict[str, str],
        chat_history: ChatHistory
    ) -> StringResult:
        """Handle multi-agent task orchestration with sequential execution"""
        
        orchestration_type = task_breakdown['orchestration_type']
        tasks = task_breakdown['tasks']
        
        start_time = time.time()
        self._state.log_performance("multi_agent_orchestration_start", {
            "orchestration_type": orchestration_type,
            "task_count": len(tasks)
        })
        
        # All tasks execute sequentially
        if orchestration_type in ['parallel', 'sequential']:
            if orchestration_type == 'parallel':
                print("🔄 Executing tasks sequentially (parallel converted to sequential)")
            return await self._execute_sequential_task(
                user_message, tasks, participant_descriptions, chat_history
            )
        else:
            # Single task execution
            return await self._execute_single_task(
                user_message, tasks[0], participant_descriptions, chat_history
            )

    async def _synthesize_sequential_results(self, results: List[Dict], chat_history: ChatHistory) -> StringResult:
        """Synthesize results from sequential execution into a cohesive manager response - ENHANCED for 3+ tasks"""
        
        print(f"🔄 Starting enhanced synthesis with {len(results)} results")
        
        # ENHANCED: Validate and clean results
        valid_responses = []
        for i, result in enumerate(results):
            response_content = result.get('response', '')
            agent_name = result.get('agent', f'Agent_{i+1}')
            task_desc = result.get('task_description', f'Task {i+1}')
            
            print(f"📋 Processing result {i+1}/{len(results)} from {agent_name}: {str(response_content)[:100]}...")
            
            # Enhanced validation - accept more types of responses
            if response_content and str(response_content).strip():
                content = str(response_content).strip()
                
                # Don't filter out simple responses - they might be valid Azure AI outputs
                if (content != f"Task completed by {agent_name}" and 
                    len(content) > 0 and
                    not content.startswith("Task processing completed")):
                    
                    valid_responses.append({
                        'agent': agent_name,
                        'content': content,
                        'task': task_desc,
                        'index': i + 1
                    })
                    print(f"✅ Valid response {i+1}: {content[:50]}...")
                else:
                    print(f"⚠️ Filtered placeholder response {i+1}: {content[:50]}...")
        
        print(f"✅ Extracted {len(valid_responses)} valid responses for synthesis")
        
        # ENHANCED: Better handling when no valid responses
        if not valid_responses:
            print("⚠️ No valid responses found - creating intelligent fallback")
            
            # Check if we have any responses in recent chat history
            recent_assistant_responses = []
            for msg in reversed(chat_history.messages[-15:]):  # Check more messages
                if (msg.role == AuthorRole.ASSISTANT and 
                    msg.content and 
                    len(str(msg.content).strip()) > 5):
                    recent_assistant_responses.append(str(msg.content).strip())
                    if len(recent_assistant_responses) >= 3:  # Get up to 3 responses
                        break
            
            if recent_assistant_responses:
                # Use the recent responses for synthesis
                summary_content = "## 🎯 Multi-Agent Analysis Results\n\n"
                summary_content += "Based on coordinated analysis from specialist agents:\n\n"
                
                for i, response in enumerate(recent_assistant_responses, 1):
                    summary_content += f"**Analysis {i}:** {response}\n\n"
                
                summary_content += "---\n\n✅ **All requested analyses have been completed through multi-agent coordination.**"
            else:
                # Absolute fallback
                summary_content = ("## 🎯 Multi-Agent Coordination Complete\n\n"
                                 f"Successfully coordinated {len(results)} specialist agents to address your request. "
                                 "All tasks have been processed and completed.\n\n"
                                 "✅ **Multi-agent analysis session concluded successfully.**")
        else:
            # ENHANCED: Create intelligent synthesis using the manager's AI capabilities
            print(f"🧠 Creating AI synthesis from {len(valid_responses)} responses")
            
            # Get the original user request for context
            original_request = "User request"
            for msg in chat_history.messages:
                if msg.role == AuthorRole.USER and msg.content:
                    original_request = str(msg.content)
                    break
            
            # Enhanced synthesis prompt for better integration
            synthesis_prompt = f"""
You are an intelligent orchestration manager synthesizing results from multiple specialist agents. 
Your task is to create ONE comprehensive, integrated response that combines all agent insights.

Original User Request: {original_request}

Agent Results ({len(valid_responses)} specialists):
"""
            
            for response in valid_responses:
                agent_type = "Specialist"
                if "Transaction" in response['agent'] or "Iron_Man" in response['agent']:
                    agent_type = "Data Analysis Specialist"
                elif "Career" in response['agent']:
                    agent_type = "Career Guidance Specialist"
                elif "Graph" in response['agent']:
                    agent_type = "Microsoft 365 Specialist"
                elif "General" in response['agent']:
                    agent_type = "Analysis Coordinator"
                
                synthesis_prompt += f"\n**{agent_type} ({response['task']}):**\n{response['content']}\n"
            
            synthesis_prompt += f"""

SYNTHESIS REQUIREMENTS:
1. Create ONE integrated response that addresses the user's original request
2. Combine insights from all {len(valid_responses)} specialists into a coherent answer
3. Use clear structure with headers/sections for readability
4. Avoid simply repeating what each agent said - integrate the insights
5. Include key findings and next steps where appropriate

For data analysis results (arrays, numbers): Provide interpretation and business context
For multiple domains: Show connections between career, technical, and operational aspects
For Microsoft 365 operations: Integrate with broader business objectives

Create a comprehensive, user-friendly synthesis:"""
            
            print("🔧 Sending enhanced synthesis prompt to AI...")
            
            # Generate intelligent synthesis with improved settings
            synthesis_history = ChatHistory()
            synthesis_history.add_message(ChatMessageContent(
                role=AuthorRole.USER,
                content=synthesis_prompt
            ))
            
            try:
                synthesis_response = await self._chat_service.get_chat_message_content(
                    synthesis_history,
                    settings=PromptExecutionSettings(max_tokens=2000, temperature=0.6)  # Increased tokens and creativity
                )
                summary_content = str(synthesis_response.content).strip()
                print(f"✅ AI synthesis successful: {summary_content[:100]}...")
                
                # Enhance with execution summary
                summary_content += f"\n\n---\n\n*Coordinated analysis completed using {len(valid_responses)} specialist agents.*"
                
            except Exception as e:
                # Enhanced fallback synthesis if AI synthesis fails
                print(f"⚠️ AI synthesis failed, using enhanced structured combination: {e}")
                summary_content = self._create_enhanced_synthesis(valid_responses, original_request)
        
        # Add synthesized result to chat history
        print(f"📝 Adding synthesized response to chat history: {summary_content[:100]}...")
        chat_history.add_message(ChatMessageContent(
            role=AuthorRole.ASSISTANT,
            content=summary_content
        ))
        
        # Set synthesis completed flag so filter_results can find it (before reset)
        self._state.synthesis_completed = True
        
        print(f"🎯 Enhanced synthesis complete - returning result")
        return StringResult(
            result="SYNTHESIS_COMPLETE",
            reason=f"Sequential execution completed and synthesized with {len(results)} tasks"
        )
    
    def _create_enhanced_synthesis(self, responses: List[Dict], original_request: str) -> str:
        """Create an enhanced synthesis when AI synthesis is not available - IMPROVED for 3+ tasks"""
        if len(responses) == 1:
            return f"## 🎯 Analysis Results\n\n{responses[0]['content']}\n\n✅ Analysis completed successfully."
        
        # Enhanced multi-response synthesis
        synthesis = f"## 🎯 Comprehensive Multi-Agent Analysis\n\n"
        synthesis += f"**Original Request:** {original_request[:100]}{'...' if len(original_request) > 100 else ''}\n\n"
        synthesis += f"**Coordination Summary:** {len(responses)} specialist agents collaborated to provide comprehensive insights.\n\n"
        
        # Group responses by agent type for better organization
        data_responses = []
        career_responses = []
        graph_responses = []
        general_responses = []
        
        for response in responses:
            agent_name = response['agent']
            if any(keyword in agent_name.lower() for keyword in ['transaction', 'iron_man', 'orchestrator']):
                data_responses.append(response)
            elif 'career' in agent_name.lower():
                career_responses.append(response)
            elif 'graph' in agent_name.lower():
                graph_responses.append(response)
            else:
                general_responses.append(response)
        
        # Add sections based on available responses
        if data_responses:
            synthesis += "### 📊 Data Analysis Results\n\n"
            for response in data_responses:
                content = response['content']
                # Enhanced data interpretation
                if '[' in content and ']' in content:
                    synthesis += f"**Raw Output:** {content}\n\n"
                    synthesis += f"**Analysis:** This appears to be a probability distribution or classification result from the data analysis model.\n\n"
                else:
                    synthesis += f"{content}\n\n"
        
        if career_responses:
            synthesis += "### � Career Guidance\n\n"
            for response in career_responses:
                synthesis += f"{response['content']}\n\n"
        
        if graph_responses:
            synthesis += "### 📧 Microsoft 365 Operations\n\n"
            for response in graph_responses:
                synthesis += f"{response['content']}\n\n"
        
        if general_responses:
            synthesis += "### 🧠 Additional Insights\n\n"
            for response in general_responses:
                synthesis += f"{response['content']}\n\n"
        
        # Enhanced conclusion
        synthesis += "---\n\n"
        synthesis += "### ✅ Integration Summary\n\n"
        synthesis += f"This comprehensive analysis involved {len(responses)} specialist agents working in coordination:\n\n"
        
        for i, response in enumerate(responses, 1):
            agent_type = "Specialist"
            if "Transaction" in response['agent'] or "Iron_Man" in response['agent']:
                agent_type = "Data Analysis"
            elif "Career" in response['agent']:
                agent_type = "Career Guidance"
            elif "Graph" in response['agent']:
                agent_type = "Microsoft 365"
            elif "General" in response['agent']:
                agent_type = "General Analysis"
            
            synthesis += f"• **Agent {i} ({agent_type}):** {response['task'][:60]}{'...' if len(response['task']) > 60 else ''}\n"
        
        synthesis += f"\n🎯 **All aspects of your request have been addressed through coordinated multi-agent analysis.**"
        
        return synthesis
    
    async def _execute_sequential_task(
        self, 
        user_message: str, 
        tasks: List[Dict[str, Any]], 
        participant_descriptions: Dict[str, str],
        chat_history: ChatHistory
    ) -> StringResult:
        """Execute tasks sequentially with enhanced dependency checking and response capture"""
        
        print(f"🔄 Starting sequential execution of {len(tasks)} tasks")
        
        # ENHANCED: More robust task state initialization
        if not hasattr(self._state, 'sequential_execution_state'):
            self._state.sequential_execution_state = {
                'current_task_index': 0,
                'total_tasks': len(tasks),
                'started_tasks': set(),
                'failed_tasks': set(),
                'execution_round': 0
            }
        
        exec_state = self._state.sequential_execution_state
        exec_state['execution_round'] += 1
        
        # CRITICAL: Enhanced loop prevention for 3+ tasks
        if exec_state['execution_round'] > len(tasks) * 2:  # Allow 2 rounds per task
            self._log_backend_conversation(
                "SequentialExecutor", 
                f"🚨 EXECUTION LIMIT REACHED: {exec_state['execution_round']} rounds for {len(tasks)} tasks",
                "system"
            )
            
            # Force completion of all remaining tasks
            for task in tasks:
                if task['id'] not in self._state.completed_tasks:
                    # Use the last available agent response or create a placeholder
                    last_response = "Task processing completed (execution limit reached)"
                    for msg in reversed(chat_history.messages[-10:]):
                        if (msg.role == AuthorRole.ASSISTANT and msg.content and 
                            len(str(msg.content).strip()) > 10):
                            last_response = str(msg.content).strip()
                            break
                    
                    self._state.completed_tasks[task['id']] = {
                        'agent': task.get('agent', 'GeneralAssistant'),
                        'response': last_response,
                        'task_description': task.get('description', 'Data analysis task'),
                        'forced_completion': True
                    }
            
            # Reset execution state and proceed to synthesis
            del self._state.sequential_execution_state
            
            completed_results = [self._state.completed_tasks[task['id']] for task in tasks]
            return await self._synthesize_sequential_results(completed_results, chat_history)
        
        # Track completed tasks for this session
        completed_results = []
        pending_tasks = []
        
        # ENHANCED: Better task dependency resolution
        for task in tasks:
            task_id = task.get('id', f'task_{len(self._state.completed_tasks)}')
            task['id'] = task_id  # Ensure task has an ID
            
            if task_id in self._state.completed_tasks:
                completed_results.append(self._state.completed_tasks[task_id])
            else:
                # Check dependencies - ensure it's always a list
                dependencies = task.get('depends_on', [])
                if dependencies is None:
                    dependencies = []
                
                # Check if all dependencies are met
                deps_met = all(dep in self._state.completed_tasks for dep in dependencies)
                
                if deps_met or not dependencies:  # Ready to execute
                    pending_tasks.append(task)
        
        self._log_backend_conversation(
            "SequentialExecutor", 
            f"📊 Status: {len(completed_results)} completed, {len(pending_tasks)} pending (round {exec_state['execution_round']})",
            "orchestration"
        )
        
        # Check for completion FIRST
        if len(completed_results) >= len(tasks):
            self._log_backend_conversation(
                "SequentialExecutor", 
                f"✅ All {len(tasks)} tasks completed - proceeding to synthesis",
                "orchestration"
            )
            
            # Clean up execution state
            if hasattr(self._state, 'sequential_execution_state'):
                del self._state.sequential_execution_state
            
            if len(completed_results) > 1:
                return await self._synthesize_sequential_results(completed_results, chat_history)
            else:
                return StringResult(
                    result="GeneralAssistant",
                    reason="Single task completed successfully"
                )
        
        # ENHANCED: Execute next ready task with better selection
        if pending_tasks:
            # Find the task with highest priority that hasn't been started too many times
            next_task = None
            
            for task in pending_tasks:
                task_id = task['id']
                # Check how many times this task has been attempted
                attempt_count = sum(1 for msg in chat_history.messages[-20:] 
                                  if (msg.role == AuthorRole.SYSTEM and 
                                      f"FOCUSED TASK" in str(msg.content) and 
                                      task_id in str(msg.content)))
                
                if attempt_count < 2:  # Allow max 2 attempts per task
                    next_task = task
                    break
            
            # If no task found with low attempt count, pick the first one but mark as final attempt
            if not next_task and pending_tasks:
                next_task = pending_tasks[0]
                self._log_backend_conversation(
                    "SequentialExecutor", 
                    f"⚠️ Final attempt for task: {next_task['id']}",
                    "system"
                )
            
            if next_task:
                exec_state['started_tasks'].add(next_task['id'])
                
                self._log_backend_conversation(
                    "SequentialExecutor", 
                    f"⚡ Executing task {len(exec_state['started_tasks'])}/{len(tasks)}: {next_task['description'][:50]}...",
                    "orchestration"
                )
                
                # Execute the task
                result = await self._execute_single_task(
                    user_message, next_task, participant_descriptions, chat_history
                )
                
                return result
        
        # FALLBACK: If we reach here, something went wrong - force completion
        self._log_backend_conversation(
            "SequentialExecutor", 
            f"🔄 No executable tasks found - checking for missed completions",
            "system"
        )
        
        # Check if there are any substantial responses that weren't captured
        self._mark_completed_tasks_from_history(chat_history)
        
        # Recheck completion status
        updated_completed = [self._state.completed_tasks[task['id']] for task in tasks 
                           if task['id'] in self._state.completed_tasks]
        
        if len(updated_completed) >= len(tasks):
            # Clean up execution state
            if hasattr(self._state, 'sequential_execution_state'):
                del self._state.sequential_execution_state
            
            if len(updated_completed) > 1:
                return await self._synthesize_sequential_results(updated_completed, chat_history)
            else:
                return StringResult(
                    result="GeneralAssistant",
                    reason="Tasks completed after re-check"
                )
        
        # Ultimate fallback - force the next task in line
        for task in tasks:
            if task['id'] not in self._state.completed_tasks:
                self._log_backend_conversation(
                    "SequentialExecutor", 
                    f"🎯 Force-executing remaining task: {task['id']}",
                    "system"
                )
                return await self._execute_single_task(
                    user_message, task, participant_descriptions, chat_history
                )
        
        # All tasks completed (shouldn't reach here, but handle gracefully)
        if hasattr(self._state, 'sequential_execution_state'):
            del self._state.sequential_execution_state
        
        return StringResult(
            result="GeneralAssistant",
            reason="All sequential tasks completed"
        )

    async def _execute_single_task(
        self, 
        user_message: str, 
        task: Dict[str, Any], 
        participant_descriptions: Dict[str, str],
        chat_history: ChatHistory
    ) -> StringResult:
        """Execute a single task with optimized preparation and response capture"""
        
        agent_name = task['agent']
        
        # Validate agent
        if agent_name not in participant_descriptions:
            agent_name = self._enhanced_agent_selection(user_message, participant_descriptions)
            task['agent'] = agent_name
        
        # Check if we've already captured a response for this task from the agent
        if self._check_and_capture_agent_response(task, agent_name, chat_history):
            return StringResult(
                result="GeneralAssistant",
                reason=f"Task completed by {agent_name}, ready for synthesis"
            )
        
        # Check if instruction already prepared
        if task['id'] not in self._state.agent_task_assignments:
            # Prepare instruction with timeout
            try:
                instruction = await asyncio.wait_for(
                    self._prepare_task_instruction(
                        user_message, task, participant_descriptions[agent_name]
                    ),
                    timeout=self._optimizer.timeout_settings['agent_instruction']
                )
                
                self._state.agent_task_assignments[task['id']] = {
                    'agent': agent_name,
                    'instruction': instruction,
                    'task': task,
                    'original_request': user_message
                }
                
            except asyncio.TimeoutError:
                # Use simple fallback instruction
                instruction = f"Please help with: {user_message}"
                
        else:
            instruction = self._state.agent_task_assignments[task['id']]['instruction']
        
        # Add instruction and track timing
        chat_history.add_message(ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content=f"FOCUSED TASK: {instruction}"
        ))
        
        self._state.task_start_times[task['id']] = time.time()
        
        return StringResult(
            result=agent_name,
            reason=f"Executing: {task['description'][:50]}..."
        )

    def _check_and_capture_agent_response(self, task: Dict[str, Any], agent_name: str, chat_history: ChatHistory) -> bool:
        """Check if agent has responded and capture the response for task completion"""
        
        # Check for responses from the agent response callback
        if (hasattr(agent_response_callback, '_latest_responses') and 
            agent_name in agent_response_callback._latest_responses):
            
            response_data = agent_response_callback._latest_responses[agent_name]
            response_content = response_data['content']
            
            # Capture this response as the task completion
            self._state.completed_tasks[task['id']] = {
                'task_id': task['id'],
                'agent': agent_name,
                'response': response_content,
                'task_description': task['description'],
                'timestamp': response_data['timestamp']
            }
            
            # Add the response to chat history for context
            chat_history.add_message(ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                name=agent_name,
                content=response_content
            ))
            
            self._log_backend_conversation(
                "ResponseCapture", 
                f"✅ Captured response from {agent_name} for task {task['id']}",
                "orchestration"
            )
            
            # Clear the captured response to prevent reuse
            del agent_response_callback._latest_responses[agent_name]
            
            return True
        
        # Also check recent chat history for agent responses
        for message in reversed(chat_history.messages[-5:]):
            if (message.role == AuthorRole.ASSISTANT and 
                message.name == agent_name and 
                message.content and 
                len(str(message.content).strip()) > 10):
                
                response_content = str(message.content).strip()
                
                # Only capture if we haven't already completed this task
                if task['id'] not in self._state.completed_tasks:
                    self._state.completed_tasks[task['id']] = {
                        'task_id': task['id'],
                        'agent': agent_name,
                        'response': response_content,
                        'task_description': task['description'],
                        'timestamp': time.time()
                    }
                    
                    self._log_backend_conversation(
                        "ResponseCapture", 
                        f"✅ Captured chat history response from {agent_name} for task {task['id']}",
                        "orchestration"
                    )
                    
                    return True
        
        return False

    async def _prepare_task_instruction(
        self, 
        user_message: str, 
        task: Dict[str, Any], 
        agent_description: str
    ) -> str:
        """Prepare task instruction with caching and optimization"""
        
        # Check cache first
        cache_key = f"{task['id']}_{hash(user_message + agent_description)}"
        if cache_key in self._optimizer.cache:
            return self._optimizer.cache[cache_key]
        
        # Use cached prompt template
        prompt_template = self._optimizer.get_cached_prompt_template('agent_instruction')
        
        # Create simple, fast instruction
        instruction = prompt_template.format(
            agent_name=task['agent'],
            task_description=task['description'],
            original_request=user_message
        )
        
        # For simple requests, use direct instruction
        if len(user_message) < 100 and not any(keyword in user_message.lower() 
                                              for keyword in ['analyze', 'complex', 'detailed', 'comprehensive']):
            instruction = f"Please respond to: {user_message}"
        
        # Cache the result
        self._optimizer.cache[cache_key] = instruction
        
        return instruction

    async def _continue_task_orchestration(
        self, 
        participant_descriptions: Dict[str, str], 
        chat_history: ChatHistory
    ) -> StringResult:
        """Continue with ongoing task orchestration - ENHANCED with aggressive loop prevention"""
        
        # IMMEDIATE loop check before any processing
        if len(chat_history.messages) >= 4:
            recent_assistant_messages = []
            for message in reversed(chat_history.messages[-6:]):
                if message.role == AuthorRole.ASSISTANT and message.content:
                    recent_assistant_messages.append(str(message.content).strip())
                    if len(recent_assistant_messages) >= 3:
                        break
            
            # If we have identical responses, force completion immediately
            if (len(recent_assistant_messages) >= 2 and 
                len(set(recent_assistant_messages[:2])) == 1):
                
                content = recent_assistant_messages[0]  # This is just the content string
                
                self._log_backend_conversation(
                    "ContinuationLoopDetector", 
                    f"🚨 IDENTICAL RESPONSES DETECTED - forcing immediate completion",
                    "system"
                )
                
                # Force mark all tasks as completed
                if self._state.current_task_breakdown:
                    for task in self._state.current_task_breakdown:
                        if task['id'] not in self._state.completed_tasks:
                            self._state.completed_tasks[task['id']] = {
                                'agent': task.get('agent', 'OrchestratorAgent'),
                                'response': content,
                                'task_description': task.get('description', 'Data analysis task')
                            }
                
                # Reset state
                self._reset_task_state_only()
                
                return StringResult(
                    result="READY_FOR_USER_INPUT",
                    reason="Loop detected and resolved - ready for next input"
                )
        
        # FIRST: Mark any recently completed tasks
        self._mark_completed_tasks_from_history(chat_history)
        
        # Check if all tasks are completed
        pending_tasks = [
            task for task in self._state.current_task_breakdown 
            if task['id'] not in self._state.completed_tasks
        ]
        
        self._log_backend_conversation(
            "TaskContinuation", 
            f"🔍 Status: {len(self._state.completed_tasks)} completed, {len(pending_tasks)} pending",
            "system"
        )
        
        if not pending_tasks:
            # All tasks completed
            if len(self._state.current_task_breakdown) > 1:
                # Multi-task completion - synthesis might be needed
                if not getattr(self._state, 'synthesis_completed', False):
                    self._log_backend_conversation(
                        "TaskContinuation", 
                        "🎯 All multi-tasks completed! Triggering synthesis.",
                        "orchestration"
                    )
                    
                    self._state.synthesis_completed = True
                    
                    return StringResult(
                        result="GeneralAssistant",
                        reason="Synthesizing multi-agent results"
                    )
            
            # All tasks completed - ready to reset for next user input
            self._log_backend_conversation(
                "TaskContinuation", 
                "🎉 All tasks completed! Resetting for next user input.",
                "orchestration"
            )
            
            # Reset task state but keep conversation alive
            self._reset_task_state_only()
            
            # Return a signal that we're ready for next user input
            return StringResult(
                result="READY_FOR_USER_INPUT",
                reason="All tasks completed - ready for next user input"
            )
        
        # Check if we're about to assign the same task again (loop prevention)
        for task in pending_tasks:
            task_id = task['id']
            agent_name = task['agent']
            
            # Count how many times this agent has been called recently
            recent_agent_calls = sum(1 for msg in chat_history.messages[-10:] 
                                   if (msg.role == AuthorRole.ASSISTANT and 
                                       msg.name == agent_name))
            
            if recent_agent_calls >= 3:  # Agent called 3+ times recently
                self._log_backend_conversation(
                    "LoopPrevention", 
                    f"🛑 Agent {agent_name} called {recent_agent_calls} times - forcing completion",
                    "system"
                )
                
                # Force mark this task as completed
                self._state.completed_tasks[task_id] = {
                    'agent': agent_name,
                    'response': "Task processing completed (loop prevention)",
                    'task_description': task['description']
                }
                
                # Continue to check for remaining tasks
                continue
        
        # Recalculate pending tasks after loop prevention
        pending_tasks = [
            task for task in self._state.current_task_breakdown 
            if task['id'] not in self._state.completed_tasks
        ]
        
        if not pending_tasks:
            # All tasks now completed after loop prevention
            self._reset_task_state_only()
            return StringResult(
                result="READY_FOR_USER_INPUT",
                reason="All tasks completed (after loop prevention)"
            )
        
        # Find next task to execute (sequential flow only)
        next_task = None
        
        # Find next task with met dependencies
        for task in pending_tasks:
            dependencies = task.get('depends_on', [])
            if not dependencies or all(dep in self._state.completed_tasks for dep in dependencies):
                next_task = task
                break
        
        if not next_task:
            next_task = pending_tasks[0]  # Fallback
            
        self._log_backend_conversation(
            "TaskContinuation", 
            f"🔗 Sequential: Next task '{next_task['id']}'",
            "orchestration"
        )
        
        if not next_task:
            # This shouldn't happen, but handle gracefully
            self._reset_task_state_only()
            return StringResult(
                result="READY_FOR_USER_INPUT",
                reason="No valid next task found"
            )
        
        agent_name = next_task['agent']
        
        # Validate agent exists
        if agent_name not in participant_descriptions:
            agent_name = self._simple_fallback_selection("", participant_descriptions)
            next_task['agent'] = agent_name
            
        # Create instruction if not already done
        if next_task['id'] not in self._state.agent_task_assignments:
            # Get original request
            original_request = ""
            for assignment in self._state.agent_task_assignments.values():
                if 'original_request' in assignment:
                    original_request = assignment['original_request']
                    break
            
            if not original_request:
                # Extract from chat history
                for message in chat_history.messages:
                    if message.role == AuthorRole.USER:
                        original_request = str(message.content)
                        break
            
            # Create focused instruction
            focused_instruction = await self.create_agent_task_instruction(
                original_request, 
                next_task, 
                participant_descriptions[agent_name]
            )
            
            # Store assignment
            self._state.agent_task_assignments[next_task['id']] = {
                'agent': agent_name,
                'instruction': focused_instruction,
                'task': next_task,
                'original_request': original_request
            }
            
            # Add instruction to chat history
            chat_history.add_message(ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=f"FOCUSED TASK: {focused_instruction}"
            ))
        
        self._log_backend_conversation(
            "TaskContinuation", 
            f"📍 Continuing with task '{next_task['id']}' → {agent_name}",
            "orchestration"
        )
        
        return StringResult(
            result=agent_name,
            reason=f"Continuing: {next_task['description'][:50]}..."
        )

    @override
    async def filter_results(self, chat_history: ChatHistory) -> MessageResult:
        """Filter and summarize the conversation results - ENHANCED to prioritize synthesized responses"""
        if not chat_history.messages:
            raise RuntimeError("No messages in chat history")

        print(f"🔍 filter_results: Processing {len(chat_history.messages)} messages")
        
        # PRIORITY 1: Check if sequential synthesis was completed
        if (hasattr(self._state, 'synthesis_completed') and 
            getattr(self._state, 'synthesis_completed', False)):
            
            print("✅ Sequential synthesis detected - looking for synthesized response")
            
            # Look for the synthesized response (should be the last assistant message without agent name)
            for message in reversed(chat_history.messages):
                if (message.role == AuthorRole.ASSISTANT and 
                    message.content and 
                    len(str(message.content).strip()) > 0 and
                    not message.name):  # Synthesized responses don't have agent names
                    
                    print(f"🎯 Returning synthesized response: {str(message.content)[:100]}...")
                    
                    # Reset synthesis flag since we're returning the synthesized result
                    self._state.synthesis_completed = False
                    self._reset_task_state_only()
                    
                    return MessageResult(
                        result=message,
                        reason="Returning manager synthesized response"
                    )
            
            # If we can't find synthesized response, fall back to regular logic
            print("⚠️ Synthesis flag set but no synthesized response found")
            self._state.synthesis_completed = False

        # PRIORITY 2: Check if we just completed parallel synthesis
        if (chat_history.messages and 
            len(chat_history.messages) > 0 and
            chat_history.messages[-1].role == AuthorRole.ASSISTANT and
            "Multi-Agent Collaboration Results" in str(chat_history.messages[-1].content)):
            
            # Return the parallel synthesis result directly
            return MessageResult(
                result=chat_history.messages[-1],
                reason="Parallel synthesis already completed"
            )

        # Mark current task as completed if we're in orchestration mode
        if (self._state.current_task_breakdown and 
            self._state.orchestration_mode != "none"):
            
            # Ensure all tasks are marked as completed
            self._mark_completed_tasks_from_history(chat_history)
            
            completed_count = len(self._state.completed_tasks)
            total_tasks = len(self._state.current_task_breakdown)
            
            self._log_backend_conversation(
                "ResultFilter", 
                f"🔍 Processing results: {completed_count}/{total_tasks} tasks completed",
                "system"
            )
            
            # Process results regardless of completion status to break loops
            if self._state.orchestration_mode == "single" and self._state.completed_tasks:
                # Single task completed - analyze the result
                self._log_backend_conversation(
                    "OrchestrationManager", 
                    f"✅ Single task - analyzing result",
                    "system"
                )
                
                # Get the first completed task result
                completed_task = list(self._state.completed_tasks.values())[0]
                agent_response = completed_task['response']
                
                # Check if this is Azure AI output that needs analysis
                if self.is_azure_ai_agent_output(agent_response):
                    self._log_backend_conversation(
                        "ResultAnalyzer", 
                        f"🔍 Detected Azure AI output - providing analysis",
                        "system"
                    )
                    
                    try:
                        # Generate analysis of the Azure AI output
                        analysis_result = await self.analyze_azure_ai_output(agent_response)
                        
                        # Combine raw output with analysis
                        enhanced_result = f"## 🔍 Transaction Analysis Results\n\n"
                        enhanced_result += f"**Raw Azure AI Output:** {agent_response}\n\n"
                        enhanced_result += f"**Analysis:**\n{analysis_result}"
                        
                        # CRITICAL: Reset state to prepare for next user input
                        self._reset_task_state_only()
                        
                        self._log_backend_conversation(
                            "ResultAnalyzer", 
                            f"✅ Analysis complete - ready for next user input",
                            "system"
                        )
                        
                        return MessageResult(
                            result=ChatMessageContent(
                                role=AuthorRole.ASSISTANT,
                                content=enhanced_result
                            ),
                            reason="Azure AI output analyzed and enhanced"
                        )
                        
                    except Exception as e:
                        self._log_backend_conversation(
                            "ResultAnalyzer", 
                            f"⚠️ Error analyzing Azure AI output: {e}",
                            "system"
                        )
                        # Fall back to returning the raw agent response
                        pass
                
                # Return the raw agent response (either because analysis failed or not Azure AI output)
                self._reset_task_state_only()
                
                return MessageResult(
                    result=ChatMessageContent(
                        role=AuthorRole.ASSISTANT,
                        content=agent_response
                    ),
                    reason="Single task completed - returning agent response"
                )
            
            elif self._state.completed_tasks:
                # Multi-task completion or fallback - generate synthesis
                self._log_backend_conversation(
                    "OrchestrationManager", 
                    f"🎯 Generating synthesis from {len(self._state.completed_tasks)} completed tasks",
                    "system"
                )
                
                synthesis_result = await self._generate_multi_agent_summary(chat_history)
                
                # CRITICAL: Reset state to prepare for next user input
                self._reset_task_state_only()
                
                return synthesis_result
            
            else:
                # No completed tasks but we're here - force basic response to break loop
                self._log_backend_conversation(
                    "ResultFilter", 
                    f"⚠️ No completed tasks found - generating fallback response",
                    "system"
                )
                
                # Find any substantial response in recent messages
                for message in reversed(chat_history.messages[-5:]):
                    if (message.role == AuthorRole.ASSISTANT and 
                        message.content and 
                        len(str(message.content).strip()) > 5):
                        
                        content = str(message.content).strip()
                        
                        # CRITICAL: Reset state to prepare for next user input
                        self._reset_task_state_only()
                        
                        return MessageResult(
                            result=ChatMessageContent(
                                role=AuthorRole.ASSISTANT,
                                content=f"## Response Summary\n\n{content}\n\n---\n\n*Please let me know if you need any clarification or have additional questions.*"
                            ),
                            reason="Fallback response to break processing loop"
                        )

        # Legacy handling for non-orchestrated responses
        # Check for special Azure AI agent outputs
        for message in reversed(chat_history.messages):
            if message.role == AuthorRole.ASSISTANT:
                content = str(message.content)
                
                if self.is_azure_ai_agent_output(content):
                    try:
                        analysis_result = await self.analyze_azure_ai_output(content)
                        combined_result = f"## 🔍 Transaction Analysis Results\n\n"
                        combined_result += f"**Raw Output:** {content}\n\n"
                        combined_result += f"**Analysis:**\n{analysis_result}"
                        
                        return MessageResult(
                            result=ChatMessageContent(
                                role=AuthorRole.ASSISTANT,
                                content=combined_result
                            ),
                            reason="Azure AI output analyzed successfully"
                        )
                    except Exception as e:
                        # Fallback to basic interpretation
                        basic_result = f"## 🔍 Transaction Analysis Results\n\n"
                        basic_result += f"**Raw Output:** {content}\n\n"
                        basic_result += f"**Basic Interpretation:** Probability distribution from transaction analysis model."
                        
                        return MessageResult(
                            result=ChatMessageContent(
                                role=AuthorRole.ASSISTANT,
                                content=basic_result
                            ),
                            reason="Basic Azure AI output interpretation"
                        )
                
                elif self.is_transaction_data_output(content):
                    try:
                        analysis_result = await self.analyze_transaction_output(content)
                        combined_result = f"{content}\n\n{'='*60}\n\n{analysis_result}"
                        
                        return MessageResult(
                            result=ChatMessageContent(
                                role=AuthorRole.ASSISTANT,
                                content=combined_result
                            ),
                            reason="Transaction data analyzed successfully"
                        )
                    except Exception as e:
                        # Fall back to regular handling
                        pass
        
        # Default: Return the last assistant response without additional processing
        for message in reversed(chat_history.messages):
            if (message.role == AuthorRole.ASSISTANT and 
                message.content and 
                len(str(message.content).strip()) > 0):
                
                return MessageResult(
                    result=message,
                    reason="Returning latest agent response"
                )
        
        # Fallback if no assistant response found
        return MessageResult(
            result=ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content="Request processed. How can I help you next?"
            ),
            reason="No substantial agent response found"
        )

    async def _generate_multi_agent_summary(self, chat_history: ChatHistory) -> MessageResult:
        """Generate a comprehensive summary from multi-agent collaboration"""
        try:
            # Collect all completed task results
            task_summaries = []
            for task_id, result in self._state.completed_tasks.items():
                task_summaries.append(f"**{result['task_description']}** (by {result['agent']}):\n{result['response']}")
            
            # Create comprehensive summary
            summary_content = (
                f"## 🎯 Multi-Agent Collaboration Results\n\n"
                f"**Orchestration:** {self._state.orchestration_mode.title()} execution\n"
                f"**Tasks Completed:** {len(self._state.completed_tasks)}\n\n"
                f"### 📋 Results:\n\n" +
                "\n\n---\n\n".join(task_summaries) +
                f"\n\n✅ **All tasks completed successfully through coordinated agent collaboration.**"
            )
            
            # Don't reset state here - let the filter_results method handle it
            
            return MessageResult(
                result=ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content=summary_content
                ),
                reason=f"Multi-agent coordination completed: {len(self._state.completed_tasks)} tasks"
            )
            
        except Exception as e:
            print(f"⚠️ Multi-agent summary generation failed: {e}")
            return MessageResult(
                result=ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content="Multi-agent tasks completed successfully."
                ),
                reason="Summary generation error"
            )

    async def analyze_azure_ai_output(self, raw_output: str) -> str:
        """
        Analyze raw output from the TransactionAgent (like [0, 0.987, 0.012])
        
        Args:
            raw_output: The raw numerical output from the Azure AI Agent
        Returns:
            Analysis interpretation of the raw output
        """
        try:
            analysis_history = ChatHistory()
            
            # Add system prompt for Azure AI Agent output analysis
            analysis_prompt = (
                "You are a financial analysis manager interpreting raw output from a specialized Azure AI Agent. "
                "This agent processes transaction data and returns numerical results, typically in array format like [0, 0.987, 0.012]. "
                "Your role is to interpret these raw numerical outputs:\n\n"
                "🔍 **INTERPRETATION FOCUS:**\n"
                "- Risk scoring and probability analysis\n"
                "- Classification results (fraud/legitimate/suspicious)\n"
                "- Confidence levels and uncertainty measures\n"
                "- Pattern matching scores\n"
                "- Anomaly detection indicators\n\n"
                "💡 **COMMON DATA PATTERNS:**\n"
                "- Boolean flags (TRUE/FALSE) often represent feature presence/absence\n"
                "- Numerical arrays typically contain normalized feature values or probabilities\n"
                "- Output arrays usually represent [class1_prob, class2_prob, class3_prob] for classification\n"
                "- Values near 0 or 1 indicate high confidence in classification\n\n"
                "⚠️ **CONFIDENCE THRESHOLDS:**\n"
                "- High confidence: >= 0.90 (strong classification)\n"
                "- Moderate confidence: 0.70-0.89 (reasonable classification)\n"
                "- Low confidence: < 0.70 (uncertain, needs review)\n\n"
                "📊 **OUTPUT FORMAT:**\n"
                "```\n"
                "=== AZURE AI AGENT OUTPUT ANALYSIS ===\n"
                "[Interpretation of the numerical values with confidence levels]\n\n"
                "=== RISK INTERPRETATION ===\n"
                "[Risk levels based on the output with specific thresholds]\n\n"
                "=== RECOMMENDED ACTIONS ===\n"
                "[Next steps based on the analysis and confidence levels]\n"
                "```\n\n"
                "Interpret the following raw Azure AI Agent output:"
            )
            
            analysis_history.messages.insert(0, ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=analysis_prompt
            ))
            
            analysis_history.add_message(ChatMessageContent(
                role=AuthorRole.USER,
                content=f"Please interpret this Azure AI Agent output:\n\n{raw_output}"
            ))

            response = await self._chat_service.get_chat_message_content(
                analysis_history,
                settings=PromptExecutionSettings(max_tokens=1500, temperature=0.3)
            )
            
            return str(response.content) if response and response.content else "Analysis could not be completed."
            
        except Exception as e:
            print(f"⚠️ Azure AI Agent output analysis failed: {e}")
            return f"Analysis error: {str(e)}"

    async def analyze_transaction_output(self, processed_data: str) -> str:
        """
        Analyze processed transaction data output from the TransactionDataProcessor
        
        Args:
            processed_data: The cleaned and structured data from the transaction agent
        Returns:
            Analysis insights and findings
        """
        try:
            analysis_history = ChatHistory()
            
            # Add system prompt for transaction analysis
            analysis_prompt = (
                "You are a financial analysis manager reviewing processed transaction data. "
                "Your role is to analyze the clean, structured data provided and generate insights:\n\n"
                "🔍 **ANALYSIS FOCUS AREAS:**\n"
                "- Pattern recognition and anomaly detection\n"
                "- Risk assessment and scoring\n"
                "- Compliance monitoring and regulatory flags\n"
                "- Statistical analysis and trend identification\n"
                "- Fraud indicators and suspicious activity detection\n\n"
                "📊 **OUTPUT FORMAT:**\n"
                "```\n"
                "=== TRANSACTION ANALYSIS REPORT ===\n"
                "[Key findings and insights]\n\n"
                "=== RISK ASSESSMENT ===\n"
                "[Risk levels and indicators]\n\n"
                "=== RECOMMENDATIONS ===\n"
                "[Action items and next steps]\n"
                "```\n\n"
                "Analyze the following processed transaction data:"
            )
            
            analysis_history.messages.insert(0, ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=analysis_prompt
            ))
            
            analysis_history.add_message(ChatMessageContent(
                role=AuthorRole.USER,
                content=f"Please analyze this processed transaction data:\n\n{processed_data}"
            ))

            response = await self._chat_service.get_chat_message_content(
                analysis_history,
                settings=PromptExecutionSettings(max_tokens=2000, temperature=0.3)
            )
            
            return str(response.content) if response and response.content else "Analysis could not be completed."
            
        except Exception as e:
            print(f"⚠️ Transaction analysis failed: {e}")
            return f"Analysis error: {str(e)}"

    def is_azure_ai_agent_output(self, content: str) -> bool:
        """
        Check if the content appears to be raw output from the Azure AI Agent
        Enhanced to detect probability arrays like [0, 0.9972063899040222, 0.002793610095977783]
        
        Args:
            content: The message content to check
        Returns:
            True if this looks like Azure AI Agent raw output
        """
        import re
        
        content = content.strip()
        
        # Check for specific probability array patterns (like your example)
        probability_array_pattern = r'\[\s*0\s*,\s*0\.\d+\s*,\s*0\.\d+\s*\]'
        if re.search(probability_array_pattern, content):
            return True
        
        # Check for general array-like patterns [number, number, number]
        array_pattern = r'\[\s*[\d\.\-\s,]+\s*\]'
        if re.search(array_pattern, content):
            # Check if it's purely numerical (likely Azure AI output)
            # Remove the brackets and check if content is mostly numbers
            inner_content = re.sub(r'[\[\]]', '', content)
            numbers_and_spaces = re.sub(r'[^0-9\.\-\s,]', '', inner_content)
            if len(numbers_and_spaces) / len(inner_content) > 0.8:  # 80% numbers
                return True
        
        # Check for simple numerical outputs
        simple_number_pattern = r'^\s*[\d\.\-\s,]+\s*$'
        if re.match(simple_number_pattern, content):
            return True
        
        # Check if the content is very short and primarily numerical
        if len(content) < 100 and any(char.isdigit() for char in content):
            numbers = sum(1 for char in content if char.isdigit() or char in '.-,[]')
            if numbers / len(content) > 0.6:  # More than 60% numerical characters
                return True
        
        return False

    def is_transaction_data_output(self, content: str) -> bool:
        """
        Check if the content appears to be processed transaction data output
        
        Args:
            content: The message content to check
        Returns:
            True if this looks like transaction data output
        """
        transaction_indicators = [
            "=== PROCESSED TRANSACTION DATA ===",
            "=== DATA SUMMARY ===",
            "=== DATA QUALITY REPORT ===",
            "Transaction ID",
            "Total Records:",
            "Date Range:",
            "Total Amount:",
            "Currencies:",
            "Transaction Types:"
        ]
        
        content_upper = content.upper()
        indicator_matches = sum(1 for indicator in transaction_indicators if indicator.upper() in content_upper)
        
        # If we have 3+ indicators, this is likely transaction data output
        return indicator_matches >= 3

    def _log_backend_conversation(self, speaker: str, message: str, conversation_type: str = "orchestration"):
        """Log backend conversations between manager and agents"""
        if not self._state.enable_backend_logging:
            return
            
        timestamp = time.strftime("%H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "speaker": speaker,
            "message": message,
            "type": conversation_type,
            "session_id": self._state.orchestration_session_id
        }
        
        self._state.backend_conversations.append(log_entry)
        
        # Only show critical orchestration messages to reduce noise
        if conversation_type == "orchestration" and any(keyword in message for keyword in ["Starting", "Orchestrating", "Selected", "routing"]):
            print(f"🎯 {message}")
        elif conversation_type == "system" and "Error" in message:
            print(f"❌ {message}")
        # Suppress most backend logging for cleaner output
    
    def _start_orchestration_session(self, user_request: str):
        """Start a new orchestration session with unique ID"""
        self._state.orchestration_session_id = str(uuid.uuid4())[:8]
        self._state.backend_conversations = []
        self._state.manager_agent_communications = {}
        
        self._log_backend_conversation(
            "OrchestrationManager", 
            f"🚀 Starting new orchestration session {self.orchestration_session_id}",
            "system"
        )
        self._log_backend_conversation(
            "OrchestrationManager", 
            f"📥 User Request: '{user_request}'",
            "system"
        )
    
    def _log_agent_communication(self, agent_name: str, task_id: str, instruction: str, response: str = None):
        """Log communication between manager and specific agent"""
        if agent_name not in self._state.manager_agent_communications:
            self._state.manager_agent_communications[agent_name] = []
        comm_entry = {
            "task_id": task_id,
            "instruction": instruction,
            "response": response,
            "timestamp": time.strftime("%H:%M:%S")
        }
        self._state.manager_agent_communications[agent_name].append(comm_entry)
    
    def _print_backend_summary(self):
        """Print a summary of backend conversations"""
        if not self._state.backend_conversations:
            return
            
        print("\n" + "="*80)
        print("🔍 BACKEND ORCHESTRATION SUMMARY")
        print("="*80)
        
        for entry in self._state.backend_conversations:
            timestamp = entry["timestamp"]
            speaker = entry["speaker"]
            message = entry["message"]
            conv_type = entry["type"]
            
            if conv_type == "system":
                print(f"[{timestamp}] 🏗️  {speaker}: {message}")
            elif conv_type == "orchestration":
                print(f"[{timestamp}] 🧠 {speaker}: {message}")
            elif conv_type == "task_assignment":
                print(f"[{timestamp}] 📋 {speaker}: {message}")
            elif conv_type == "agent_response":
                print(f"[{timestamp}] 🤖 {speaker}: {message}")
        
        print("="*80)
    
    def reset_orchestration_state(self):
        """Reset orchestration state after successful task completion"""
        self._log_backend_conversation(
            "StateManager", 
            "🔄 Resetting orchestration state for next conversation",
            "system"
        )
        
        # Call the new method that only resets task state
        self._reset_task_state_only()
    
    def _reset_task_state_only(self):
        """Reset only task execution state while maintaining conversation context - ENHANCED for 3+ tasks"""
        # Reset core task execution state
        self._state.orchestration_mode = "none"
        self._state.current_task_breakdown = []
        self._state.completed_tasks = {}
        self._state.agent_task_assignments = {}
        
        # ENHANCED: Reset sequential execution state for 3+ tasks
        if hasattr(self._state, 'sequential_execution_state'):
            del self._state.sequential_execution_state
        
        # Reset task-specific flags
        if hasattr(self._state, 'synthesis_triggered'):
            self._state.synthesis_triggered = False
        # NOTE: Don't reset synthesis_completed here - let filter_results handle it
        
        # Reset termination flags
        self._state.force_terminate_for_analysis = False
        
        # Keep conversation session alive - DON'T reset conversation context
        self._state.conversation_active = True
        
        # Clear only recent backend conversations to prevent memory bloat but keep context
        if len(self._state.backend_conversations) > 100:  # Increased threshold for 3+ task debugging
            self._state.backend_conversations = self._state.backend_conversations[-20:]  # Keep more history for 3+ tasks
        
        # Reset task timing information
        if hasattr(self._state, 'task_start_times'):
            self._state.task_start_times = {}
        
        # Keep manager-agent communications for debugging but clear old ones
        if hasattr(self._state, 'manager_agent_communications'):
            # Keep only the most recent communications for each agent
            for agent_name in list(self._state.manager_agent_communications.keys()):
                comms = self._state.manager_agent_communications[agent_name]
                if len(comms) > 5:  # Keep last 5 communications per agent
                    self._state.manager_agent_communications[agent_name] = comms[-5:]
        
        self._log_backend_conversation(
            "StateManager", 
            "🔄 Task state reset complete - ready for next orchestration (3+ task capable)",
            "system"
        )

    def _mark_completed_tasks_from_history(self, chat_history: ChatHistory):
        """Mark tasks as completed based on substantial responses - ENHANCED for 3+ task coordination"""
        if not self._state.current_task_breakdown:
            return
        
        # ENHANCED: More aggressive loop detection and task completion for 3+ tasks
        recent_responses = []
        agent_response_counts = {}
        
        # Check last 15 messages for better detection with 3+ tasks
        for message in reversed(chat_history.messages[-15:]):
            if message.role == AuthorRole.ASSISTANT and message.name and message.content:
                content = str(message.content).strip()
                agent_name = message.name
                
                recent_responses.append((agent_name, content))
                agent_response_counts[agent_name] = agent_response_counts.get(agent_name, 0) + 1
                
                if len(recent_responses) >= 10:  # Check more responses for 3+ task scenarios
                    break
        
        # CRITICAL: Enhanced repetition detection for 3+ tasks
        if len(recent_responses) >= 2:
            # Check for ANY identical responses (even just 2 in a row)
            if recent_responses[0] == recent_responses[1]:
                agent_name, content = recent_responses[0]
                
                self._log_backend_conversation(
                    "TaskCompletion", 
                    f"🔄 Immediate repetition detected from {agent_name} - marking ALL agent tasks complete",
                    "system"
                )
                
                # Mark ALL tasks for this agent as completed to prevent further loops
                for task in self._state.current_task_breakdown:
                    task_id = task.get('id', f'task_{len(self._state.completed_tasks)}')
                    assignment = self._state.agent_task_assignments.get(task_id, {})
                    
                    if (assignment.get('agent') == agent_name and 
                        task_id not in self._state.completed_tasks):
                        
                        self._state.completed_tasks[task_id] = {
                            'agent': agent_name,
                            'response': content,
                            'task_description': task.get('description', 'Data analysis task'),
                            'completion_reason': 'repetition_detected'
                        }
                        
                        self._log_backend_conversation(
                            "TaskCompletion", 
                            f"✅ Auto-completed task '{task_id}' due to repetition",
                            "orchestration"
                        )
                
                return  # Exit early to break the loop
            
            # Check for response loops (3+ identical responses from same agent)
            if len(recent_responses) >= 3:
                first_response = recent_responses[0]
                identical_count = sum(1 for resp in recent_responses[:3] if resp == first_response)
                
                if identical_count >= 3:  # 3 identical responses
                    agent_name, content = first_response
                    
                    self._log_backend_conversation(
                        "TaskCompletion", 
                        f"🔄 Response loop detected from {agent_name} ({identical_count} identical) - forcing completion",
                        "system"
                    )
                    
                    # Complete all tasks for this agent
                    for task in self._state.current_task_breakdown:
                        task_id = task.get('id', f'task_{len(self._state.completed_tasks)}')
                        assignment = self._state.agent_task_assignments.get(task_id, {})
                        
                        if (assignment.get('agent') == agent_name and 
                            task_id not in self._state.completed_tasks):
                            
                            self._state.completed_tasks[task_id] = {
                                'agent': agent_name,
                                'response': content,
                                'task_description': task.get('description', 'Data analysis task'),
                                'completion_reason': 'loop_prevention'
                            }
                            
                            self._log_backend_conversation(
                                "TaskCompletion", 
                                f"✅ Force-completed task '{task_id}' due to loop",
                                "orchestration"
                            )
                    
                    return
        
        # ENHANCED: Check for agents with multiple responses but no task completion
        for agent_name, count in agent_response_counts.items():
            if count >= 2:  # Agent has responded multiple times
                # Find tasks assigned to this agent that aren't completed
                uncompleted_agent_tasks = []
                for task in self._state.current_task_breakdown:
                    task_id = task.get('id', f'task_{len(self._state.completed_tasks)}')
                    assignment = self._state.agent_task_assignments.get(task_id, {})
                    
                    if (assignment.get('agent') == agent_name and 
                        task_id not in self._state.completed_tasks):
                        uncompleted_agent_tasks.append((task_id, task))
                
                # If agent has responded multiple times but has uncompleted tasks, force completion
                if uncompleted_agent_tasks:
                    # Get the most recent response from this agent
                    latest_response = None
                    for agent, content in recent_responses:
                        if agent == agent_name:
                            latest_response = content
                            break
                    
                    if latest_response:
                        self._log_backend_conversation(
                            "TaskCompletion", 
                            f"🔄 Agent {agent_name} has {count} responses but {len(uncompleted_agent_tasks)} uncompleted tasks - force completing",
                            "system"
                        )
                        
                        # Complete the first uncompleted task for this agent
                        task_id, task = uncompleted_agent_tasks[0]
                        
                        self._state.completed_tasks[task_id] = {
                            'agent': agent_name,
                            'response': latest_response,
                            'task_description': task.get('description', 'Data analysis task'),
                            'completion_reason': 'multiple_responses'
                        }
                        
                        self._log_backend_conversation(
                            "TaskCompletion", 
                            f"✅ Force-completed task '{task_id}' due to multiple responses",
                            "orchestration"
                        )
        
        # ENHANCED: Time-based task completion for stalled scenarios
        if len(self._state.current_task_breakdown) >= 3:  # For 3+ task scenarios
            # If we have substantial responses but many uncompleted tasks, be more aggressive
            uncompleted_count = len([task for task in self._state.current_task_breakdown 
                                   if task.get('id', f'task_{len(self._state.completed_tasks)}') not in self._state.completed_tasks])
            
            if uncompleted_count >= 2 and len(recent_responses) >= 3:
                self._log_backend_conversation(
                    "TaskCompletion", 
                    f"🔄 3+ task scenario: {uncompleted_count} uncompleted tasks with {len(recent_responses)} responses - forcing completion",
                    "system"
                )
                
                # Match recent responses to uncompleted tasks
                for task in self._state.current_task_breakdown:
                    task_id = task.get('id', f'task_{len(self._state.completed_tasks)}')
                    if task_id in self._state.completed_tasks:
                        continue
                    
                    assignment = self._state.agent_task_assignments.get(task_id, {})
                    agent_name = assignment.get('agent', 'GeneralAssistant')
                    
                    # Find a response from this agent
                    agent_response = None
                    for resp_agent, resp_content in recent_responses:
                        if resp_agent == agent_name:
                            agent_response = resp_content
                            break
                    
                    # If no specific response found, use the most recent substantial response
                    if not agent_response and recent_responses:
                        agent_response = recent_responses[0][1]
                    
                    if agent_response:
                        self._state.completed_tasks[task_id] = {
                            'agent': agent_name,
                            'response': agent_response,
                            'task_description': task.get('description', 'Data analysis task'),
                            'completion_reason': 'force_completion_3plus'
                        }
                        
                        self._log_backend_conversation(
                            "TaskCompletion", 
                            f"✅ Force-completed task '{task_id}' in 3+ task scenario",
                            "orchestration"
                        )
                        
                        # Only complete one task per call to avoid race conditions
                        break
        
        # Normal task completion detection (only if no aggressive actions taken)
        latest_assistant_message = None
        for message in reversed(chat_history.messages[-5:]):  # Check last 5 messages
            if message.role == AuthorRole.ASSISTANT and message.name and message.content:
                latest_assistant_message = message
                break
        
        if not latest_assistant_message:
            return
        
        agent_name = latest_assistant_message.name
        content = str(latest_assistant_message.content).strip()
        
        # Enhanced detection for specialized agents
        is_specialized_agent = any(keyword in agent_name.lower() 
                                 for keyword in ['transaction', 'iron_man', 'orchestrator'])
        
        # ENHANCED: Better response validation for different agent types
        if is_specialized_agent:
            # For specialized agents, any meaningful output is considered substantial
            import re
            array_pattern = r'\[\s*[\d\.\-\s,]+\s*\]'
            has_array = re.search(array_pattern, content)
            has_numbers = any(char.isdigit() for char in content)
            
            is_substantial_response = (
                has_array or 
                (has_numbers and len(content) > 5) or 
                len(content) > 10
            )
        else:
            # For other agents, use normal criteria but be more lenient
            is_substantial_response = (
                len(content) > 20 and  # Reduced from 50
                not content.lower().startswith(('hello', 'hi', 'how can i help')) and
                'error code:' not in content.lower() and
                'rate limit' not in content.lower() and
                not content.startswith('An error has occurred')
            )
        
        if is_substantial_response:
            # Find the most recent task this agent was assigned to (and not already completed)
            for task in self._state.current_task_breakdown:
                task_id = task.get('id', f'task_{len(self._state.completed_tasks)}')
                assignment = self._state.agent_task_assignments.get(task_id, {})
                
                if (assignment.get('agent') == agent_name and 
                    task_id not in self._state.completed_tasks):
                    
                    # Mark task as completed
                    self._state.completed_tasks[task_id] = {
                        'agent': agent_name,
                        'response': content,
                        'task_description': task.get('description', 'Data analysis task'),
                        'completion_reason': 'substantial_response'
                    }
                    
                    self._log_backend_conversation(
                        "TaskCompletion", 
                        f"✅ Task '{task_id}' completed by {agent_name} (substantial response)",
                        "orchestration"
                    )
                    
                    self._log_backend_conversation(
                        "TaskCompletion", 
                        f"📈 Progress: {len(self._state.completed_tasks)}/{len(self._state.current_task_breakdown)} tasks complete",
                        "system"
                    )
                    
                    # Update communication log
                    if agent_name in self._state.manager_agent_communications:
                        for comm in self._state.manager_agent_communications[agent_name]:
                            if comm['task_id'] == task_id and not comm.get('response'):
                                comm['response'] = content[:100] + "..." if len(content) > 100 else content
                    
                    # Only mark one task per agent response
                    break

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics about routing decisions and agent performance"""
        return {
            "total_routings": len(self._state.completed_tasks),
            "agent_success_rates": {"TransactionAgent": 1.0, "GeneralAssistant": 1.0, "CareerAdvisor": 1.0, "GraphAssistant": 1.0},
            "agent_usage": {"TransactionAgent": 0, "GeneralAssistant": 0, "CareerAdvisor": 0, "GraphAssistant": 0},
            "domain_distribution": {"data_analysis": 0, "general": 0, "career": 0, "microsoft365": 0},
            "complexity_distribution": {"simple": 0, "medium": 0, "complex": 0}
        }

    def update_routing_success(self, agent_name: str, success: bool):
        """Update routing success rate for learning"""
        # Placeholder for future implementation
        pass
        
        self._log_backend_conversation(
            "RoutingLearning",
            f"📊 Updated {agent_name} success rate: {'✅' if success else '❌'}",
            "system"
        )


def agent_response_callback(message: ChatMessageContent) -> None:
    """OPTIMIZED: Fast agent response display with minimal overhead"""
    agent_name = message.name or "Agent"
    content = str(message.content or "")
    
    # Skip empty responses
    if not content.strip():
        return
    
    # Fast agent type detection
    agent_emojis = {
        'Foundry': '🔬',
        'Azure': '🤖', 
        'Transaction': '🤖',
        'Iron_Man': '🤖',
        'Career': '💼',
        'Surface': '💼',
        'Graph': '📧',
        'General': '💬'
    }
    
    # Find emoji quickly
    emoji = '�'  # default
    for key, emoji_val in agent_emojis.items():
        if key in agent_name:
            emoji = emoji_val
            break
    
    # Show only progress, not content (content will be synthesized)
    print(f"✅ {emoji} {agent_name} completed task")
    print("-" * 30)
    
    # SILENT CAPTURE: Store response for manager synthesis (no user display)
    if not hasattr(agent_response_callback, '_latest_responses'):
        agent_response_callback._latest_responses = {}
    
    agent_response_callback._latest_responses[agent_name] = {
        'content': content,
        'timestamp': time.time(),
        'message': message
    }


def _is_raw_numerical_output(content: str) -> bool:
    """Helper function to detect raw numerical output from Azure AI Agent"""
    content = content.strip()
    
    # Check for array-like patterns [number, number, number]
    array_pattern = r'\[\s*[\d\.\-\s,]+\s*\]'
    if re.search(array_pattern, content):
        return True
        
    # Check for simple numerical outputs
    simple_number_pattern = r'^\s*[\d\.\-\s,]+\s*$'
    if re.match(simple_number_pattern, content):
        return True
        
    # Check if the content is very short and primarily numerical
    if len(content) < 50 and any(char.isdigit() for char in content):
        numbers = sum(1 for char in content if char.isdigit() or char in '.-')
        if numbers / len(content) > 0.5:  # More than 50% numerical characters
            return True
            
    return False


async def interactive_foundry_orchestration():
    """
    Interactive session using Group Chat Orchestration with Azure Foundry Agent
    """
    print("=== Azure Foundry Group Chat Orchestration ===")
    print("🧠 AI-Powered Task Decomposition & Multi-Agent Coordination")
    print("🤖 All requests analyzed by AI - from simple greetings to complex multi-domain tasks")
    print("🚀 Routes each task to the most appropriate specialist agent")
    print("🔗 Coordinates parallel and sequential task execution\n")

    # Configuration - updated with API key from transaction analysis agent
    endpoint = "https://aif-e2edemo.services.ai.azure.com/api/projects/project-thanos"
    api_key = "D0b8REYc0wXONcnJu7fmj6kyFciM5XTrxjzJmoL1PtAXiqw1GHjXJQQJ99BFACYeBjFXJ3w3AAAAACOGpETv"  # Updated API key from transaction agent
    agent_id = "asst_JE3DIZwUr7MWbb7KCM4OHxV4"
    model_name = "gpt-4o"
    
    # Copilot Studio Bot Secret for Career Advice
    bot_secret = "EIyanQMLVDOeluIdbvfddzUpO2mO14oGy8MKH04lprY08zqu0fqOJQQJ99BFAC4f1cMAArohAAABAZBS2U6n.CXzSOwihZ7dl5h9sI70U5VGr7ydVp75Nfr69psUNlP6KmQneluqoJQQJ99BFAC4f1cMAArohAAABAZBS3VyD"

    if not all([endpoint, api_key, agent_id]):
        print("❌ Error: Missing required configuration")
        return

    try:
        # Initialize orchestration with Copilot Studio bot
        orchestration_system = AzureFoundryOrchestration(
            endpoint=endpoint, 
            api_key=api_key, 
            agent_id=agent_id, 
            model_name=model_name,
            bot_secret=bot_secret
        )
        
        # Create agents
        agents = await orchestration_system.get_agents()
        
        # Create group chat orchestration with improved settings
        group_chat_orchestration = GroupChatOrchestration(
            members=agents,
            manager=SmartGroupChatManager(
                service=orchestration_system.chat_service,
                max_rounds=15,  # Increased to allow multi-agent coordination
            ),
            agent_response_callback=agent_response_callback,
        )

        # Create and start runtime
        runtime = InProcessRuntime()
        runtime.start()
        
        print("✅ Orchestration system ready!")
        print("\n=== Interactive Session Started ===")
        print("Commands: [quit] to exit, [help] for assistance")
        print("💡 The smart manager maintains context throughout the conversation")
        print("🔄 Conversation continues until you type 'quit'\n")

        # Conversation state tracking
        conversation_active = True
        
        while conversation_active:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ("quit", "exit", "bye"):
                    print("\n👋 Ending orchestration session...")
                    conversation_active = False
                    break
                
                if user_input.lower() == "analytics":
                    print("\n" + "="*90)
                    print("📊 ENHANCED ROUTING ANALYTICS")
                    print("="*90)
                    
                    try:
                        analytics = group_chat_orchestration.manager.get_routing_analytics()
                        
                        print(f"\n📈 **Routing Overview:**")
                        print(f"  • Total Routings: {analytics['total_routings']}")
                        
                        print(f"\n🤖 **Agent Usage & Success Rates:**")
                        for agent_name in analytics['agent_success_rates']:
                            success_rate = analytics['agent_success_rates'][agent_name]
                            usage_count = analytics['agent_usage'].get(agent_name, 0)
                            print(f"  • {agent_name}: {success_rate:.1%} success rate ({usage_count} uses)")
                        
                        print(f"\n🎯 **Domain Distribution:**")
                        for domain, count in analytics['domain_distribution'].items():
                            print(f"  • {domain.title()}: {count} requests")
                        
                        print(f"\n⚙️ **Complexity Distribution:**")
                        for complexity, count in analytics['complexity_distribution'].items():
                            print(f"  • {complexity.title()}: {count} requests")
                        
                        print("="*90)
                        
                    except Exception as e:
                        print(f"❌ Error retrieving analytics: {e}")
                        print("="*90)
                    continue
                    
                if user_input.lower() == "backend":
                    print("\n" + "="*90)
                    print("🔍 BACKEND CONVERSATION LOG")
                    print("="*90)
                    
                    if hasattr(group_chat_orchestration.manager, '_state') and group_chat_orchestration.manager._state.backend_conversations:
                        for entry in group_chat_orchestration.manager._state.backend_conversations:
                            timestamp = entry["timestamp"]
                            speaker = entry["speaker"]
                            message = entry["message"]
                            conv_type = entry["type"]
                            
                            if conv_type == "system":
                                print(f"[{timestamp}] 🏗️  {speaker}: {message}")
                            elif conv_type == "orchestration":
                                print(f"[{timestamp}] 🧠 {speaker}: {message}")
                            elif conv_type == "task_assignment":
                                print(f"[{timestamp}] 📋 {speaker}: {message}")
                            elif conv_type == "agent_response":
                                print(f"[{timestamp}] 🤖 {speaker}: {message}")
                        
                        print("="*90)
                        
                        # Show agent communication summary
                        if hasattr(group_chat_orchestration.manager, '_state') and group_chat_orchestration.manager._state.manager_agent_communications:
                            print("\n📞 MANAGER-AGENT COMMUNICATIONS:")
                            print("-" * 50)
                            for agent, comms in group_chat_orchestration.manager._state.manager_agent_communications.items():
                                print(f"\n🤖 **{agent}**:")
                                for comm in comms:
                                    print(f"  📋 Task: {comm['task_id']}")
                                    print(f"  📤 Instruction: {comm['instruction'][:80]}...")
                                    if comm['response']:
                                        print(f"  📥 Response: {comm['response'][:80]}...")
                                    print(f"  ⏰ Time: {comm['timestamp']}")
                                    print()
                    else:
                        print("No backend conversations logged yet.")
                        print("="*90)
                    continue
                    
                if user_input.lower() == "help":
                    print("\n📋 Available Commands:")
                    print("• Type any question or request (simple or complex)")
                    print("• [quit/exit/bye] - End session")
                    print("• [backend] - View backend orchestration conversations")
                    print("• [analytics] - View enhanced routing analytics and agent performance")
                    print("• The system will automatically decompose and route tasks")
                    print("\n🧠 **Enhanced Agent Selection**")
                    print("  • Sophisticated multi-step reasoning for agent routing")
                    print("  • Confidence scoring and alternative agent suggestions")
                    print("  • Domain expertise matching with detailed capability profiles")
                    print("  • Learning from routing success/failure patterns")
                    print("\n🔍 **NEW: Backend Conversation Monitoring**")
                    print("  • Real-time logging of manager-agent communications")
                    print("  • Task decomposition and assignment tracking")
                    print("  • Agent response monitoring and orchestration flow")
                    print("  • Use 'backend' command to view detailed conversation log")
                    print("\n🧠 **Intelligent Task Decomposition**")
                    print("  • **Single Task**: Simple requests go to one specialist agent")
                    print("  • **Parallel Tasks**: Multiple independent tasks run simultaneously")
                    print("  • **Sequential Tasks**: Dependent tasks run in order")
                    print("\n🤖 Agent Specialists:")
                    print("• **GeneralAssistant**: General questions, explanations, technical analysis")
                    print("• **TransactionAgent**: Azure AI transaction data processing")
                    print("• **CareerAdvisor**: Microsoft career advice (Copilot Studio)")
                    print("• **GraphAssistant**: Microsoft 365 operations (Graph API)")
                    print("\n🔥 **Multi-Task Examples:**")
                    print("  • 'Find user John Smith and send him an email about the project'")
                    print("    → Task 1: GraphAssistant finds user")
                    print("    → Task 2: GraphAssistant sends email")
                    print("\n  • 'Create a todo list for Q1 planning and analyze transaction data'")
                    print("    → Task 1: GraphAssistant creates todo list")
                    print("    → Task 2: TransactionAgent analyzes data (parallel)")
                    print("\n  • 'Give me career advice and create a project folder'")
                    print("    → Task 1: CareerAdvisor provides guidance")
                    print("    → Task 2: GraphAssistant creates folder (parallel)")
                    print("\n📧 Microsoft Graph Operations (GraphAssistant):")
                    print("\n🧑‍💼 USER MANAGEMENT:")
                    print("  • 'List all users in organization'")
                    print("  • 'Find user John Smith'")
                    print("  • 'Search for users named Sarah'")
                    print("  • 'Get email address for Michael Johnson'")
                    print("  • 'Show organization directory'")
                    print("\n📧 EMAIL OPERATIONS:")
                    print("  • 'Send email to marketing@company.com about project update'")
                    print("  • 'Get my latest inbox messages'")
                    print("  • 'Search emails containing quarterly report'")
                    print("  • 'Show my email folders'")
                    print("  • 'Find emails from last week about budget'")
                    print("\n✅ TASK & TODO MANAGEMENT:")
                    print("  • 'Create a todo list called Project Tasks'")
                    print("  • 'Add task: Review quarterly budget'")
                    print("  • 'Show all my todo lists'")
                    print("  • 'List tasks from my Work list'")
                    print("\n📁 ONEDRIVE FILE OPERATIONS:")
                    print("  • 'Create folder called Project Documents'")
                    print("  • 'Get my OneDrive information'")
                    print("  • 'Show my drive root folder'")
                    print("\n🔧 SMART ROUTING KEYWORDS:")
                    print("  User ops: 'users', 'directory', 'find user', 'employee directory'")
                    print("  Email ops: 'email', 'inbox', 'send mail', 'outlook', 'compose'")
                    print("  Task ops: 'todo', 'task', 'checklist', 'task list'")
                    print("  File ops: 'folder', 'onedrive', 'drive', 'documents'")
                    print("  Career: 'career', 'job', 'resume', 'interview'")
                    print("  Transaction: 'transaction', 'analyze', 'data'")
                    continue
                    
                if not user_input:
                    continue

                print(f"\n🚀 Processing request: '{user_input}'")
                print("📡 Smart manager analyzing and orchestrating response...\n")

                try:
                    # Invoke orchestration with proper error handling
                    orchestration_result = await group_chat_orchestration.invoke(
                        task=user_input,
                        runtime=runtime,
                    )

                    # Get results
                    result = await orchestration_result.get()
                    if result:
                        print(f"\n{result}")
                        print("=" * 60)
                        print("💬 Ready for your next question or request!")
                        print("=" * 60)
                    
                    # The conversation continues - the smart manager maintains context
                    
                except Exception as invoke_error:
                    print(f"❌ Error during orchestration: {invoke_error}")
                    print("🔄 Smart manager is still active - you can try another request")
                    # Don't break - continue the conversation

            except KeyboardInterrupt:
                print("\n👋 Session interrupted by user.")
                conversation_active = False
                break
            except Exception as e:
                print(f"❌ Error processing request: {e}")
                print("🔄 Smart manager is still active - you can try another request")
                # Continue the conversation even after errors

        
        # End of conversation loop
        print("\n🎯 Smart manager conversation session ended")
        print("📊 Final stats: Context maintained throughout the session")
        
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user.")
    except Exception as e:
        print(f"❌ Error during orchestration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Comprehensive cleanup
        print("\n🧹 Cleaning up resources...")
        
        try:
            # Close DirectLine client sessions properly
            if hasattr(orchestration_system, 'directline_client') and orchestration_system.directline_client:
                await orchestration_system.directline_client.close()
                print("✅ DirectLine client sessions closed")
        except Exception as e:
            print(f"⚠️ DirectLine cleanup warning: {e}")
        
        try:
            # Cleanup Azure services and HTTP clients
            if hasattr(orchestration_system, 'chat_service'):
                # Close any underlying HTTP clients
                if hasattr(orchestration_system.chat_service, '_client'):
                    if hasattr(orchestration_system.chat_service._client, 'close'):
                        await orchestration_system.chat_service._client.close()
                        print("✅ Azure chat service HTTP client closed")
        except Exception as e:
            print(f"⚠️ Azure service cleanup warning: {e}")
        
        try:
            # Stop the runtime properly
            await runtime.stop_when_idle()
            print("✅ Runtime stopped")
        except Exception as e:
            print(f"⚠️ Runtime shutdown warning: {e}")
        
        print("✅ Orchestration system shutdown complete.")

if __name__ == "__main__":
    asyncio.run(interactive_foundry_orchestration())




    
