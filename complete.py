import os
import sys
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', ''),
    openai_api_version=os.environ.get('AZURE_OPENAI_API_VERSION', ''),
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT', ''),
    api_key=os.environ.get('AZURE_OPENAI_API_KEY', ''),
    temperature=0
)


def load_transcript(filename):
    """Load a meeting transcript from a file."""
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def extract_attendees(state):
    """Extract attendees from the transcript"""
    transcript = state.get('transcript', '')
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""Extract the names and roles of all attendees from the following meeting transcript.
        Return ONLY a valid Python list of strings, with each string in the format 'Name (Role)'.
        Do not include any other text or explanations.

        Transcript:
        {transcript}
        """
    )
    prompt_str = prompt.format(transcript=transcript)

    try:
        # Get response from Azure OpenAI
        response = llm.invoke(prompt_str).content
        print(f"LLM Response for attendees: {response}")

        # Parse the response - handle both list literals and JSON formats
        if response.strip().startswith('[') and response.strip().endswith(']'):
            try:
                attendees = eval(response)
            except:
                attendees = json.loads(response)
        else:
            # Try to extract a list from the text if not properly formatted
            import re
            list_pattern = r'\[(.+)\]'
            match = re.search(list_pattern, response, re.DOTALL)
            if match:
                list_str = match.group(1)
                # Convert to proper Python list format
                try:
                    attendees = eval(f"[{list_str}]")
                except:
                    attendees = []
            else:
                attendees = []

        # Ensure we have a list of strings
        if isinstance(attendees, list):
            attendees = [str(a).strip() for a in attendees]
        else:
            attendees = []

    except Exception as e:
        print(f"Error parsing attendees: {e}")
        attendees = []

    # Return updated state
    return {**state, 'attendees': attendees}


def extract_key_points(state):
    """Extract key discussion points from the transcript"""
    transcript = state.get('transcript', '')
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""Extract the key discussion points from the following meeting transcript.
        Return ONLY a valid Python list of strings, with each string representing one key point discussed.
        Focus on the main topics, decisions, and important considerations mentioned.
        Do not include any other text or explanations.

        Transcript:
        {transcript}
        """
    )
    prompt_str = prompt.format(transcript=transcript)

    try:
        # Get response from Azure OpenAI
        response = llm.invoke(prompt_str).content
        print(f"LLM Response for key points: {response}")

        # Parse the response - handle both list literals and JSON formats
        if response.strip().startswith('[') and response.strip().endswith(']'):
            try:
                key_points = eval(response)
            except:
                key_points = json.loads(response)
        else:
            # Try to extract a list from the text if not properly formatted
            import re
            list_pattern = r'\[(.+)\]'
            match = re.search(list_pattern, response, re.DOTALL)
            if match:
                list_str = match.group(1)
                # Convert to proper Python list format
                try:
                    key_points = eval(f"[{list_str}]")
                except:
                    key_points = []
            else:
                key_points = []

        # Ensure we have a list of strings
        if isinstance(key_points, list):
            key_points = [str(kp).strip() for kp in key_points]
        else:
            key_points = []

    except Exception as e:
        print(f"Error parsing key points: {e}")
        key_points = []

    # Return updated state
    return {**state, 'key_points': key_points}


def extract_action_items(state):
    """Extract action items from the transcript"""
    transcript = state.get('transcript', '')
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""Extract all action items and their assignees from the following meeting transcript.
        Return ONLY a valid Python list of dictionaries, where each dictionary has the keys 'action' and 'assignee'.
        The 'action' value should be the task to be completed, and the 'assignee' value should be the person responsible.
        Focus on explicit tasks that were assigned to specific people.
        Do not include any other text or explanations.

        Transcript:
        {transcript}
        """
    )
    prompt_str = prompt.format(transcript=transcript)

    try:
        # Get response from Azure OpenAI
        response = llm.invoke(prompt_str).content
        print(f"LLM Response for action items: {response}")

        # Parse the response - handle both list literals and JSON formats
        if response.strip().startswith('[') and response.strip().endswith(']'):
            try:
                action_items = eval(response)
            except:
                action_items = json.loads(response)
        else:
            # Try to extract a list from the text if not properly formatted
            import re
            list_pattern = r'\[(.+)\]'
            match = re.search(list_pattern, response, re.DOTALL)
            if match:
                list_str = match.group(1)
                # Convert to proper Python list format
                try:
                    action_items = eval(f"[{list_str}]")
                except:
                    action_items = []
            else:
                action_items = []

        # Ensure we have a list of dictionaries with the right keys
        if isinstance(action_items, list):
            # Standardize the keys if they're not already 'action' and 'assignee'
            standardized_items = []
            for item in action_items:
                if isinstance(item, dict):
                    # Look for common key variations
                    action = item.get('action') or item.get('task') or item.get('item') or ''
                    assignee = item.get('assignee') or item.get('owner') or item.get('person') or ''
                    standardized_items.append({'action': str(action).strip(), 'assignee': str(assignee).strip()})
            action_items = standardized_items
        else:
            action_items = []

    except Exception as e:
        print(f"Error parsing action items: {e}")
        action_items = []

    # Return updated state
    return {**state, 'action_items': action_items}


def build_minutes(state):
    """Build meeting minutes from extracted information"""
    attendees = state.get('attendees', [])
    key_points = state.get('key_points', [])
    action_items = state.get('action_items', [])

    # Build minutes
    minutes = "# Meeting Minutes\n\n"

    # Add attendees section
    minutes += "## Attendees\n"
    if attendees:
        for attendee in attendees:
            minutes += f"- {attendee}\n"
    else:
        minutes += "- No attendees recorded\n"
    minutes += "\n"

    # Add key points section
    minutes += "## Key Discussion Points\n"
    if key_points:
        for point in key_points:
            minutes += f"- {point}\n"
    else:
        minutes += "- No key points recorded\n"
    minutes += "\n"

    # Add action items section
    minutes += "## Action Items\n"
    if action_items:
        for item in action_items:
            action = item.get('action', '')
            assignee = item.get('assignee', '')
            if action and assignee:
                minutes += f"- {action} (Assigned to: {assignee})\n"
            elif action:
                minutes += f"- {action}\n"
    else:
        minutes += "- No action items recorded\n"

    # Return updated state
    return {**state, 'minutes': minutes}


class MeetingState(dict):
    pass


# Create a workflow using a different approach for LangGraph 0.0.19
graph = StateGraph(MeetingState)

# Add all processing nodes
graph.add_node('extract_attendees', extract_attendees)
graph.add_node('extract_key_points', extract_key_points)
graph.add_node('extract_action_items', extract_action_items)
graph.add_node('build_minutes', build_minutes)

# Define a conditional function that always returns the same next node
# This is a workaround for the sequential flow
def next_step(state):
    return "next"

# Create the sequential flow
graph.add_conditional_edges('extract_attendees', next_step,
                          {'next': 'extract_key_points'})
graph.add_conditional_edges('extract_key_points', next_step,
                          {'next': 'extract_action_items'})
graph.add_conditional_edges('extract_action_items', next_step,
                          {'next': 'build_minutes'})

# Add a final edge from build_minutes to complete the flow
def final_step(state):
    # Return the completed state
    return "complete"

graph.add_conditional_edges('build_minutes', final_step,
                          {'complete': 'extract_attendees'})

# Set the entry point to the first node
graph.set_entry_point('extract_attendees')
compiled_graph = graph.compile()


def test():
    # Create a detailed transcript with five speakers discussing a patient-centric chatbot
    transcript = '''
Meeting Transcript: Patient-Centric Chatbot Project Kickoff
Date: April 15, 2025

Sarah (Project Manager): Good morning everyone! Thanks for joining our kickoff meeting for the new patient-centric chatbot project. Let's start with quick introductions. I'm Sarah, the project manager for this initiative.

David (UX Designer): Hi team, I'm David, the UX designer. I'll be focusing on creating intuitive conversation flows and ensuring the chatbot feels natural and empathetic when interacting with patients.

Michael (Backend Developer): Hello, I'm Michael. I'll be handling the backend integration with our patient database and electronic health records system.

Jennifer (Healthcare Specialist): Hi everyone, I'm Jennifer. As our healthcare specialist, I'll ensure all medical information provided by the chatbot is accurate and compliant with healthcare regulations.

Rachel (AI Engineer): And I'm Rachel, the AI engineer. I'll be working on the natural language processing models and training the chatbot to understand patient queries effectively.

Sarah: Great! Now let's discuss our project goals. We need to build a chatbot that can help patients schedule appointments, answer basic health questions, and provide medication reminders.

Jennifer: I think we should prioritize patient privacy. We need to ensure the chatbot is HIPAA compliant and handles sensitive information appropriately.

Michael: Absolutely. I'll need to work closely with the IT security team to implement proper encryption and data protection measures.

David: From a user experience perspective, we should make the chatbot accessible to elderly patients who might not be tech-savvy. Simple language and clear navigation options will be key.

Rachel: I agree. We should also consider implementing voice recognition for patients who have difficulty typing.

Sarah: These are all excellent points. What about our timeline? I'm thinking we should aim for a prototype in 6 weeks.

Michael: That's ambitious but doable if we focus on core functionality first. I can have the database integration ready in 3 weeks.

Rachel: I'll need at least 4 weeks to train the initial NLP models and test them with sample patient queries.

David: I can have the conversation flows and UI mockups ready in 2 weeks for everyone to review.

Jennifer: I'll need to compile a list of common patient questions and appropriate responses. That will take me about 2 weeks, and then I'll need to review all the medical content before we launch.

Sarah: Perfect. Let's also plan for a mid-project review in 3 weeks to make sure we're on track.

Jennifer: One more thing - we should consider how the chatbot will handle emergency situations. We need clear escalation paths for urgent medical concerns.

Rachel: Good point. We could implement keyword recognition for emergency terms and immediately provide contact information for emergency services.

Michael: We should also have a feature that connects patients directly to a human healthcare provider if the chatbot can't adequately address their concerns.

David: I'll make sure that option is prominently displayed in the interface.

Sarah: These are all great ideas. Let's summarize our action items before we wrap up. Rachel, can you prepare a document outlining the NLP approach and training methodology?

Rachel: Yes, I'll have that ready by the end of the week.

Sarah: Michael, please schedule a meeting with the IT security team to discuss HIPAA compliance requirements.

Michael: Will do. I'll set that up for early next week.

Sarah: Jennifer, please start compiling that list of common patient questions and appropriate responses.

Jennifer: I'll get started right away and share a draft for everyone to review.

Sarah: David, we'll need those UI mockups and conversation flows in two weeks.

David: No problem, I'll have preliminary designs ready for our next meeting.

Sarah: Excellent! I'll create a shared project timeline and send it out later today. Let's reconvene next week to check on our progress. Thank you all for your input!
'''

    print(f"Test transcript: \n{transcript}")

    # Create a proper MeetingState instance
    test_state = MeetingState(transcript=transcript)
    print(f"Initial state: {test_state}")

    # Invoke the graph with the state
    print("\nExtracting attendees...")
    result = extract_attendees(test_state)
    print(f"After attendees extraction: {result}")

    print("\nExtracting key points...")
    result = extract_key_points(result)
    print(f"After key points extraction: {result}")

    print("\nExtracting action items...")
    result = extract_action_items(result)
    print(f"After action items extraction: {result}")

    print("\nBuilding minutes...")
    result = build_minutes(result)
    print(f"After building minutes: {result}")

    # Print the final minutes
    if 'minutes' in result:
        print("\nFinal Meeting Minutes:")
        print(result['minutes'])
    else:
        print("\nNo minutes were generated!")
        print(f"Available keys in result: {result.keys()}")



if __name__ == "__main__":
    test()
