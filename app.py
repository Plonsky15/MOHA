import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from io import BytesIO
from system_message import messages

load_dotenv()

# Streamlit web app configuration
st.set_page_config(
	page_title="Labelvie chatbot",
	page_icon="💬",
	layout="wide"
)

# Streamlit page title
st.markdown("<h1 style='text-align: center; color: #1e1e1e';></h1>", unsafe_allow_html=True)

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Function to read the uploaded file into a DataFrame
def read_data(file):
	if file.name.endswith(".csv"):
		return pd.read_csv(file, low_memory=False)
	else:
		return pd.read_excel(file)


# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
	st.session_state.chat_history = []

# Initialize DataFrame in session state
if "df" not in st.session_state:
	st.session_state.df = None

# Initialize memory for conversation history
if "memory" not in st.session_state:
	st.session_state.memory = ConversationBufferMemory()

# # Image logo
# base_dir = os.path.dirname(__file__)
# image_path = os.path.join(base_dir, "moha-removebg-preview.png")
# st.sidebar.image(image_path, width=250)

# File upload widget
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file:
	st.session_state.df = read_data(uploaded_file)
	st.write("DataFrame Preview:")
	st.dataframe(st.session_state.df.head())
	st.success("File uploaded successfully")

# Display chat history
for message in st.session_state.chat_history:
	with st.chat_message(message["role"]):
		if "content" in message:
			st.markdown(message["content"])
		elif "plot" in message:
			st.markdown(message["plot"])

# Input field for user's message
user_prompt = st.chat_input("Your message")

if user_prompt:
	# Add user's message to chat history and display it
	st.chat_message("user").markdown(user_prompt)
	st.session_state.chat_history.append({"role": "user", "content": user_prompt})

	# Initialize the general LLM
	general_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=os.environ["OPENAI_API_KEY"],
							 max_tokens=1500)

	# Prepare messages for the agent
	messages = messages

	def respond_to_user():
		if st.session_state.df is not None:

			# Initialize the LLM for data
			llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.environ["OPENAI_API_KEY"], max_tokens=1500)

			# Create pandas agent
			pandas_df_agent = create_pandas_dataframe_agent(
				llm,
				st.session_state.df,
				agent_type=AgentType.OPENAI_FUNCTIONS,
				memory=st.session_state.memory,
				tools=[PythonREPL()],
				verbose=True,
				return_intermediate_steps=True,
				allow_dangerous_code=True,
				max_iterations=30,
				max_time=60,
				handle_parsing_errors=True
			)

			# Prepare messages for the agent and get response
			messages.append({"role": "user", "content": user_prompt})
			response = pandas_df_agent.invoke(messages)

			if "plot" in user_prompt:
				fig = plt.gcf()  # Get current figure
				buf = BytesIO()  # convert to bytesio
				fig.savefig(buf, format="png")
				buf.seek(0)

				# Append plot to chat history
				st.session_state.chat_history.append({"role": "assistant", "plot": buf})
				st.image(buf, use_column_width=True)
			return response["output"]

		else:
			messages.append({"role": "user", "content": user_prompt})
			response = general_llm(messages)
			return response.content


	# Get response from the appropriate agent
	assistant_response = respond_to_user()

	# Add the assistant's response to chat history and display it
	st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
	with st.chat_message("assistant"):
		st.markdown(assistant_response)