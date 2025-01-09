import os
import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from system_message import messages
import seaborn as sns

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

# Initialize mode in session state
if "mode" not in st.session_state:
	st.session_state.mode = "chat"  # Default mode is "chat"

# File upload widget
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file:
	st.session_state.df = read_data(uploaded_file)
	st.write("DataFrame Preview:")
	st.dataframe(st.session_state.df.head())
	st.success("File uploaded successfully")

# Sidebar buttons to switch modes
if st.sidebar.button("Switch to Chat Mode"):
	st.session_state.mode = "chat"

if st.sidebar.button("Switch to Visualization Mode"):
	st.session_state.mode = "visualization"

# Display chat history
for message in st.session_state.chat_history:
	with st.chat_message(message["role"]):
		if "content" in message:
			st.markdown(message["content"])
		elif "plot" in message:
			st.markdown(message["plot"])

# Input field for user's message
user_prompt = st.chat_input("Your message")

# Chat Mode
if st.session_state.mode == "chat":
	if user_prompt:
		# Add user's message to chat history and display it
		st.chat_message("user").markdown(user_prompt)
		st.session_state.chat_history.append({"role": "user", "content": user_prompt})

		# Initialize the general LLM
		general_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=os.environ["OPENAI_API_KEY"],
								 max_tokens=1500)

		# Prepare messages for the agent
		messages = messages + st.session_state.chat_history[-5:]


		def respond_to_user():
			if st.session_state.df is not None:

				# Initialize the LLM for data
				llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.environ["OPENAI_API_KEY"],
								 max_tokens=1500)

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

# Visualization Mode
elif st.session_state.mode == "visualization":
	st.write("### Visualization Mode")
	if st.session_state.df is not None:
		# Let the user select columns for visualization
		numeric_columns = st.session_state.df.select_dtypes(include=["number"]).columns.tolist()
		categorical_columns = st.session_state.df.select_dtypes(include=["object", "category"]).columns.tolist()

		# User selects columns
		selected_columns = st.multiselect(
			"Select columns for visualization",
			numeric_columns + categorical_columns
		)

		# User selects chart type
		chart_type = st.selectbox(
			"Select chart type",
			["Histogram", "Bar Plot", "Scatter Plot", "Line Plot", "Box Plot", "Heatmap"]
		)

		# Generate visualization based on user selection
		if selected_columns and chart_type:
			st.write(f"### {chart_type} for Selected Columns")

			if chart_type == "Histogram":
				for column in selected_columns:
					if column in numeric_columns:
						fig, ax = plt.subplots(figsize=(8, 4))
						st.session_state.df[column].hist(ax=ax)
						ax.set_title(f"Histogram of {column}")
						st.pyplot(fig)

			elif chart_type == "Bar Plot":
				for column in selected_columns:
					if column in categorical_columns:
						fig, ax = plt.subplots(figsize=(8, 4))
						st.session_state.df[column].value_counts().plot(kind="bar", ax=ax)
						ax.set_title(f"Bar Plot of {column}")
						st.pyplot(fig)

			elif chart_type == "Scatter Plot":
				if len(selected_columns) >= 2:
					x_axis = st.selectbox("Select X-axis", selected_columns)
					y_axis = st.selectbox("Select Y-axis", selected_columns)
					fig, ax = plt.subplots(figsize=(8, 4))
					sns.scatterplot(data=st.session_state.df, x=x_axis, y=y_axis, ax=ax)
					ax.set_title(f"Scatter Plot of {x_axis} vs {y_axis}")
					st.pyplot(fig)
				else:
					st.warning("Select at least 2 numeric columns for a scatter plot.")

			elif chart_type == "Line Plot":
				if len(selected_columns) >= 2:
					x_axis = st.selectbox("Select X-axis", selected_columns)
					y_axis = st.selectbox("Select Y-axis", selected_columns)
					# Convert x_axis to datetime if it contains date strings
					if st.session_state.df[x_axis].dtype == "object":
						try:
							st.session_state.df[x_axis] = pd.to_datetime(st.session_state.df[x_axis])
						except ValueError:
							st.warning(
								f"Column '{x_axis}' contains invalid date strings. Please select a valid datetime column for the X-axis.")

					# Ensure numeric data for y_axis
					if st.session_state.df[y_axis].dtype == "object":
						st.warning(
							f"Column '{y_axis}' contains non-numeric data. Please select a numeric column for the Y-axis.")
					else:
						fig, ax = plt.subplots(figsize=(10, 6))
						sns.lineplot(data=st.session_state.df, x=x_axis, y=y_axis, ax=ax)
						ax.set_title(f"Line Plot of {x_axis} vs {y_axis}")
						st.pyplot(fig)
				else:
					st.warning("Select at least 2 numeric columns for a line plot.")

			elif chart_type == "Box Plot":
				if len(selected_columns) >= 1:
					fig, ax = plt.subplots(figsize=(8, 4))
					sns.boxplot(data=st.session_state.df[selected_columns], ax=ax)
					ax.set_title(f"Box Plot of {', '.join(selected_columns)}")
					st.pyplot(fig)
				else:
					st.warning("Select at least 1 numeric column for a box plot.")

			elif chart_type == "Heatmap":
				if len(selected_columns) >= 2:
					fig, ax = plt.subplots(figsize=(8, 4))
					sns.heatmap(st.session_state.df[selected_columns].corr(), annot=True, ax=ax)
					ax.set_title(f"Heatmap of {', '.join(selected_columns)}")
					st.pyplot(fig)
				else:
					st.warning("Select at least 2 numeric columns for a heatmap.")
	else:
		st.warning("Please upload a file to generate visualizations.")
