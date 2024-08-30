import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from googleapiclient.discovery import build
import json

# Load environment variables
load_dotenv()

# Set up YouTube API client
youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))

class FitnessChatbot:
    def __init__(self, persist_directory, processed_data_dir, expert_name):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.persist_directory = persist_directory
        self.processed_data_dir = processed_data_dir
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.vectorstore = None
        self.qa = None
        self.user_profile = None
        self.expert_name = expert_name
        self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        else:
            self.process_videos()
        
        self.initialize_qa_chain()

    def process_videos(self):
        texts = []
        metadatas = []

        for filename in os.listdir(self.processed_data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.processed_data_dir, filename), 'r', encoding='utf-8') as f:
                    video_data = json.load(f)
                    texts.append(video_data['combined_text'])
                    metadatas.append({
                        'title': video_data['title'],
                        'video_id': video_data['id'],
                        'description': video_data['description']
                    })

        self.vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

    def initialize_qa_chain(self):
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )

    def set_user_profile(self, profile):
        self.user_profile = profile

    def get_response_and_recommendations(self, query):
        expert_name = self.expert_name
        user_profile = self.user_profile

        system_prompt = f"""
        You are {expert_name}, an AI fitness assistant based on Jeff Cavaliere's expertise and AthleanX YouTube content. Your role is to provide personalized fitness advice and recommend relevant AthleanX videos.

        Key points:
        1. Use knowledge from AthleanX YouTube videos to answer questions and give advice.
        2. Tailor responses to the user's fitness level: {user_profile['fitness_level']}.
        3. Consider the user's goals: {user_profile['goals']}.
        4. Keep in mind their preferred workouts: {user_profile['preferred_workouts']}.
        5. Adapt advice based on their equipment access: {user_profile['equipment']}.
        6. Prioritize recommending specific exercises, techniques, or workouts from AthleanX videos.
        7. Avoid referring to external websites or programs. Focus on providing advice and information available in the YouTube content.
        8. If relevant, mention video titles or topics that would be helpful, but don't invent video names.

        Respond in Jeff Cavaliere's typical style: direct, informative, and encouraging. Use "we" to create a sense of shared journey in fitness.

        User query: {query}

        Provide a thorough response addressing the query and incorporating relevant AthleanX principles and techniques.
        """

        # Generate the response
        result = self.qa({"question": system_prompt})
        
        # Generate video recommendations
        recommendations = self.get_video_recommendations(query)
        
        return result['answer'], recommendations


    def get_video_recommendations(self, query):
        # Combine the query with user profile information for better context
        contextualized_query = f"{query} {self.user_profile['fitness_level']} {self.user_profile['goals']} {self.user_profile['preferred_workouts']} {self.user_profile['equipment']}"
        
        results = self.vectorstore.similarity_search(contextualized_query, k=3)
        recommendations = []
        for doc in results:
            video_id = doc.metadata.get('video_id', '')
            title = doc.metadata.get('title', 'No title available')
            description = doc.metadata.get('description', 'No description available')
            
            thumbnail_url = self.get_video_thumbnail(video_id)
            summary = self.generate_concise_summary(title, description)
            
            recommendations.append({
                'title': title,
                'video_id': video_id,
                'description': description,
                'summary': summary,
                'youtube_link': f"https://www.youtube.com/watch?v={video_id}",
                'thumbnail_url': thumbnail_url
            })
        return recommendations

    def generate_concise_summary(self, title, description):
        prompt = f"Provide a very concise 1-2 sentence summary of this AthleanX video, highlighting its key points or exercises. Title: {title}. Description: {description}"
        response = self.llm.invoke(prompt)
        return response.content

    def get_video_thumbnail(self, video_id):
        try:
            response = youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            
            if response['items']:
                thumbnails = response['items'][0]['snippet']['thumbnails']
                return thumbnails['medium']['url']  # You can choose 'default', 'medium', or 'high'
        except Exception as e:
            print(f"Error fetching thumbnail: {e}")
        
        return "https://via.placeholder.com/320x180.png?text=No+Thumbnail"

    def generate_video_summary(self, title, description):
        prompt = f"Summarize the following fitness video in 2-3 sentences. Title: {title}. Description: {description}"
        response = self.llm.invoke(prompt)
        return response.content

# Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Fitness AI Chatbot",
        page_icon="💪",
        # layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for gradient background and text styling
    st.markdown("""
        <style>
        body {
            background: linear-gradient(to right, #00aaff, #00ffaa);
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 3.5em;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            margin-top: 0;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px #000000;
        }
        .welcome-message {
            font-size: 1.5em;
            color: #ffffff;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 1px 1px 2px #000000;
        }
        </style>
        """, unsafe_allow_html=True)

    # Display title and welcome message with custom CSS classes
    st.markdown('<h1 class="title">Fitness AI Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-message">Welcome to your personalized fitness assistant!</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### What's this App about?
        
        - AI-powered fitness coach
        - Answers to fitness questions
        - Video recommendations from AthleanX
        - Tailored guidance for all fitness level
        - Expert advice from Jeff Cavaliere AI
        """)
    
    with col2:
        st.markdown("""
        ### Who is this App for?
        
        - Beginners starting their fitness journey
        - Intermediate exercisers optimizing workouts
        - Advanced athletes seeking specific tips
        - Effective, science-based exercise enthusiasts
        - Anyone looking for expert fitness advice
        """)
    st.markdown('<p class="welcome-message">Let\'s start your personalized fitness journey!</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize session state
    if 'chatbot' not in st.session_state:
        persist_directory = "data/chroma_db"
        processed_data_dir = 'data/processed'
        st.session_state.chatbot = FitnessChatbot(persist_directory, processed_data_dir, expert_name="Jeff Cavaliere AI")

    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User profile setup
    if not st.session_state.user_profile:
        st.header("Fitness Profile Setup")
        with st.form("user_profile_form"):
            fitness_level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])
            # goals = st.text_input("Fitness Goals (e.g., weight loss, muscle gain)")
            goals = st.selectbox("Fitness Goals", [
            "Weight loss",
            "Muscle gain",
            "Improve overall fitness",
            "Increase strength",
            "Enhance flexibility",
            "Boost endurance",
            "Athletic performance"
        ])
            preferred_workouts = st.selectbox("Preferred Workouts", [
            "Cardio",
            "Strength training",
            "HIIT (High-Intensity Interval Training)",
            "Bodyweight exercises",
            "Weightlifting",
            "Running"
        ])
            equipment = st.selectbox("Equipment Access", ["Home", "Gym"])
            
            submit_button = st.form_submit_button("Start Chatting")
            if submit_button:
                st.session_state.user_profile = {
                    "fitness_level": fitness_level.lower(),
                    "goals": goals.lower(),
                    "preferred_workouts": preferred_workouts.lower(),
                    "equipment": equipment.lower()
                }
                st.session_state.chatbot.set_user_profile(st.session_state.user_profile)
                st.experimental_rerun()

    # Chat interface
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # User input
        user_input = st.chat_input("Type your message here...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response, recommendations = st.session_state.chatbot.get_response_and_recommendations(user_input)
                response_placeholder.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Display recommendations
            if recommendations:
                st.subheader("Video Recommendations:")
                cols = st.columns(3)
                for idx, rec in enumerate(recommendations):
                    with cols[idx]:
                        st.image(rec['thumbnail_url'], use_column_width=True)
                        st.write(f"**{rec['title']}**")
                        st.write(f"[Watch on YouTube]({rec['youtube_link']})")
                        st.write("**Summary:**")
                        st.write(rec['summary'])
                        st.write("---")

if __name__ == "__main__":
    main()