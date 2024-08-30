# import os
# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma  # Updated import
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import uuid

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# class FitnessChatbot:
#     def __init__(self, persist_directory, processed_data_dir):
#         self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         self.persist_directory = persist_directory
#         self.processed_data_dir = processed_data_dir
#         self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
#         self.vectorstore = None
#         self.qa = None
#         self.user_sessions = {}
        
#         self.load_or_create_vectorstore()

#     def load_or_create_vectorstore(self):
#         if os.path.exists(self.persist_directory):
#             print("Loading existing Chroma database...")
#             self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
#             if self.vectorstore._collection.count() == 0:
#                 print("Chroma database is empty. Creating new vectors...")
#                 self.process_videos()
#         else:
#             print("Chroma database not found. Creating new vectors...")
#             self.process_videos()
        
#         self.initialize_qa_chain()

#     def process_videos(self):
#         texts = []
#         metadatas = []

#         for filename in os.listdir(self.processed_data_dir):
#             if filename.endswith('.json'):
#                 with open(os.path.join(self.processed_data_dir, filename), 'r', encoding='utf-8') as f:
#                     video_data = json.load(f)
#                     texts.append(video_data['combined_text'])
#                     metadatas.append({
#                         'title': video_data['title'],
#                         'video_id': video_data['id'],
#                         'description': video_data['description']
#                     })

#         self.vectorstore = Chroma.from_texts(
#             texts=texts,
#             metadatas=metadatas,
#             embedding=self.embeddings,
#             persist_directory=self.persist_directory
#         )

#     def initialize_qa_chain(self):
#         self.qa = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=self.vectorstore.as_retriever()
#         )

#     def create_user_session(self, user_profile):
#         session_id = str(uuid.uuid4())
#         self.user_sessions[session_id] = {
#             'profile': user_profile,
#             'memory': ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         }
#         return session_id

#     def get_response_and_recommendations(self, session_id, query):
#         user_session = self.user_sessions.get(session_id)
#         if not user_session:
#             return "Session not found. Please create a new session.", []

#         result = self.qa({"question": query, "chat_history": user_session['memory'].chat_memory})
#         user_session['memory'].chat_memory.add_user_message(query)
#         user_session['memory'].chat_memory.add_ai_message(result['answer'])

#         recommendations = self.get_video_recommendations(query, user_session['profile'])
#         return result['answer'], recommendations

#     def get_video_recommendations(self, query, user_profile):
#         combined_query = f"{query} {user_profile['fitness_level']} {user_profile['goals']} {user_profile['preferred_workouts']} {user_profile['equipment']}"
#         results = self.vectorstore.similarity_search(combined_query, k=3)
        
#         recommendations = []
#         for doc in results:
#             video_id = doc.metadata.get('video_id', 'No ID available')
#             recommendations.append({
#                 'title': doc.metadata.get('title', 'No title available'),
#                 'video_id': video_id,
#                 'description': doc.metadata.get('description', 'No description available'),
#                 'youtube_link': f"https://www.youtube.com/watch?v={video_id}"
#             })
        
#         return recommendations

# chatbot = FitnessChatbot("data/chroma_db", 'data/processed')

# @app.route('/')
# def home():
#     return "Welcome to the Fitness Chatbot API. Use /create_session to start a new session and /chat to interact with the chatbot."

# @app.route('/create_session', methods=['POST'])
# def create_session():
#     user_profile = request.json
#     session_id = chatbot.create_user_session(user_profile)
#     return jsonify({"session_id": session_id})

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     session_id = data.get('session_id')
#     query = data.get('query')
    
#     if not session_id or not query:
#         return jsonify({"error": "Missing session_id or query"}), 400

#     response, recommendations = chatbot.get_response_and_recommendations(session_id, query)
#     return jsonify({
#         "response": response,
#         "recommendations": recommendations
#     })

# if __name__ == "__main__":
#     app.run(debug=True)

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
    def __init__(self, persist_directory, processed_data_dir):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.persist_directory = persist_directory
        self.processed_data_dir = processed_data_dir
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.vectorstore = None
        self.qa = None
        self.user_profile = None
        
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
        combined_query = f"{query} {self.user_profile['fitness_level']} {self.user_profile['goals']} {self.user_profile['preferred_workouts']} {self.user_profile['equipment']}"
        result = self.qa({"question": combined_query})
        recommendations = self.get_video_recommendations(combined_query)
        return result['answer'], recommendations

    def get_video_recommendations(self, query):
        results = self.vectorstore.similarity_search(query, k=3)
        recommendations = []
        for doc in results:
            video_id = doc.metadata.get('video_id', '')
            title = doc.metadata.get('title', 'No title available')
            description = doc.metadata.get('description', 'No description available')
            
            # Fetch thumbnail
            thumbnail_url = self.get_video_thumbnail(video_id)
            
            # Generate summary
            summary = self.generate_video_summary(title, description)
            
            recommendations.append({
                'title': title,
                'video_id': video_id,
                'description': description,
                'summary': summary,
                'youtube_link': f"https://www.youtube.com/watch?v={video_id}",
                'thumbnail_url': thumbnail_url
            })
        return recommendations

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
    st.title("Fitness AI Chatbot")

    # Initialize session state
    if 'chatbot' not in st.session_state:
        persist_directory = "data/chroma_db"
        processed_data_dir = 'data/processed'
        st.session_state.chatbot = FitnessChatbot(persist_directory, processed_data_dir)

    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User profile setup
    if not st.session_state.user_profile:
        st.header("Fitness Profile Setup")
        with st.form("user_profile_form"):
            fitness_level = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])
            goals = st.text_input("Fitness Goals (e.g., weight loss, muscle gain)")
            preferred_workouts = st.text_input("Preferred Workouts (e.g., cardio, strength training)")
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