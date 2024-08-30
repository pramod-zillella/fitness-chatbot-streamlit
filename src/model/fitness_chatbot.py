import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import json

# Load environment variables
load_dotenv()

class FitnessChatbot:
    def __init__(self, persist_directory, processed_data_dir):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.persist_directory = persist_directory
        self.processed_data_dir = processed_data_dir
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.vectorstore = None
        self.qa = None
        self.user_profile = {}
        
        self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        if os.path.exists(self.persist_directory):
            print("Loading existing Chroma database...")
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            if self.vectorstore._collection.count() == 0:
                print("Chroma database is empty. Creating new vectors...")
                self.process_videos()
        else:
            print("Chroma database not found. Creating new vectors...")
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

    def get_user_profile(self):
        print("To provide better recommendations, please answer a few questions:")
        self.user_profile['fitness_level'] = input("What's your current fitness level? (Beginner/Intermediate/Advanced): ").lower()
        self.user_profile['goals'] = input("What are your primary fitness goals? (e.g., weight loss, muscle gain, overall health): ").lower()
        self.user_profile['preferred_workouts'] = input("What types of workouts do you prefer? (e.g., cardio, strength training, HIIT): ").lower()
        self.user_profile['equipment'] = input("Do you have access to gym equipment or prefer home workouts? (Gym/Home): ").lower()

    def get_response_and_recommendations(self, query):
        result = self.qa({"question": query})
        recommendations = self.get_video_recommendations(query)
        return result['answer'], recommendations

    def get_video_recommendations(self, query):
        combined_query = f"{query} {self.user_profile['fitness_level']} {self.user_profile['goals']} {self.user_profile['preferred_workouts']} {self.user_profile['equipment']}"
        results = self.vectorstore.similarity_search(combined_query, k=3)
        
        recommendations = []
        for doc in results:
            video_id = doc.metadata.get('video_id', 'No ID available')
            recommendations.append({
                'title': doc.metadata.get('title', 'No title available'),
                'video_id': video_id,
                'description': doc.metadata.get('description', 'No description available'),
                'youtube_link': f"https://www.youtube.com/watch?v={video_id}"
            })
        
        return recommendations

def main():
    persist_directory = "data/chroma_db"
    processed_data_dir = 'data/processed'
    
    chatbot = FitnessChatbot(persist_directory, processed_data_dir)
    chatbot.get_user_profile()

    print("\nFitness AI: Hello! I'm your fitness AI assistant. What would you like to know about fitness?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Fitness AI: Goodbye! Stay healthy!")
            break

        response, recommendations = chatbot.get_response_and_recommendations(user_input)
        print(f"Fitness AI: {response}\n")
        
        print("Video Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   YouTube Link: {rec['youtube_link']}")
            print(f"   Description: {rec['description'][:100]}...\n")

if __name__ == "__main__":
    main()