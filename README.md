# Fitness AI Chatbot: Your Personal AthleanX Assistant

<p align="center">
  <img src="demo.gif" alt="Fitness AI Chatbot Demo" width="600">
</p>

Welcome to the Fitness AI Chatbot, an intelligent assistant powered by Jeff Cavaliere's AthleanX expertise. This Streamlit-based application provides personalized fitness advice and recommends relevant AthleanX YouTube videos based on your fitness profile and queries.

## Features

- 🏋️‍♂️ Personalized Fitness Profiling
- 🤖 AI-powered Chatbot for Fitness Queries
- 🎥 AthleanX Video Recommendations with Thumbnails and Summaries
- 🔍 Smart Content Retrieval using LangChain and Chroma
- 🎨 User-friendly Interface with Streamlit

## How It Works

1. **Create Your Fitness Profile**: Input your fitness level, goals, preferred workouts, and available equipment.
2. **Chat with the AI**: Ask any fitness-related questions or seek workout advice.
3. **Get Personalized Responses**: Receive tailored advice based on AthleanX principles and your profile.
4. **Discover Relevant Videos**: Get recommendations for AthleanX videos that match your query and fitness needs.

## Technology Stack

- Streamlit: For the web application interface
- LangChain: For building the conversational AI
- Google Generative AI: Powering the language model
- Chroma: For efficient similarity search and retrieval
- YouTube Data API: For fetching video information and thumbnails

## Setup and Installation

1. Clone the repository:
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in a `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key
   YOUTUBE_API_KEY=your_youtube_api_key
   ```
4. Run the app: `streamlit run app.py`

## Deployment

This app is optimized for deployment on Streamlit Cloud. Follow the [Streamlit deployment guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app) for step-by-step instructions.

## Contributing

We welcome contributions to enhance the Fitness AI Chatbot! Here's how you can contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- Jeff Cavaliere and AthleanX for the inspirational content
- The Streamlit team for their amazing framework
- LangChain community for the powerful tools

---

Created with ❤️. Let's get fit together! 💪
