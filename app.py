import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from fpdf import FPDF
import base64
import io
import hashlib
import json
import os



# Download required NLTK data with better error handling
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download punkt tokenizer
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            try:
                nltk.download('punkt', quiet=True)
            except:
                st.error("Could not download punkt tokenizer. Please run: nltk.download('punkt_tab')")
    
    # Download stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            st.error("Could not download stopwords. Please run: nltk.download('stopwords')")
    
    # Download vader lexicon
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True)
        except:
            st.error("Could not download vader_lexicon. Please run: nltk.download('vader_lexicon')")

# Call the download function
download_nltk_data()

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False

if 'users_db' not in st.session_state:
    st.session_state.users_db = {}

class MentalHealthAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
        # Mental health keywords and patterns
        self.depression_keywords = [
            'depressed', 'sad', 'hopeless', 'worthless', 'empty', 'numb',
            'lonely', 'isolated', 'tired', 'exhausted', 'sleep', 'insomnia',
            'cry', 'crying', 'tears', 'suicide', 'kill myself', 'end it all',
            'no point', 'give up', 'can\'t go on', 'hate myself', 'failure',
            'disappointed', 'regret', 'guilty', 'shame', 'burden'
        ]
        
        self.anxiety_keywords = [
            'anxious', 'worried', 'nervous', 'panic', 'fear', 'scared',
            'overwhelmed', 'stressed', 'tension', 'restless', 'uneasy',
            'paranoid', 'worried', 'concern', 'afraid', 'terrified',
            'heart racing', 'can\'t breathe', 'sweating', 'trembling',
            'what if', 'catastrophe', 'disaster', 'danger'
        ]
        
        self.stress_keywords = [
            'stress', 'pressure', 'overwhelmed', 'burden', 'too much',
            'can\'t handle', 'breaking point', 'exhausted', 'burned out',
            'deadline', 'rushed', 'hurry', 'chaos', 'hectic', 'frantic',
            'struggling', 'difficult', 'hard', 'challenging', 'demanding'
        ]
        
        self.positive_keywords = [
            'happy', 'joy', 'excited', 'grateful', 'blessed', 'amazing',
            'wonderful', 'great', 'fantastic', 'love', 'peaceful', 'calm',
            'relaxed', 'confident', 'proud', 'accomplished', 'successful',
            'optimistic', 'hopeful', 'content', 'satisfied', 'fulfilled'
        ]
    
    def safe_tokenize(self, text):
        """Safe tokenization with fallback methods"""
        try:
            sentences = sent_tokenize(text.lower())
            words = word_tokenize(text.lower())
            return sentences, words
        except:
            # Fallback to simple tokenization
            import re
            sentences = re.split(r'[.!?]+', text.lower())
            sentences = [s.strip() for s in sentences if s.strip()]
            words = re.findall(r'\b\w+\b', text.lower())
            return sentences, words
    
    def analyze_text(self, text):
        """Analyze text for mental health indicators"""
        if not text.strip():
            return None
            
        # Basic sentiment analysis
        try:
            sentiment_scores = self.sia.polarity_scores(text)
        except:
            sentiment_scores = {'compound': 0.0, 'neg': 0.0, 'neu': 1.0, 'pos': 0.0}
        
        # Tokenize text with fallback
        sentences, words = self.safe_tokenize(text)
        
        # Count mental health indicators
        depression_count = sum(1 for word in self.depression_keywords if word in text.lower())
        anxiety_count = sum(1 for word in self.anxiety_keywords if word in text.lower())
        stress_count = sum(1 for word in self.stress_keywords if word in text.lower())
        positive_count = sum(1 for word in self.positive_keywords if word in text.lower())
        
        # Calculate severity scores (0-1)
        total_words = len(words)
        depression_score = min(depression_count / max(total_words * 0.1, 1), 1)
        anxiety_score = min(anxiety_count / max(total_words * 0.1, 1), 1)
        stress_score = min(stress_count / max(total_words * 0.1, 1), 1)
        positive_score = min(positive_count / max(total_words * 0.1, 1), 1)
        
        # Determine severity levels
        def get_severity(score):
            if score >= 0.3:
                return "High"
            elif score >= 0.15:
                return "Moderate"
            elif score >= 0.05:
                return "Mild"
            else:
                return "Low"
        
        # Find highlighted phrases
        highlighted_phrases = self.find_concerning_phrases(text)
        
        # Overall mental health assessment
        overall_score = (depression_score + anxiety_score + stress_score) / 3
        overall_assessment = self.get_overall_assessment(overall_score, positive_score)
        
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'text': text,
            'sentiment': sentiment_scores,
            'depression': {
                'score': depression_score,
                'severity': get_severity(depression_score),
                'count': depression_count
            },
            'anxiety': {
                'score': anxiety_score,
                'severity': get_severity(anxiety_score),
                'count': anxiety_count
            },
            'stress': {
                'score': stress_score,
                'severity': get_severity(stress_score),
                'count': stress_count
            },
            'positive': {
                'score': positive_score,
                'severity': get_severity(positive_score),
                'count': positive_count
            },
            'highlighted_phrases': highlighted_phrases,
            'overall_assessment': overall_assessment,
            'overall_score': overall_score
        }
    
    def find_concerning_phrases(self, text):
        """Find specific phrases that indicate mental health concerns"""
        phrases = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for depression indicators
            for keyword in self.depression_keywords:
                if keyword in sentence_lower:
                    phrases.append({
                        'phrase': sentence.strip(),
                        'type': 'depression',
                        'keyword': keyword
                    })
                    break
            
            # Check for anxiety indicators
            for keyword in self.anxiety_keywords:
                if keyword in sentence_lower:
                    phrases.append({
                        'phrase': sentence.strip(),
                        'type': 'anxiety',
                        'keyword': keyword
                    })
                    break
            
            # Check for stress indicators
            for keyword in self.stress_keywords:
                if keyword in sentence_lower:
                    phrases.append({
                        'phrase': sentence.strip(),
                        'type': 'stress',
                        'keyword': keyword
                    })
                    break
        
        return phrases
    
    def get_overall_assessment(self, negative_score, positive_score):
        """Provide overall mental health assessment"""
        if negative_score >= 0.3:
            if positive_score >= 0.2:
                return "Mixed emotions with significant concerns - consider professional support"
            else:
                return "High concern detected - strongly recommend professional help"
        elif negative_score >= 0.15:
            if positive_score >= 0.3:
                return "Moderate concerns with positive elements - monitor closely"
            else:
                return "Moderate concern - consider speaking with someone"
        elif negative_score >= 0.05:
            return "Mild concerns detected - practice self-care"
        else:
            if positive_score >= 0.2:
                return "Positive mental state - keep up the good work!"
            else:
                return "Neutral state - maintain healthy habits"

def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, email):
    """Create a new user"""
    if username in st.session_state.users_db:
        return False, "Username already exists"
    
    st.session_state.users_db[username] = {
        'password': hash_password(password),
        'email': email,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return True, "User created successfully"

def verify_user(username, password):
    """Verify user credentials"""
    if username not in st.session_state.users_db:
        return False, "User not found"
    
    if st.session_state.users_db[username]['password'] == hash_password(password):
        return True, "Login successful"
    else:
        return False, "Invalid password"

def show_auth_page():
    """Show enhanced authentication page"""
    st.markdown("""
    <div class="auth-container">
        <div class="auth-header">
            <h1>🧠 Mental Health Analyzer</h1>
            <p>Advanced AI-powered mental health analysis platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for login and signup
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### Welcome Back!")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                login_button = st.form_submit_button("Login", type="primary", use_container_width=True)
            with col2:
                forgot_password = st.form_submit_button("Forgot Password?", use_container_width=True)
            
            if login_button:
                if username and password:
                    success, message = verify_user(username, password)
                    if success:
                        st.session_state.user_logged_in = True
                        st.session_state.current_user = username
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all fields")
            
            if forgot_password:
                st.info("Password reset functionality would be implemented here")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            st.markdown("### Create Account")
            new_username = st.text_input("Username", placeholder="Choose a username")
            new_email = st.text_input("Email", placeholder="Enter your email")
            new_password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            signup_button = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            
            if signup_button:
                if new_username and new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif not terms:
                        st.error("Please accept the terms and conditions")
                    else:
                        success, message = create_user(new_username, new_password, new_email)
                        if success:
                            st.success(message)
                            st.info("You can now login with your credentials")
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_severity_chart(analysis):
    """Create a severity chart using Plotly"""
    categories = ['Depression', 'Anxiety', 'Stress', 'Positive']
    scores = [
        analysis['depression']['score'],
        analysis['anxiety']['score'],
        analysis['stress']['score'],
        analysis['positive']['score']
    ]
    colors = ['#FF6B6B', '#FFA500', '#FFD700', '#4CAF50']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=scores,
            marker_color=colors,
            text=[f"{score:.2f}" for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Mental Health Analysis Scores",
        xaxis_title="Categories",
        yaxis_title="Score (0-1)",
        yaxis=dict(range=[0, 1]),
        height=400,
        font=dict(family="Inter, sans-serif", size=12)
    )
    
    return fig

def create_sentiment_pie_chart(sentiment_scores):
    """Create sentiment pie chart"""
    # Convert VADER scores to percentages
    total = sentiment_scores['pos'] + sentiment_scores['neu'] + sentiment_scores['neg']
    if total == 0:
        return None
    
    labels = ['Positive', 'Neutral', 'Negative']
    values = [sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg']]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.3
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        height=400,
        font=dict(family="Inter, sans-serif", size=12)
    )
    
    return fig

def get_severity_emoji(severity):
    """Get emoji based on severity level"""
    emoji_map = {
        'Low': '🟢',
        'Mild': '🟡',
        'Moderate': '🟠',
        'High': '🔴'
    }
    return emoji_map.get(severity, '⚪')

def export_results_pdf(analysis_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, "Mental Health Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("helvetica", size=10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    for i, analysis in enumerate(analysis_history[-10:], 1):
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(0, 10, f"Analysis {i} - {analysis['timestamp']}", ln=True)
        pdf.set_font("helvetica", size=10)
        pdf.cell(0, 10, f"Depression: {analysis['depression']['severity']}", ln=True)
        pdf.cell(0, 10, f"Anxiety: {analysis['anxiety']['severity']}", ln=True)
        pdf.cell(0, 10, f"Stress: {analysis['stress']['severity']}", ln=True)
        pdf.multi_cell(0, 10, f"Overall: {analysis['overall_assessment']}")
        pdf.ln(5)
    result = pdf.output(dest='S')
    if isinstance(result, str):
        result = result.encode('latin1')
    return result

def main():
    st.set_page_config(
        page_title="Mental Health Chat Analyzer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS for professional look
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: rgba(18, 18, 18, 1);
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
    }
    
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .auth-header {
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    
    .auth-header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .auth-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .auth-form {
        background: rgb(243 166 0 / 90%);
        backdrop-filter: blur(10px);
        padding: 0.5rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(50, 90, 200, 0.9) 0%, rgba(80, 150, 250, 0.9) 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .user-info {
        background: rgba(0, 150, 136, 0.8);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3);
    }
    
    .user-info h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .analysis-card {
        background: rgba(30, 30, 30, 0.95);
        color: #fff;
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .analysis-card h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        color: #333;
    }
    
    .analysis-card p {
        font-size: 1rem;
        line-height: 1.6;
        color: #555;
    }
    
    .metric-card {
        background: rgba(0, 180, 170, 0.2);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .highlighted-phrase {
        background: rgba(255, 243, 205, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ffc107;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .depression-phrase { 
        border-left-color: #dc3545; 
        background: rgba(248, 215, 218, 0.9); 
    }
    
    .anxiety-phrase { 
        border-left-color: #fd7e14; 
        background: rgba(255, 234, 167, 0.9); 
    }
    
    .stress-phrase { 
        border-left-color: #ffc107; 
        background: rgba(255, 243, 205, 0.9); 
    }
    
    .sidebar-content {
        background: rgba(30, 30, 30, 0.95);
        color: #fff;
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: rgba(255, 140, 0, 0.9);
        color: #fff;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: rgba(255, 140, 0, 1);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stTextInput > div > div > input {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        border-radius: 10px;
        border: 1px solid rgb(38, 39, 48);
        padding: 0.75rem;
    }
    
    .stTextArea > div > div > textarea {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 0.75rem;
    }
    
    .stSelectbox > div > div > select {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        background: rgba(255, 255, 255, 0.7);
        color: #333;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
    }
    
    .stMetric {
        background: rgba(0, 180, 170, 0.2);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .stMetric > div {
        font-family: 'Inter', sans-serif;
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
    }
    
    .stExpander > div {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    
    .footer {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    
    p, div, span, label {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    
    .stSidebar {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
    
    .stSidebar > div {
        background: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if user is logged in
    if not st.session_state.user_logged_in:
        show_auth_page()
        return
    
    # Initialize analyzer
    analyzer = MentalHealthAnalyzer()
    
    # Sidebar with user info and logout
    with st.sidebar:
        st.markdown(f"""
        <div class="user-info">
            <h3>👋 Welcome, {st.session_state.current_user}!</h3>
            <p>Ready to analyze your mental health</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            st.session_state.user_logged_in = False
            st.session_state.current_user = None
            st.session_state.analysis_history = []
            st.rerun()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Mental Health Chat Analyzer</h1>
        <p>Advanced AI-powered mental health analysis using state-of-the-art NLP techniques</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Text Analysis")
        
        # Text input
        user_text = st.text_area(
            "Enter your text here for analysis:",
            height=150,
            placeholder="Type your message here... The analyzer will check for depression, anxiety, and stress indicators.",
            key="text_input"
        )
        
        # Analysis button
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            if user_text.strip():
                with st.spinner("Analyzing text..."):
                    analysis = analyzer.analyze_text(user_text)
                    
                    if analysis:
                        # Add to history
                        st.session_state.analysis_history.append(analysis)
                        
                        # Display results
                        st.success("Analysis completed successfully!")
                        
                        # Overall assessment
                        st.markdown(f"""
                        <div class="analysis-card">
                            <h3>🎯 Overall Assessment</h3>
                            <p><strong>{analysis['overall_assessment']}</strong></p>
                            <p>Overall Score: {analysis['overall_score']:.2f}/1.0</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Severity scores
                        st.markdown("### 📊 Mental Health Metrics")
                        col_dep, col_anx, col_str, col_pos = st.columns(4)
                        
                        with col_dep:
                            emoji = get_severity_emoji(analysis['depression']['severity'])
                            st.metric(
                                f"{emoji} Depression",
                                f"{analysis['depression']['severity']}",
                                f"Score: {analysis['depression']['score']:.2f}"
                            )
                        
                        with col_anx:
                            emoji = get_severity_emoji(analysis['anxiety']['severity'])
                            st.metric(
                                f"{emoji} Anxiety",
                                f"{analysis['anxiety']['severity']}",
                                f"Score: {analysis['anxiety']['score']:.2f}"
                            )
                        
                        with col_str:
                            emoji = get_severity_emoji(analysis['stress']['severity'])
                            st.metric(
                                f"{emoji} Stress",
                                f"{analysis['stress']['severity']}",
                                f"Score: {analysis['stress']['score']:.2f}"
                            )
                        
                        with col_pos:
                            emoji = get_severity_emoji(analysis['positive']['severity'])
                            st.metric(
                                f"{emoji} Positive",
                                f"{analysis['positive']['severity']}",
                                f"Score: {analysis['positive']['score']:.2f}"
                            )
                        
                        # Charts
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown("### 📈 Visual Analysis")
                        
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            severity_chart = create_severity_chart(analysis)
                            st.plotly_chart(severity_chart, use_container_width=True)
                        
                        with chart_col2:
                            sentiment_chart = create_sentiment_pie_chart(analysis['sentiment'])
                            if sentiment_chart:
                                st.plotly_chart(sentiment_chart, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Highlighted phrases
                        if analysis['highlighted_phrases']:
                            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                            st.markdown("### 🔍 Concerning Phrases")
                            for phrase in analysis['highlighted_phrases']:
                                phrase_class = f"{phrase['type']}-phrase"
                                st.markdown(f"""
                                <div class="highlighted-phrase {phrase_class}">
                                    <strong>{phrase['type'].title()}:</strong> "{phrase['phrase']}"
                                    <br><small>Keyword: {phrase['keyword']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to analyze.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### 📈 Analysis History")
        
        if st.session_state.analysis_history:
            # Export options
            st.markdown("#### 💾 Export Options")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                if st.button("📄 PDF Report", use_container_width=True):
                    pdf_data = export_results_pdf(st.session_state.analysis_history)
                    if pdf_data is not None:
                        # Ensure pdf_data is bytes, not bytearray
                        if isinstance(pdf_data, bytearray):
                            pdf_data = bytes(pdf_data)
                        st.download_button(
                            label="Download PDF",
                            data=pdf_data,
                            file_name=f"mental_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate PDF report.")
            
            with col_export2:
                if st.button("📝 Text Report", use_container_width=True):
                    text_data = ""
                    for i, analysis in enumerate(st.session_state.analysis_history, 1):
                        text_data += f"Analysis {i} - {analysis['timestamp']}\n"
                        text_data += f"Depression: {analysis['depression']['severity']}\n"
                        text_data += f"Anxiety: {analysis['anxiety']['severity']}\n"
                        text_data += f"Stress: {analysis['stress']['severity']}\n"
                        text_data += f"Overall: {analysis['overall_assessment']}\n"
                        text_data += "-" * 50 + "\n\n"
                    
                    st.download_button(
                        label="Download Text",
                        data=text_data,
                        file_name=f"mental_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            # Clear history
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                st.session_state.analysis_history = []
                st.success("History cleared!")
                st.rerun()
            
            # Display history
            st.markdown("#### 📋 Recent Analyses")
            
            # Show last 10 analyses
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-10:]), 1):
                with st.expander(f"Analysis {len(st.session_state.analysis_history) - i + 1} - {analysis['timestamp']}"):
                    st.write(f"**Text:** {analysis['text'][:100]}...")
                    st.write(f"**Depression:** {get_severity_emoji(analysis['depression']['severity'])} {analysis['depression']['severity']}")
                    st.write(f"**Anxiety:** {get_severity_emoji(analysis['anxiety']['severity'])} {analysis['anxiety']['severity']}")
                    st.write(f"**Stress:** {get_severity_emoji(analysis['stress']['severity'])} {analysis['stress']['severity']}")
                    st.write(f"**Assessment:** {analysis['overall_assessment']}")
        else:
            st.info("No analyses yet. Start by entering some text above!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>⚠️ Important Disclaimer:</strong> This tool is designed for informational and educational purposes only. It should not be used as a substitute for professional mental health consultation, diagnosis, or treatment.</p>
        <p>If you're experiencing mental health concerns, please consult with a qualified healthcare provider or mental health professional.</p>
        <p><strong>Crisis Resources:</strong> If you're in crisis, please contact your local emergency services or a crisis helpline immediately.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()