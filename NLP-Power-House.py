import os
import streamlit as st
from streamlit_option_menu import option_menu
import speech_recognition as sr
from PIL import Image
from fpdf import FPDF
import pytesseract
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import fitz  # PyMuPDF for PDF processing
from gtts import gTTS
import tempfile
import pandas as pd
import bcrypt
import base64
from pathlib import Path
from datetime import datetime
import re
import google.generativeai as genai
import cv2
import numpy as np
import time
import easyocr


user_data_file = "login_data.csv"
feedback_file = "feedback.csv"
genai.configure(api_key='AIzaSyDGMkXv8Qqh9Bwf2Xs_M6j1UNTSFJC9wBw')  # Replace with your actual API key
# Ensure necessary files exist
def ensure_user_data():
    if not os.path.exists(user_data_file):
        df = pd.DataFrame(columns=['Username', 'Password'])
        df.to_csv(user_data_file, index=False)
def ensure_feedback_file():
    if not os.path.exists(feedback_file):
        pd.DataFrame(columns=["Name", "Age", "Gender", "Rating", "Feedback"]).to_csv(feedback_file, index=False)

ensure_user_data()
ensure_feedback_file()
# Load user data
def load_user_data():
    return pd.read_csv(user_data_file)
# Save new user data
def save_user_data(username, password):
    df = load_user_data()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = pd.DataFrame([[username, hashed_password]], columns=['Username', 'Password'])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(user_data_file, index=False)

# Check if the username exists
def username_exists(username):
    df = load_user_data()
    return not df[df['Username'] == username].empty

# Validate login
def validate_login(username, password):
    df = load_user_data()
    user_record = df[df['Username'] == username]
    if not user_record.empty:
        stored_hashed_password = user_record['Password'].values[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8'))
    return False

# Change password functionality
def change_password(username, new_password):
    df = load_user_data()
    if username_exists(username):
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        df.loc[df['Username'] == username, 'Password'] = hashed_password
        df.to_csv(user_data_file, index=False)
        return True
    return False
# Save feedback data to CSV
def save_feedback(name, age, gender, rating, feedback):
    rating_map = {
        1: "1 Star - Poor",
        2: "2 Stars - Fair",
        3: "3 Stars - Average",
        4: "4 Stars - Good",
        5: "5 Stars - Excellent"
    }
    formatted_rating = rating_map[rating]
    
    
    feedback_data = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Gender": [gender],
        "Rating": [formatted_rating],
        "Feedback": [feedback]
    })

    if os.path.exists(feedback_file):
        existing_data = pd.read_csv(feedback_file)
        feedback_data = pd.concat([existing_data, feedback_data], ignore_index=True)

    feedback_data.to_csv(feedback_file, index=False)
# Streamlit app title
st.title("Real Time AI-Driven NLP Powerhouse")

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Display logout message if logged out
if 'logout_message' in st.session_state:
    st.success(st.session_state.logout_message)
    del st.session_state.logout_message  # Clear the message after displaying it
# Display login interface only if not logged in
if not st.session_state.logged_in:
    # Set the background image for the login interface
    def set_login_background(image_file):
        login_bg_img = f'''
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{image_file});
            background-size: cover;
            background-position: center;
        }}
        </style>
        '''
        st.markdown(login_bg_img, unsafe_allow_html=True)

    # Load the background image for the login interface
    with open("digital-art.jpg", "rb") as image_file:  # Change this path to your image
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Set the background for the login interface
    set_login_background(encoded_image)

    with st.expander("Authentication", expanded=True):
        menu = option_menu(
                            menu_title=None,
                            options=['Login', 'Register', 'Forgot Password'],
                            icons=['box-arrow-right', 'person-plus', 'key'],
                            orientation='horizontal'
  
                          )

        if menu == 'Register':
            st.subheader('Register')
            username = st.text_input("Choose a Username", key="register_username")  # Unique key for username
            password = st.text_input("Choose a Password", type="password", key="register_password")  # Unique key for password
            confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")  # Unique key for confirm password

            def is_valid_password(password):
                # Check if password meets the requirements
                if len(password) < 8 or len(password) > 12:
                    st.error("Password must be between 8 to 12 characters length.")
                    return False
                if not any(char.isupper() for char in password):
                    st.error("Password must include at least one uppercase letter (A-Z).")
                    return False
                if not any(char.islower() for char in password):
                    st.error("Password must include at least one lowercase letter (a-z).")
                    return False
                if not any(char.isdigit() for char in password):
                    st.error("Password must include at least one digit (0-9).")
                    return False
                if not re.search(r'[!@#$%^&*]', password):
                    st.error("Password must include at least one special character (!@#$%^&*).")
                    return False
                return True

            if st.button("Register"):
                if password != confirm_password:
                    st.error("Passwords do not match!")
                elif username_exists(username):
                    st.error("Username already exists. Please choose a different one.")
                elif not is_valid_password(password):
                    # Password requirements not met, error messages will be displayed in is_valid_password
                    pass
                else:
                    save_user_data(username, password)
                    st.success("Registration successful! You can now log in.")

        elif menu == 'Forgot Password':
            st.subheader('Reset Password')
            username = st.text_input("Enter your Username")
            new_password = st.text_input("Enter your New Password", type='password')
            confirm_password = st.text_input("Confirm New Password", type='password')

            if st.button("Reset Password"):
                if username_exists(username):
                    if new_password == confirm_password:
                        if change_password(username, new_password):
                            st.success("Your password has been reset successfully.")
                        else:
                            st.error("Failed to reset password. Please try again.")
                    else:
                        st.error("Passwords do not match! Please try again.")
                else:
                    st.error("Username not found!")

        elif menu == 'Login':
            st.subheader('Login')
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if validate_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.show_success_message = True  # Flag to show success message
                    st.rerun()
                else:
                    st.error("Invalid username or password")
# Main project interface
if st.session_state.logged_in:
    # Set the background image for the Streamlit interface
    def set_background(image_file):
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{image_file});
            background-size: cover;
            background-position: center;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

    # Load the background image for the interface
    with open("dark-abstract.jpg", "rb") as image_file:  # Change this path to your image
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Set the background
    set_background(encoded_image)
    
    # Display the "Login successful!" message if the user has just logged in
    if 'show_success_message' in st.session_state and st.session_state.show_success_message:
        st.success("Login successful!✅")
        time.sleep(3)  # Keep the message for 3 seconds
        st.session_state.show_success_message = False
        st.rerun() # Reset the flag
    # Main title for the project features

    # Function to create a PDF with the chatbot response
    def create_response_pdf(response_text):
        pdf = FPDF()
        pdf.add_page()

        # Set fonts and colors
        pdf.set_font('Arial', 'B', 16)
        pdf.set_fill_color(255, 255, 255)  # Background color (white)
        
        # Title
        title = 'Response for Your Query'
        pdf.cell(0, 10, title, ln=True, align='C')

        # Add the response text
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        
        # Replace bold markers with FPDF's bold
        formatted_text = response_text.replace('*', '')
        pdf.multi_cell(0, 10, formatted_text)
        
        return pdf.output(dest='S').encode('latin1')  # Return PDF as a binary string

    with st.sidebar:
        selected=option_menu(
            menu_title='Main Menu',
            options=['Ask AI Chatbot', 'Speech to Text', 'Text to Speech', 'Text from Image', 'Sentimental Analysis', 'Feedback', 'About Us', 'Logout'],
            icons=['robot', 'card-text', 'mic-fill', 'image-fill', 'emoji-smile-fill', 'star-fill', 'person-circle', 'box-arrow-left'],
            menu_icon='cast',
            default_index=0
                            )
    if selected == 'Ask AI Chatbot':
        st.title("Ask AI Chatbot")
        st.subheader("Ask your queries to the Chatbot")
        user_query = st.text_input("What do you want to know about?",
                                 placeholder="Message with AI Chatbot...")
        
        if st.button("Submit"):
            if user_query:
                # Generate a response from the Gemini API
                response = genai.GenerativeModel('gemini-1.5-flash').generate_content(user_query)
                st.write("Chatbot:", response.text)
                # Provide an option to download the response as PDF
                pdf_data = create_response_pdf(response.text)
                st.download_button(
                    label="Download Response as PDF",
                    data=pdf_data,
                    file_name="chatbot_response.pdf",
                    mime="application/pdf"
                                  )
            else:
                st.warning("⚠️ Please ask your question.")  # Show a warning if input is empty
    if selected == 'Speech to Text':
        # Function to listen and recognize speech
        st.title("Sppech to Text")
        st.subheader("Convert Speech into Text")
        st.error("This feature is not yet available on Streamlit Cloud. It will be introduced soon, allowing for more direct browser-based audio recording and microphone access.")
    if selected == 'Text from Image':
        st.title("Text from Image")
        st.subheader("OCR Text Detection from Images")

        # Function to extract text using EasyOCR
        def extract_text(image):
            reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader
            result = reader.readtext(np.array(image))  # Convert image to numpy array
            # Extract only the text part from the result
            detected_text = [text[1] for text in result]
            return detected_text

        # Function to create a PDF from text with title and subheader
        def create_pdf_with_title(text):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Title: OCR Text Detection from Images
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="OCR Text Detection from Images", ln=True, align='C')

            # Subheader: Detected Text
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Detected Text:", ln=True, align='L')

            # Add text to PDF, wrapping text properly
            pdf.set_font("Arial", size=12)
            for line in text.split('\n'):
                pdf.multi_cell(0, 10, line)
            
            return pdf.output(dest="S").encode('latin1')  # Return PDF as a binary string

        # Streamlit file uploader
        uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

        if uploaded_image is not None:
            # Display the uploaded image
            img = Image.open(uploaded_image)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            # Extract and display detected text
            detected_text = extract_text(img)

            # Check if any text was detected
            if detected_text:
                # Join detected text into a single string with new lines between each text block
                text_output = "\n".join(detected_text)
                
                # Display the detected text in a text area
                st.text_area("Detected Text:", value=text_output, height=300)
                
                # Generate PDF with title and subheader, and add download button to download text as a .pdf file
                pdf_data = create_pdf_with_title(text_output)
                st.download_button(
                    label="Download Detected Text as .pdf",
                    data=pdf_data,  # The binary data to download (PDF)
                    file_name="detected_text.pdf",  # Filename for the download
                    mime="application/pdf"  # MIME type for PDF
                )
            else:
                st.write("No text detected.")

    if selected == 'Text to Speech':
        # Streamlit interface
        st.title("Text to Speech")
        st.subheader("Convert Text into Speech")

        # Function to extract text from PDF
        def extract_text_from_pdf(pdf_file):
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text

        # Function to convert text to speech and save as audio file
        def convert_text_to_speech(text, lang='en'):
            tts = gTTS(text=text, lang=lang)
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name

        # Option 1: User can input text manually
        user_input = st.text_area("Enter Text Manually", height=150)

        # Button for manual text input conversion
        if st.button("Convert Text to Speech for Input Text"):
            if user_input:
                audio_file = convert_text_to_speech(user_input)
                st.success("Speech conversion for text input completed!")
                # Play the converted audio
                st.audio(audio_file, format='audio/mp3')
            else:
                st.warning("Please enter some text!")

        # Separator for better UI
        st.markdown("---")

        # Option 2: User can upload a text or PDF file
        st.subheader("Upload a Text or PDF File")
        uploaded_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])

        # Process uploaded file if available
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                # Read the text from the uploaded .txt file
                text_from_file = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                # Extract text from the uploaded PDF
                text_from_file = extract_text_from_pdf(uploaded_file)
            
            # Display the text from the file
            st.text_area("Text from Uploaded File", value=text_from_file, height=150)

            # Button for file upload conversion
            if st.button("Convert Text to Speech for Uploaded File"):
                if text_from_file:
                    audio_file = convert_text_to_speech(text_from_file)
                    st.success("Speech conversion for uploaded file completed!")
                    # Play the converted audio
                    st.audio(audio_file, format='audio/mp3')
                else:
                    st.warning("The uploaded file is empty!")
    if selected == 'Sentimental Analysis':
        # Download the VADER lexicon if it's not already installed
        nltk.download('vader_lexicon')

        # Initialize the Sentiment Intensity Analyzer
        sia = SentimentIntensityAnalyzer()

        # Function to extract text from PDF
        def extract_text_from_pdf(pdf_file):
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text

        # Streamlit app interface
        st.title("Sentiment Analysis")

        # Option 1: Manual Text Input
        st.subheader("Enter Text for Sentiment Analysis")
        user_input = st.text_area("Enter Text Manually", height=150)

        # Button to analyze sentiment for manually entered text
        if st.button("Analyze Sentiment for Text Input"):
            if user_input:
                # Perform sentiment analysis
                sentiment_scores = sia.polarity_scores(user_input)
                
                # Display the sentiment scores
                st.subheader("Sentiment Scores for Text Input")
                st.write(f"Positive: {sentiment_scores['pos'] * 100:.2f}%")
                st.write(f"Neutral: {sentiment_scores['neu'] * 100:.2f}%")
                st.write(f"Negative: {sentiment_scores['neg'] * 100:.2f}%")
                st.write(f"Overall Sentiment (Compound Score): {sentiment_scores['compound']}")

                # Display overall sentiment based on compound score
                if sentiment_scores['compound'] >= 0.05:
                    st.success("Overall Sentiment: Positive 😊")
                elif sentiment_scores['compound'] <= -0.05:
                    st.error("Overall Sentiment: Negative 😞")
                else:
                    st.info("Overall Sentiment: Neutral 😐")
            else:
                st.warning("Please enter some text to analyze!")

        # Option 2: Text File or PDF Upload
        st.subheader("Upload a Text or PDF File for Sentiment Analysis")
        uploaded_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])

        # Process the uploaded file
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                # Read the text from the uploaded .txt file
                file_text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                # Extract text from the uploaded PDF
                file_text = extract_text_from_pdf(uploaded_file)
            
            # Display the text from the file
            st.text_area("Text from Uploaded File", value=file_text, height=150)

            # Button to analyze sentiment for uploaded file
            if st.button("Analyze Sentiment for Uploaded File"):
                # Perform sentiment analysis
                sentiment_scores = sia.polarity_scores(file_text)
                
                # Display the sentiment scores
                st.subheader("Sentiment Scores for Uploaded File")
                st.write(f"Positive: {sentiment_scores['pos'] * 100:.2f}%")
                st.write(f"Neutral: {sentiment_scores['neu'] * 100:.2f}%")
                st.write(f"Negative: {sentiment_scores['neg'] * 100:.2f}%")
                st.write(f"Overall Sentiment (Compound Score): {sentiment_scores['compound']}")

                # Display overall sentiment based on compound score
                if sentiment_scores['compound'] >= 0.05:
                    st.success("Overall Sentiment: Positive 😊")
                elif sentiment_scores['compound'] <= -0.05:
                    st.error("Overall Sentiment: Negative 😞")
                else:
                    st.info("Overall Sentiment: Neutral 😐")
    if selected == 'Feedback':
        st.title("Feedback")
        st.subheader("Give Us Your Feedback")

        user_name = st.text_input("Enter Your Name:")
        user_age = st.number_input("Enter Your Age:", min_value=1, max_value=120, step=1, format="%d")

        gender_options = ["Male", "Female"]
        selected_gender = st.selectbox("Select Your Gender:", gender_options)

        #feedback_rating = st.radio("Rate your experience (1-5 stars):", range(1, 6))
        # Rating selection
        feedback_rating = st.radio(
            "Rate Your Experience (1-5 stars):",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "1 Star - Poor",
                2: "2 Stars - Fair",
                3: "3 Stars - Average",
                4: "4 Stars - Good",
                5: "5 Stars - Excellent"
            }[x]
        )

        feedback_text = st.text_area("Share Your Suggestions: (if any)")

        if st.button("Submit Feedback"):
            if user_name and user_age and selected_gender:
                
                save_feedback(user_name, user_age, selected_gender, feedback_rating, feedback_text)
                st.success("Thank you for your feedback!")
            else:
                st.error("Please fill in all fields before submitting.")
    if selected == 'About Us':
        # Add Font Awesome CDN link to your Streamlit app
        st.markdown(
            """
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            """,
            unsafe_allow_html=True
        )
        st.markdown("## <i class='fas fa-info-circle'></i> About Us", unsafe_allow_html=True)
        st.write("We are a dedicated team committed to providing the best service.")

        # Mission Section
        st.markdown("<h3><i class='fas fa-bullseye'></i> Our Mission:</h3>", unsafe_allow_html=True)
        st.markdown("""
        1. <i class='fas fa-robot'></i> **Develop an AI-powered Chatbot** to assist users with their queries. The chatbot will leverage advanced natural language processing techniques to provide accurate and timely information about medicinal plants, including their uses, benefits, and identification. It will also offer personalized responses based on user interactions to enhance the overall user experience and provide valuable insights into medicinal plant knowledge.
        2. <i class='fas fa-microphone'></i> **Implement Speech to Text functionality** to convert spoken words into written text for ease of documentation and interaction.
        3. <i class='fas fa-volume-up'></i> **Integrate Text to Speech capabilities** to convert written text into spoken words, enhancing accessibility for visually impaired users.
        4. <i class='fas fa-image'></i> **Enable Text from Image extraction** using OCR technology to identify and extract text from images of plant labels and documents.
        5. <i class='fas fa-smile'></i> **Perform Sentimental Analysis** on user inputs and feedback to gauge sentiments and improve user experience.
        6. <i class='fas fa-comment-dots'></i> **Collect and analyze Feedback** from users to continuously refine and enhance the application’s features and functionality.
        7. <i class='fas fa-user'></i> **Provide an 'About Us' section** to inform users about the team and vision behind the project.
        8. <i class='fas fa-sign-out-alt'></i> **Offer a 'Logout' feature** to ensure secure and personalized user sessions.
        """, unsafe_allow_html=True)

        # Team Section
        st.markdown("<h3><i class='fas fa-users'></i> The Team:</h3>", unsafe_allow_html=True)
        st.markdown("""
        - <i class='fas fa-user'></i> **Dr. [Name], Lead Researcher** - Specializes in machine learning and image classification algorithms.
        - <i class='fas fa-user'></i> **[Name], Botanist** - Provides domain expertise on medicinal plants and their identification.
        - <i class='fas fa-user'></i> **[Name], Web Developer** - Responsible for designing and implementing the project's web application.
        - <i class='fas fa-user'></i> **[Name], Data Scientist** - Analyzes plant data and develops insights to improve the identification system.
        """, unsafe_allow_html=True)

        # Project Mentor
        st.markdown("<h3><i class='fas fa-chalkboard-teacher'></i> Faculty Instructor:</h3>", unsafe_allow_html=True)
        st.write("**ABC**")

        ## Project Evaluator
        #st.markdown("<h3><i class='fas fa-pen'></i> Project Evaluator:</h3>", unsafe_allow_html=True)
        #st.write("**XYZ**")

        # Adding some separation
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3><i class='fas fa-star'></i> We are dedicated to leveraging cutting-edge NLP technologies to enhance communication and interaction. This project is part of our commitment to advancing digital solutions, offering services such as chatbots, voice and text conversion, OCR, and sentiment analysis to enrich user experience and accessibility !!</h3>", unsafe_allow_html=True)

    # Logout button
    if selected == 'Logout':
        st.session_state.logged_in = False
        st.session_state.logout_message = "You have successfully logged out! Please log in again to continue exploring our NLP services, including chatbots, voice and text conversion, OCR, and sentiment analysis."
        st.rerun()  # Refresh the page
