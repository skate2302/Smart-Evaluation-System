import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os
import google.generativeai as genai
import gensim.downloader as api
import re
import json

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math

# Load GPT-2 Model and Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def calculate_perplexity(text):
    """
    Calculates the perplexity score of a given text using GPT-2.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity

def perplexity_to_ai_probability(perplexity, max_ppl=200):
    """
    Converts perplexity score to AI probability using a sigmoid function.
    """
    return 100 / (1 + math.exp((perplexity - 40) / 10))  # Adjust 40 and 10 for sensitivity



##############################################################################################################################################################################

# ✅ Load SBERT model
sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# ✅ Load Word2Vec model from gensim
word2vec_model = api.load("word2vec-google-news-300") 

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# try:
#     df = pd.read_csv("../data/Final_Dataset.csv", encoding='windows-1252') #or whatever encoding your file has.
# except UnicodeDecodeError:
#     df = pd.read_csv("../data/Final_Dataset.csv", encoding='latin1') #latin1 is very forgiving.

df = pd.read_csv("../data/Final_Dataset.csv") 

# ✅ Function to preprocess text
def preprocess_text(text):
    text = text.lower().strip()
    return text

# ✅ Function to get Word2Vec embedding
def get_word2vec_embedding(text):
    words = text.split()
    word_vectors = [word2vec_model[word] for word in words if word in word2vec_model]
    if word_vectors:
        return np.mean(word_vectors, axis=0).tolist()
    else:
        return np.zeros(word2vec_model.vector_size).tolist()
    
# ✅ Function to get word alignment score
def get_alignment_score(ref_text, stud_text):
    ref_words = Counter(ref_text.split())
    stud_words = Counter(stud_text.split())
    common_words = sum((ref_words & stud_words).values())  # Intersection count
    return common_words / max(len(ref_words), 1)  # Normalize

def get_gemini_feedback(question, student_answer, ref_answer):
    """Get feedback from Gemini for every question, regardless of grade"""
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    chat_session = model.start_chat(history=[])

    prompt = f"""
    You are a strict AI grading system. Provide feedback for the student's answer compared to the reference answer.

    **Feedback Criteria:**
    - Be specific about what's correct or incorrect
    - Keep feedback under 25 words
    - Be constructive and helpful

    - **Question:** {question}  
    - **Student Answer:** {student_answer}  
    - **Reference Answer:** {ref_answer}

    **Response Format (JSON only, no explanation)**:
    ```json
    {{
        "feedback": "your specific feedback here (max 25 words)"
    }}
    ```
    """ 

    response = chat_session.send_message(prompt)
    text_response = response.text.strip()
    print("Gemini feedback response:", text_response)

    try:
        # Try to extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', text_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            feedback_data = json.loads(json_str)
            return feedback_data.get("feedback", "No feedback available")
        
        # If JSON extraction fails, try to find "feedback" in the text
        feedback_match = re.search(r'"feedback":\s*"([^"]*)"', text_response)
        if feedback_match:
            return feedback_match.group(1)
        
        return "Feedback extraction failed"
    except Exception as e:
        print(f"Error extracting feedback: {str(e)}")
        return f"Error extracting feedback: {str(e)}"

def get_gemini_grade(question, student_answer, ref_answer):
    """Get both grade and feedback from Gemini for borderline cases"""
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    chat_session = model.start_chat(history=[])

    prompt = f"""
    You are a very strict AI grading system. Evaluate the student's answer based on the reference answer and the question provided. (If only keywords are provided without semantic formation, grade as Incorrect with feedback)

    **Grading Criteria:**
    - Completely Correct (2) → The answer fully aligns with the reference, slight mistakes allowed.
    - Partially Incorrect (1) → Some information is correct, but the answer is incomplete.
    - Incorrect (0) → The answer is mostly incorrect.

    - **Question:** {question}  
    - **Student Answer:** {student_answer}  
    - **Reference Answer:** {ref_answer}

    **Response Format (JSON only, no explanation)**:
    ```json
    {{
        "grade": 2,  # 2 for Completely Correct, 1 for Partially Incorrect, 0 for Incorrect
        "feedback": "feedback (max 25 words)"
    }}
    ```
    """ 
    # print("Prompt for Gemini Grade:", prompt)
    response = chat_session.send_message(prompt)
    text_response = response.text.strip()
    print("Gemini grade response text:", text_response)

    try:
        # Try to extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', text_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            grade_data = json.loads(json_str)
            return grade_data.get("grade"), grade_data.get("feedback", "No feedback available")
        
        # If JSON extraction fails, try to extract grade and feedback manually
        grade_match = re.search(r'"grade":\s*(\d)', text_response)
        feedback_match = re.search(r'"feedback":\s*"([^"]*)"', text_response)
        
        if grade_match:
            grade = int(grade_match.group(1))
            feedback = feedback_match.group(1) if feedback_match else "No feedback available"
            return grade, feedback
        
        # Fallback to old extraction method
        if "2" in text_response:
            grade = 2
        elif "1" in text_response:
            grade = 1
        elif "0" in text_response:
            grade = 0
        else:
            return None, "Failed to extract grade"

        # Extract feedback (fallback method)
        if feedback_match:
            feedback = feedback_match.group(1)
        else:
            feedback = "No detailed feedback available"

        return grade, feedback
    except Exception as e:
        print(f"Error extracting grade and feedback: {str(e)}")
        return None, f"Error extracting feedback: {str(e)}"

    
def evaluate_student_answer(question, student_answer):


    # Compute AI Probability
    perplexity = calculate_perplexity(student_answer)
    ai_probability = perplexity_to_ai_probability(perplexity)

    # Find reference answer
    matched_row = df[df["question"] == question]
    
    if matched_row.empty:
        return {"error": "Question not found in dataset!"}
    
    ref_answer = matched_row["ref_answer"].values[0]
    
    question_id = matched_row["question_id"].values[0]
    
    # Preprocess answers
    student_modified = preprocess_text(student_answer)
    ref_modified = preprocess_text(ref_answer)
    qn_modified = preprocess_text(question)  # Preprocess question
    
    # Demote answers (remove question words)
    student_demoted = student_modified.replace(qn_modified, "").strip()
    ref_demoted = ref_modified.replace(qn_modified, "").strip()
    
    # Compute length ratio
    length_ratio = len(student_modified.split()) / max(len(ref_modified.split()), 1)
    
    # Compute embeddings (SBERT)
    embed_ref_modified = sbert_model.encode(ref_modified).tolist()
    embed_stud_modified = sbert_model.encode(student_modified).tolist()
    embed_ref_demoted = sbert_model.encode(ref_demoted).tolist()
    embed_stud_demoted = sbert_model.encode(student_demoted).tolist()
    
    # Compute Word2Vec embeddings
    embed_ref = get_word2vec_embedding(ref_answer)
    embed_stud = get_word2vec_embedding(student_answer)
    embed_ref_demoted = get_word2vec_embedding(ref_demoted)
    embed_stud_demoted = get_word2vec_embedding(student_demoted)

    # Compute cosine similarities
    cos_similarity_modified = cosine_similarity([embed_ref_modified], [embed_stud_modified])[0][0]
    cos_similarity_demo = cosine_similarity([embed_ref_demoted], [embed_stud_demoted])[0][0]
    cos_similarity = cosine_similarity([embed_ref], [embed_stud])[0][0]

    # Compute word alignment scores
    aligned_score = get_alignment_score(ref_answer, student_answer)
    aligned_score_demo = get_alignment_score(ref_demoted, student_demoted)

    # Compute combined similarity
    combined_similarity = (0.4 * cos_similarity_demo) + (0.6 * cos_similarity_modified)
    
    # ✅ Assign grade based on combined similarity
    def assign_grades(cos_sim):
        if cos_sim > 0.81:
            return "Completely Correct", 2
        elif 0.27 < cos_sim <= 0.69:  # Fixed range condition
            return "Partially Incorrect", 1
        elif cos_sim <= 0.27:
            return "Incorrect", 0
        return "Uncertain", None  # Added fallback case


    grade_result = assign_grades(combined_similarity)  # Call function

    print("Grade Result from assign grades",grade_result)

    if grade_result is not None:  # Prevent unpacking None
        grade_text, grades_auto = grade_result
    else:
        grade_text, grades_auto = "Uncertain", None  # Handle unexpected cases

    
     # Get Gemini feedback for all questions
    gemini_feedback = get_gemini_feedback(question, student_answer, ref_answer)
    
    # ✅ Check if Gemini LLM is needed for grading borderline cases
    if 0.69 <= combined_similarity <= 0.81 or 0.27 <= combined_similarity <= 0.49:
        gemini_grade, borderline_feedback = get_gemini_grade(question, student_answer, ref_answer)

        print("This is gemini grade:", gemini_grade) 

        if gemini_grade is not None:
            grades_auto = gemini_grade
            grade_text = ["Incorrect", "Partially Incorrect", "Completely Correct"][gemini_grade]  # Correctly assign text
            gemini_feedback = borderline_feedback  # Use the feedback from grading for borderline cases

    print("Combined Similarity",combined_similarity)
    # ✅ Return results
    return {
        "question_id": int(question_id),  # Convert int64 to Python int
        "question": question,
        "student_answer": student_answer,
        "ref_answer": ref_answer,
        "student_modified": student_modified,
        "qn_modified": qn_modified,
        "ref_modified": ref_modified,
        "student_demoted": student_demoted,
        "ref_demoted": ref_demoted,
        "length_ratio": length_ratio,
        "embed_ref": embed_ref,
        "embed_stud": embed_stud,
        "embed_ref_modified": embed_ref_modified,
        "embed_stud_modified": embed_stud_modified,
        "embed_ref_demoted": embed_ref_demoted,
        "embed_stud_demoted": embed_stud_demoted,
        "aligned_score": float(aligned_score),
        "aligned_score_demo": aligned_score_demo,
        "cos_similarity": float(cos_similarity),  # Convert NumPy float64 to Python float
        "cos_similarity_modified": cos_similarity_modified,
        "cos_similarity_demo": cos_similarity_demo,
        "combined_similarity": float(combined_similarity),
        "grade_text": grade_text,
        "grades_auto": int(grades_auto),  # Convert int64 to Python int
        "gemini_feedback": gemini_feedback,
        "ai_probability": ai_probability
    }

