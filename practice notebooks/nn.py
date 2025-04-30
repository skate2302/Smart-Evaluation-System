import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

import torch
print("PyTorch using GPU:", torch.cuda.is_available())

import tensorflow as tf
print("TensorFlow using GPU:", len(tf.config.list_physical_devices('GPU')) > 0)

# ✅ Load SBERT model
sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# # ✅ Load Word2Vec model from gensim
# word2vec_model = api.load("fasttext-wiki-news-subwords-300")  

# ✅ Load Word2Vec model from gensim
word2vec_model = api.load("word2vec-google-news-300")  

# ✅ Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ✅ Load dataset
df = pd.read_csv("data/Final_Dataset.csv")

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

# ✅ Function to use Gemini LLM for ambiguous cases
def get_gemini_grade(question, student_answer, ref_answer):
    """ Uses Gemini LLM to assign a grade for uncertain cases. """

    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    chat_session = model.start_chat(history=[])

    prompt = f"""
    You are a strict and an expert automated grading system. Evaluate the student's answer based on the reference answer.
    
    **Grading System:**
    - Completely Correct (2) → The answer fully aligns with the reference, Slight mistakes are allowed.
    - Partially Incorrect (1) → Some information is correct, but the answer is incomplete or partially incorrect based on the Reference answer provided.
    - Incorrect (0) → The answer is incorrect based on the reference answer provided.

    **Evaluate the following:**
    - **Question:** {question}
    - **Student Answer:** {student_answer}
    - **Reference Answer:** {ref_answer}
    
    
    """    
    print("im Gemini")
    response = chat_session.send_message(prompt)

    print("Response from Gemini", response.text)

    try:
        text_response = response.text.strip()
        if "2" in text_response:
            return 2
        elif "1" in text_response:
            return 1
        elif "0" in text_response:
            return 0
        else:
            return None  # Handle unexpected response
    except:
        return None  # Handle errors

# ✅ Function to evaluate student answer
def evaluate_student_answer(question, student_answer):
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

    
    # ✅ Check if Gemini LLM is needed
    if 0.69 <= combined_similarity <= 0.81 or 0.27 <= combined_similarity <= 0.49:
        gemini_grade = get_gemini_grade(question, student_answer, ref_answer)

        print("This is gemini grade:", gemini_grade) 

        if gemini_grade is not None:
            grades_auto = gemini_grade
            grade_text = ["Incorrect", "Partially Incorrect", "Completely Correct"][gemini_grade]

    # ✅ Return results
    return {
        "question_id": question_id,
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
        "aligned_score": aligned_score,
        "aligned_score_demo": aligned_score_demo,
        "cos_similarity": cos_similarity,
        "cos_similarity_modified": cos_similarity_modified,
        "cos_similarity_demo": cos_similarity_demo,
        "combined_similarity": combined_similarity,
        "grade_text": grade_text,
        "grades_auto": grades_auto
    }

#✅ Example Usage
question_input = "Give a definition for the term ‘artificial neural network’ and mention, how it resembles the human brain!"
student_answer_input = "An artificial neural network is a model inspired by the human brain."

result = evaluate_student_answer(question_input, student_answer_input)

# ✅ Print Results
for key, value in result.items():
    print(f"{key}: {value}")
