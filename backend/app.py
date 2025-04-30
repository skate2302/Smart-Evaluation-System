from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz
from grading_script import evaluate_student_answer
import logging
import re
import os
import io
from google.cloud import vision
from pdf2image import convert_from_path
import google.generativeai as genai


app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)


# Initialize Google Vision API client
client = vision.ImageAnnotatorClient()

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

def refine_text_with_gemini(ocr_text):
    """Send extracted OCR text to Gemini for refinement."""
    if not ocr_text.strip():
        return "No text detected."

    chat_session = model.start_chat(history=[])
    
    prompt = f"""
    You are an expert at cleaning and formatting text extracted using OCR.
    The given text may contain line breaks in incorrect places, missing punctuation, 
    and minor recognition errors due to handwriting variations.

    **Your task:**
    - Correct only OCR-related misinterpretations (e.g., 'Ql' → 'Q1', 'QS.' → 'Q5').
    - Keep all spelling, grammar, and content mistakes as they are.
    - Ensure that paragraphs flow naturally without unnecessary breaks.
    - Return the corrected text in a clean, readable format.

    **OCR Extracted Text:**
    {ocr_text}

    **Refined Text:**
    """
    
    response = chat_session.send_message(prompt)
    return response.text.strip()


def extract_questions_answers(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = ""
    # Extract text from all pages
    for page in doc:
        full_text += page.get_text("text") + "\n"
    # Use regex to find questions and their content
    # This pattern looks for "Q1.", "Q2.", etc. followed by text until the next question or end of text
    pattern = r'Q(\d+)\.\s+(.*?)(?=Q\d+\.|$)'
    matches = re.findall(pattern, full_text, re.DOTALL)
    
    questions = {}
    for num, content in matches:
        # Clean up the extracted text
        cleaned_content = content.strip()
        # Remove any extra whitespace or newlines
        cleaned_content = ' '.join(cleaned_content.split())
        questions[f"Q{num}"] = cleaned_content
    logging.debug(f"Extracted questions: {questions}")
    return questions


@app.route("/upload", methods=["POST"])
def upload_pdfs():
    try:
        # Get files from request
        question_pdf = request.files.get('questionPdf')
        
        # Check if it's bulk or single grading
        answer_pdfs = []
        for key in request.files:
            if key.startswith('answerPdf') and key != 'questionPdf':
                answer_pdfs.append(request.files[key])
        
        if not question_pdf or len(answer_pdfs) == 0:
            return jsonify({"error": "Question PDF and at least one answer PDF are required"}), 400
        
        # Extract questions from question PDF
        questions = extract_questions_answers(question_pdf)
        logging.debug(f"Questions: {questions}")
        
        # Store all results
        all_grading_results = []
        
        # Process each answer PDF
        for idx, answer_pdf in enumerate(answer_pdfs):
            # Extract answers for this student
            answers = extract_questions_answers(answer_pdf)
            logging.debug(f"Answers for Student {idx+1}: {answers}")
            
            # Grade each answer
            student_results = []
            for q_num in sorted(questions.keys()):  # Sort to maintain order
                question = questions.get(q_num, "")
                answer = answers.get(q_num, "")
                
                if question:  # If question exists
                    if answer:  # If there's an answer, grade it normally
                        try:
                            result = evaluate_student_answer(question, answer)
                            result['question_number'] = q_num
                            result['student_name'] = f"Student {idx+1}"
                            if 'gemini_feedback' not in result:
                                result['gemini_feedback'] = "No feedback available"
                        except Exception as e:
                            logging.error(f"Error grading {q_num} for Student {idx+1}: {str(e)}")
                            # Add a placeholder result for failed grading
                            result = {
                                "question_number": q_num,
                                "student_name": f"Student {idx+1}",
                                "question": question,
                                "student_answer": answer,
                                "ref_answer": "Error processing answer",
                                "grade_text": "Error",
                                "grades_auto": 0,
                                "combined_similarity": 0,
                                "gemini_feedback": f"Error during grading: {str(e)}",
                                "ai_probability": 0
                            }
                    else:  # No answer provided for this question
                        # Use the same logic as in evaluate_student_answer to find reference answer
                        try:
                            # Import df if needed (you might need to adjust this import)
                            from grading_script import df
                            
                            # Find reference answer
                            matched_row = df[df["question"] == question]
                            
                            if matched_row.empty:
                                ref_answer = "Question not found in dataset!"
                                question_id = 0
                            else:
                                ref_answer = matched_row["ref_answer"].values[0]
                                question_id = matched_row["question_id"].values[0]
                            
                            # Create a default result for missing answers
                            result = {
                                "question_id": int(question_id),
                                "question_number": q_num,
                                "student_name": f"Student {idx+1}",
                                "question": question,
                                "student_answer": "",  # Empty answer
                                "ref_answer": ref_answer,
                                "grade_text": "Incorrect",
                                "grades_auto": 0,
                                "combined_similarity": 0,
                                "ai_probability": 0,
                                "gemini_feedback": "No answer provided for this question."
                            }
                        except Exception as e:
                            logging.error(f"Error retrieving reference answer for {q_num}: {str(e)}")
                            result = {
                                "question_number": q_num,
                                "student_name": f"Student {idx+1}",
                                "question": question,
                                "student_answer": "",
                                "ref_answer": "Error retrieving reference answer",
                                "grade_text": "Incorrect",
                                "grades_auto": 0,
                                "combined_similarity": 0,
                                "ai_probability": 0,
                                "gemini_feedback": "No answer provided for this question."
                            }
                    
                    student_results.append(result)
            
            # Add this student's results to overall results
            all_grading_results.append(student_results)
        
        return jsonify({
            "success": True,
            "results": all_grading_results
        })
    except Exception as e:
        logging.error(f"Error processing PDFs: {str(e)}")
        return jsonify({"error": str(e)}), 500



def extract_text_from_pdf(pdf_path):
    """Convert PDF pages to images & extract text using Google Vision API, then refine with Gemini."""
    images = convert_from_path(pdf_path)
    full_text = ""

    for i, image in enumerate(images):
        print(f"\n🔍 Processing Page {i+1}...\n")

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        vision_image = vision.Image(content=img_byte_arr)
        response = client.document_text_detection(image=vision_image)

        # ✅ Extract only the full text (ignoring individual characters)
        extracted_text = response.full_text_annotation.text if response.full_text_annotation.text else "No text detected"

        # Append extracted text from all pages into a single string
        full_text += extracted_text + "\n\n"  # ✅ Ensure paragraphs are properly separated

    # 🔹 Send entire extracted text to Gemini for refinement
    refined_text = refine_text_with_gemini(full_text)

    # 🔹 Extract question-based sections from refined text while preserving paragraphs
    pattern = r'Q(\d+)\.\s+(.*?)(?=Q\d+\.|$)'  # Detects "Q1.", "Q2.", ..., followed by text
    matches = re.findall(pattern, refined_text, re.DOTALL)

    refined_questions = {}
    for num, content in matches:
        cleaned_content = content.strip()
        cleaned_content = re.sub(r'\n+', '\n', cleaned_content)  # ✅ Preserve line breaks correctly
        refined_questions[f"Q{num}"] = cleaned_content  # Store question as key-value pair

    logging.debug(f"Refined Questions: {refined_questions}")

    return refined_questions  # ✅ Returns refined text question by question


@app.route("/ocr-upload", methods=["POST"])
def ocr_upload():
    try:
        # Get files from request
        question_pdf = request.files.get("questionPdf")
        handwritten_pdf = request.files.get("handwrittenPdf")

        if not question_pdf or not handwritten_pdf:
            return jsonify({"error": "Both Question PDF and Handwritten PDF are required"}), 400

        # Save PDFs temporarily
        question_pdf_path = os.path.join("uploads", question_pdf.filename)
        handwritten_pdf_path = os.path.join("uploads", handwritten_pdf.filename)
        os.makedirs("uploads", exist_ok=True)
        question_pdf.save(question_pdf_path)
        handwritten_pdf.save(handwritten_pdf_path)

        # Extract questions and student answers
        with open(question_pdf_path, "rb") as pdf_file:
            extracted_questions = extract_questions_answers(pdf_file)

        refined_questions = extract_text_from_pdf(handwritten_pdf_path)  # ✅ Update variable name


        # Match answers to questions
        matched_answers = {}
        for q_num, student_answer in refined_questions.items():  # ❌ Incorrect variable name
            question_text = extracted_questions.get(q_num, f"Q{q_num} (question not found)")
            matched_answers[q_num] = {"question": question_text, "student_answer": student_answer}

        grading_results = []

        # Compare extracted answers with reference answers & grade them
        for q_num, data in matched_answers.items():
            question_text = data["question"]
            student_answer = data["student_answer"]

            try:
                # ✅ Call evaluate_student_answer() using extracted question text
                result = evaluate_student_answer(question_text, student_answer)
                result["question_number"] = q_num  # ✅ Add question number for frontend display

                # ✅ Ensure Gemini feedback exists
                if "gemini_feedback" not in result:
                    result["gemini_feedback"] = "No feedback available"

                grading_results.append(result)
            except Exception as e:
                logging.error(f"Error grading {q_num}: {str(e)}")
                grading_results.append({
                    "question_number": q_num,
                    "question": "Error retrieving reference answer",
                    "student_answer": student_answer,
                    "ref_answer": "Error processing answer",
                    "grades_auto": 0,
                    "combined_similarity": 0,
                    "ai_probability": 0,  # Default AI probability when error occurs
                    "gemini_feedback": f"Error during grading: {str(e)}"
                })

        logging.debug(f"Grading Results: {grading_results}")

        return jsonify({
            "success": True,
            "grading_results": grading_results,
            "extracted_text": {
                "refined_text": refined_questions,  # ✅ Send full refined text
                "questions": extracted_questions  # ✅ Send extracted questions
            }

        })

    except Exception as e:
        logging.error(f"Error processing OCR and grading: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

