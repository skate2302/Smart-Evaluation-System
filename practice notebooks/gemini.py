import os
import google.generativeai as genai

# print(os.environ.get("GEMINI_API_KEY"))

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {  # Dictionary to hold configuration settings for text generation
    "temperature": 1,      # Controls the "creativity" or randomness of the generated text.
                           # Higher values (e.g., 1) make the output more random and creative,
                           # while lower values (e.g., 0.2) make it more deterministic and predictable.
                           # Typical range: 0-1.

    "top_p": 0.95,        # Nucleus sampling.  A way to control randomness.
                           # Considers the most likely tokens whose cumulative probability exceeds 'top_p'.
                           # Higher values include more tokens, leading to more diverse text.
                           # Typical range: 0-1.  Often used in conjunction with top_k.

    "top_k": 40,          #  Considers the top 'k' most likely tokens.  Limits the model's choices.
                           #  A smaller value makes the text more focused and predictable.
                           #  A larger value allows for more diverse text.  
                           #  Often used in conjunction with top_p.  If top_p is also set, the
                           #  model will consider the top_k tokens *and* those that meet the top_p
                           #  criteria.

    "max_output_tokens": 8192,  # Maximum number of tokens (roughly words or sub-words) to generate.
                               # Limits the length of the generated text.  8192 is a large value;
                               # you might want to adjust it based on your needs and the model's
                               # capabilities.  There might be limits on the maximum.

    "response_mime_type": "text/plain", # Specifies the desired output format as plain text.  
                                       # Other MIME types might be supported depending on the model
                                       # and the API.
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("Abraham Lincoln was president twice?")

print(response.text)

# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import os
# import google.generativeai as genai

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# # ✅ Function to use Gemini for re-evaluation of mismatched grades
# def get_gemini_grade(question, student_answer, ref_answer):
#     """ Uses Google's Gemini LLM to verify and assign a grade if there's a mismatch. """
    

#     generation_config = {  # Dictionary to hold configuration settings for text generation
#         "temperature": 1,      # Controls the "creativity" or randomness of the generated text.
#                            # Higher values (e.g., 1) make the output more random and creative,
#                            # while lower values (e.g., 0.2) make it more deterministic and predictable.
#                            # Typical range: 0-1.

#         "top_p": 0.95,        # Nucleus sampling.  A way to control randomness.
#                            # Considers the most likely tokens whose cumulative probability exceeds 'top_p'.
#                            # Higher values include more tokens, leading to more diverse text.
#                            # Typical range: 0-1.  Often used in conjunction with top_k.

#         "top_k": 40,          #  Considers the top 'k' most likely tokens.  Limits the model's choices.
#                            #  A smaller value makes the text more focused and predictable.
#                            #  A larger value allows for more diverse text.  
#                            #  Often used in conjunction with top_p.  If top_p is also set, the
#                            #  model will consider the top_k tokens *and* those that meet the top_p
#                            #  criteria.

#         "max_output_tokens": 8192,  # Maximum number of tokens (roughly words or sub-words) to generate.
#                                # Limits the length of the generated text.  8192 is a large value;
#                                # you might want to adjust it based on your needs and the model's
#                                # capabilities.  There might be limits on the maximum.

#         "response_mime_type": "text/plain", # Specifies the desired output format as plain text.  
#                                        # Other MIME types might be supported depending on the model
#                                        # and the API.
#     }
    
#     model = genai.GenerativeModel(
#         model_name="gemini-2.0-flash",
#         generation_config=generation_config,
#     )
#     chat_session = model.start_chat(history=[])

#     # ✅ Send grading prompt to Gemini
#     prompt = f"""
#     You are an expert automated grading system. Evaluate the student's answer based on the reference answer.
    
#     **Grading System:**
#     - Completely Correct (2) → The answer fully aligns with the reference.
#     - Partially Incorrect (1) → Some information is correct, but the answer is incomplete or slightly incorrect.
#     - Incorrect (0) → The answer is mostly incorrect.

#     **Evaluate the following:**
#     - **Question:** {question}
#     - **Student Answer:** {student_answer}
#     - **Reference Answer:** {ref_answer}
    
#     What grade (0, 1, or 2) should be assigned?
#     """

#     response = chat_session.send_message(prompt)
    
#     # ✅ Extract numeric grade from Gemini's response
#     try:
#         text_response = response.text.strip()
#         if "0" in text_response:
#             return 0
#         elif "1" in text_response:
#             return 1
#         elif "2" in text_response:
#             return 2
#         else:
#             return None  # Default case if Gemini returns an unexpected format
#     except:
#         return None  # Handle errors gracefully
    

# # ✅ Apply Gemini for mismatched grades
# df["gemini_grade"] = df.apply(
#     lambda row: get_gemini_grade(row["question"], row["student_answer"], row["ref_answer"])
#     if row["grades_auto"] != row["grades_round"] else row["grades_auto"], axis=1
# )

# # ✅ Save final results
# df.to_csv("graded_results.csv", index=False)

# print("Grading complete! Results saved in graded_results.csv")