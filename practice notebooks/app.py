content     = """
The idea of AI taking over the world is a mix of science fiction and real concerns.
While AI is revolutionizing industries, automating jobs, and influencing decisions, true autonomy remains far off. 
Current AI lacks human-like reasoning, emotions, and self-awareness. 
However, rapid advancements in machine learning and robotics raise ethical and security concerns. 
Unchecked AI could disrupt economies, amplify biases, and challenge human control. 
Governments and researchers must ensure AI remains beneficial through regulation and ethical design. 
Rather than a hostile takeover, the real challenge is managing AI’s integration into society without compromising human values and decision-making.
"""

content2 = """
AI taking over the world is a combination of real-world worries and science fiction.
True autonomy is still a long way off, even though AI is transforming industries, automating tasks, and influencing decisions. 
Human-like emotions, reasoning, and self-awareness are absent from current AI. 
However, there are security and ethical questions raised by the quick developments in robotics and machine learning. 
Unchecked AI has the potential to undermine human control, increase biases, and upend economies. 
Through ethical design and regulation, governments and researchers must guarantee AI's continued benefits. 
Managing AI's integration into society without sacrificing human values and decision-making is the true challenge, not a hostile takeover.

"""
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
        perplexity = torch.exp(loss).item()  # Convert loss to perplexity
    
    return perplexity

def perplexity_to_ai_probability(perplexity, max_ppl=200):
    """
    Converts perplexity score to an AI probability percentage.
    Lower perplexity → Higher AI confidence.
    """
    if perplexity <= 1:  # Prevent divide-by-zero or log(0)
        return 100.0
    
    probability = (1 - (math.log(perplexity) / math.log(max_ppl))) * 100
    return max(0, min(probability, 100))  # Ensure range is 0-100%


# 🔹 **Test Cases**
test_texts = [
    """
The idea of AI taking over the world is a mix of science fiction and real concerns.
While AI is revolutionizing industries, automating jobs, and influencing decisions, true autonomy remains far off. 
Current AI lacks human-like reasoning, emotions, and self-awareness. 
However, rapid advancements in machine learning and robotics raise ethical and security concerns. 
Unchecked AI could disrupt economies, amplify biases, and challenge human control. 
Governments and researchers must ensure AI remains beneficial through regulation and ethical design. 
Rather than a hostile takeover, the real challenge is managing AI’s integration into society without compromising human values and decision-making.
"""
,
    "The sun was shining, and the birds were singing as I walked through the park.",
    "010101110 AI text detection is a fascinating field of research with deep learning advancements.",
    """
apply input data to input layer and initialize small values weights minimize error according to 
difference between desired signal and output signal assign the test vector the class that has smallest 
error 
    ""","""
 SOM is refered to as Self organized maps which is an unsupervised training algorithm for finding 
spatial pattern in data without using any external help.The process in SOM is explained below: 
Initialization: Initialize random weights wj for input patterns - Sampling: Take nth random sample from 
the input (say x) - similarity matching :for the input x, find the best match in the weight vector. $i(x) = 
argmin(x - w)$ - update: the next step is to update the weights $w(n+1) = w(n) + eta*hji(x)*i(x)$ - 
continuation : continue from sampling until there is no significant change in the feature map 
""",
"""
Self-Organizing Maps (SOM) is an unsupervised learning algorithm designed to detect spatial patterns in data without external supervision. The process begins with the initialization of random weight vectors for input patterns. During training, a random sample is selected from the dataset, and the algorithm identifies the best-matching unit (BMU) by minimizing the difference between the input vector and weight vectors. The weight vectors are then updated using a learning rate and a neighborhood function to ensure that similar data points are mapped closely together. This weight adjustment follows the rule 
𝑤
(
𝑛
+
1
)
=
𝑤
(
𝑛
)
+
𝜂
⋅
ℎ
𝑗
𝑖
(
𝑥
)
⋅
(
𝑖
(
𝑥
)
−
𝑤
(
𝑛
)
)
w(n+1)=w(n)+η⋅h 
ji
​
 (x)⋅(i(x)−w(n)), where 
𝜂
η is the learning rate and 
ℎ
𝑗
𝑖
(
𝑥
)
h 
ji
​
 (x) is the neighborhood function. The process continues iteratively until the feature map stabilizes, ensuring that the SOM effectively clusters and organizes high-dimensional data while preserving its topological structure.
""",
"""
In multi layer ff networks the error is only available in the last layer. Therefore the error is propagated back through the network using the backpropagtion algorithm. In order to do so the local gradient has to be calculated. Update of the weight: w+1 = w + n * x * gradient where the iput x is the output of the previous layer. The local gradient is calculated diffrently depending if the neuron is in the output layer or in the hidden layer. Output layer: $ gradient = phi`(x) * (y -d) $ Hidden Layer: $ gradient = phij`(x) * SUM(wi * local gradienti) $,
""","""
For your project, I would recommend the Perplexity-Based Approach (GPT-2) over the Deep Learning Classifier (RoBERTa). Here’s why:

🔥 Why Perplexity?
✅ Lightweight & Efficient → You’re already using SBERT + Word2Vec for similarity, so adding a pre-trained GPT-2 (without fine-tuning) keeps things fast & efficient.
"""
]

for i, text in enumerate(test_texts, 1):
    ppl = calculate_perplexity(text)
    ai_prob = perplexity_to_ai_probability(ppl)
    print(f"Test {i}: {text}\n🔥 Perplexity: {ppl:.2f} → 🤖 AI Probability: {ai_prob:.2f}%\n")