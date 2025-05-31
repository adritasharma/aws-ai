# AWS AI
Learning AI AWS services

### 

How does AI work:

- Collect Dataser (eg: images of different fruits - bananas, peaches, apples etc)
- Train model to create a AI model (AI model is a smart comupter code)
- Run a algorithm (eg : Classification Algorithm ) - it groups the images of all the fruits - apples together,  bananas togethe etcr)
- User provides a new image of apple to the model - since model is trained with so many images og apple that it identifies it as an apple

### AI Components

- Data Layer
  
  Collect vast amount fa data
  
- ML Framework and Algorithm Layer

  Defina a ML framework to solve a problem

- Model Layer
  
  We set parameter, function, optimizer function to create a actual model and train it

- Application Layer
  
  Expose the model to users

### Machone Laerning

  ML is a type of AI to build methods that allow machines to learn from data. We can make prediction based on data used to train the models.

**Regression**

Regression is a machine learning technique used to predict numerical values based on past data.
Example: Predicting house prices based on size, location, and amenities.

**Classification**

Classification is used when we need to categorize data into groups.
Example: Classifying emails as spam or not spam, or recognizing if an image contains a dog or a cat.

**Deep Learning**

Deep learning is an advanced type of machine learning that uses multi-layered neural networks to process data and learn patterns.
Example: Deep learning powers speech recognition, self-driving cars, and AI-generated images.

Neural Networks
Neural networks are inspired by how the human brain works. They consist of layers of nodes (neurons) that process data in a structured way.
Example: Facial recognition on smartphones uses neural networks to detect faces.

Newral network example:

A neural network is trained to recognize handwritten numberts.
Tge input layer process pixel data while the hiddenlayers identify features like lines and curves. For instance, vertical lines are present in numbers 1, 4 and 7 so a layer might detect such lines. Similarly, the numbers 6, 8 and 0 have curved buttoms which another layer may recognize. By combining these learned features, the network can accurately identify numbers

Generative AI (Gen-AI) 

It refers to artificial intelligence that uses foundation models trained on vast amounts of unlabeled data to generate new content—such as text, images, music, and code. These models learn underlying patterns and relationships, allowing them to adapt to different generation tasks without needing specific training for each scenario. This adaptability makes Gen-AI powerful for tasks like content creation, automated design, and AI-assisted problem-solving.

Gem AI models leverage Transformer model (LLM)

A Transformer model process sentence as whole instead of word by word. It processes text by focusing on important words in a sentence, instead of reading everything in order. This makes it really good at tasks like translating languages, summarizing text, and answering questions.

A Large Language Model (LLM) is a Transformer trained on a huge amount of text to learn how people write and speak. It can generate responses, explain concepts, and even help with writing code. Examples include GPT, BERT, and LLaMA, which power chatbots and AI assistants.

A diffusion model is an AI technique that learns to create realistic images by gradually refining noise into meaningful patterns. It works by reversing a process that first adds random noise to an image, then systematically removes it to generate high-quality visuals. These models are widely used in AI-powered image generation, like DALL·E and Stable Diffusion.
Would you like an example of how diffusion models improve image quality? 🚀


A multimodal model is an AI system that processes and understands multiple types of data, like text, images, audio, and video, at the same time. It can combine information from different sources to provide more accurate and meaningful responses. Examples include GPT-4V, which can analyze both text and images together.

### ML Terms

Sure! Here are the one-liner explanations along with their full forms:

- **GPT (Generative Pre-trained Transformer)** – A deep learning model that generates human-like text by predicting the next word in a sentence.  
- **BERT (Bidirectional Encoder Representations from Transformers)** – An NLP model that understands the context of words in a sentence by analyzing them from both directions.  
- **RNN (Recurrent Neural Network)** – A neural network designed for sequential data, like speech and time-series analysis, by remembering previous inputs.  
- **ResNet (Residual Neural Network)** – A deep learning model that solves the vanishing gradient problem, making neural networks more efficient for image recognition.  
- **SVM (Support Vector Machine)** – A machine learning algorithm that classifies data by finding the optimal boundary between different categories.  
- **WaveNet** – A deep learning model for generating realistic human speech by processing audio waveforms.  
- **GAN (Generative Adversarial Network)** – A framework where two neural networks (generator and discriminator) compete to create highly realistic images and data.  
- **XGBoost (eXtreme Gradient Boosting)** – A powerful machine learning algorithm that improves prediction accuracy by sequentially combining multiple decision trees while minimizing errors



### **Training Data in AI & Machine Learning**
Training data is the **foundation** of any AI model—it consists of examples that help the model learn patterns and make predictions. High-quality data leads to **accurate AI models**, while poor data can result in **biased or faulty results**.

---

### **Types of Training Data**
1️⃣ **Labeled vs. Unlabeled Data**  
- **Labeled Data**: Each input has a **known outcome** (e.g., images labeled as "dog" or "cat").  
  ✅ Used in **supervised learning** for classification and regression.  
- **Unlabeled Data**: The model finds patterns **without predefined labels** (e.g., clustering customer behaviors).  
  ✅ Used in **unsupervised learning** for feature discovery.  

---

2️⃣ **Structured vs. Unstructured Data**  
- **Structured Data**: Organized in a fixed format, like tables or databases (e.g., financial transactions, Excel sheets, **time series data** (stocks price over a year)).  
  ✅ Easy to process using SQL, dataframes, and traditional algorithms.  
- **Unstructured Data**: No predefined format, like , **text, images, videos, audio files** (e.g., emails, social media posts).  
  ✅ Requires advanced AI techniques like **Natural Language Processing (NLP)** or **Computer Vision**.

---

3️⃣ **Garbage Data (Bad Training Data)**  
🚨 **Garbage In, Garbage Out** – Poor-quality data leads to inaccurate AI models!  
- **Noisy Data**: Contains **errors, duplicate values, inconsistencies**.  
- **Bias in Data**: Unbalanced datasets can create **biased AI decisions**.  
- **Incomplete Data**: Missing values can **mislead AI models**.  

Example: An AI model trained on **biased hiring data** might unintentionally **discriminate** against certain candidates.  

---

### **Why Good Training Data Matters?**
✅ Improves AI accuracy  
✅ Reduces bias in predictions  
✅ Helps AI make meaningful decisions  

Would you like an example of **how training data affects real-world AI models**? 🚀




  

  
