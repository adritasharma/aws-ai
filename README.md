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

### ML Algorithms

Supervised vs. Unsupervised Learning
Supervised Learning
Supervised learning is when an AI model learns from labeled data to predict outcomes for new, unseen input data.
For example, imagine we want to predict a person's weight based on their height. We have a dataset where each person’s height is mapped to their weight. By plotting this data on a graph, we see a trend—some people are tall and light, while others are short and heavy.
To find a pattern, we apply linear regression, which creates a straight line that fits the general trend of these data points. Even if the data isn’t perfect, this line helps predict new values—for instance, a 1.6-meter-tall person might weigh 60 kg based on the model.
Supervised learning is powerful because it learns from labeled examples. However, collecting labeled data for millions of data points can be difficult.

![image](https://github.com/user-attachments/assets/73c82da0-5647-4abd-bf47-c0eb164f4a6e)


Unsupervised Learning
Unsupervised learning doesn’t need labeled data—instead, it finds hidden patterns in raw information.
For example, instead of predicting weight, imagine we want to group animals by height and weight without pre-labeling them as dogs, cats, or giraffes. The AI will cluster similar animals based on physical traits—some may share weight ranges, while others have similar heights.
Since the AI isn’t given categories upfront, it discovers patterns on its own, which is useful for tasks like customer segmentation, anomaly detection, and recommendation systems.

![image](https://github.com/user-attachments/assets/8c9f2594-035c-4f33-9f65-4c7084b84425)

Example: Customer Segmentation in E-Commerce
An online store wants to group customers based on their spending habits and purchase frequency. Using unsupervised learning (clustering), the AI identifies four customer segments:
1️⃣ High Spenders – Frequently buy expensive products.
2️⃣ Occasional Buyers – Purchase small items occasionally.
3️⃣ Discount Seekers – Shop mostly during sales.
4️⃣ Loyal Customers – Regular shoppers with high brand loyalty



Key Differences
| Feature | Supervised Learning | Unsupervised Learning | 
| Uses labeled data? | ✅ Yes | ❌ No | 
| Predicts outcomes? | ✅ Yes (Regression, Classification) | ❌ No (Finds patterns) | 
| Example task | Predicting weight from height | Grouping animals by traits | 
| Common algorithms | Linear regression, Decision Trees | Clustering (K-Means), PCA | 



![image](https://github.com/user-attachments/assets/64223a35-9aa8-4e81-b5cc-d419a770fcc4)

Regression vs. Classification in Machine Learning
Machine learning algorithms can get more complex, but let’s break down the basics using a simple example.

Classification – Identifying a Giraffe
Imagine we have different animals—dogs, cats, and giraffes.
Each has a height and weight, and we want our AI model to identify animals based on these traits.
✔ Giraffes are tall and heavy, while dogs and cats are smaller.
✔ If we give the algorithm a height of 4.5 meters and a weight of 800 kg,
✔ The model checks the data and classifies it as a giraffe.
Here, the AI isn’t predicting a numerical value—it’s categorizing the input into predefined labels.
✅ This is classification, not regression.

Regression – Predicting House Prices
Regression is different—it predicts continuous numerical values based on input data.
✔ For example, let’s predict house prices based on size.
✔ We plot house sizes against their prices.
✔ A linear regression model draws a straight line through the data.
✔ Now, if we input a new house size, the model predicts its price based on the trend.
✅ Regression predicts a quantity (house price, stock value, temperature) rather than categories.

Classification in Real Life – Spam Filters
Binary Classification separates data into two categories (spam or not spam).
✔ We train a model with labeled emails—some spam, some not spam.
✔ The AI learns patterns distinguishing spam messages.
✔ When a new email arrives, the model checks it and decides:
Spam or Not Spam?
Similarly, Multi-Class Classification applies to cases like recognizing mammals, birds, reptiles, or multi-label tasks, such as tagging a movie as both comedy and action.


Understanding Training, Validation, and Test Sets in Machine Learning
When training a machine learning model, data is typically divided into three sets to ensure accuracy and reliability:

![image](https://github.com/user-attachments/assets/ef978d41-c49d-4f5e-8c5f-382766aabe77)


1️⃣ Training Set – Learning Phase
✅ Purpose: Teaches the AI model how to recognize patterns.
✅ Size: ~60-80% of total data
✅ Example: If we have 1,000 images, we use 800 labeled images to train the model.
📌 The model learns relationships between inputs and expected outputs.

2️⃣ Validation Set – Tuning the Model
✅ Purpose: Adjusts model parameters to improve performance.
✅ Size: ~10-20% of total data
✅ Example: From 1,000 images, we use 100 labeled images to fine-tune settings.
📌 Helps optimize model accuracy before testing.

3️⃣ Test Set – Final Evaluation
✅ Purpose: Checks how well the trained model performs on new unseen data.
✅ Size: ~10-20% of total data
✅ Example: From 1,000 images, we keep 100 images for final testing.

📌 If we submit an image of a cat, the model should correctly label it as “Cat.”

Why This Split Matters?
✔ Prevents Overfitting – Ensures the model generalizes well.
✔ Improves Accuracy – Identifies errors before real-world use.
✔ Reliable Predictions – Validates performance across different datasets


Feature Engineering – Preparing Data for Machine Learning
Feature engineering is the process of selecting, transforming, and creating meaningful features from raw data to improve machine learning model performance.

Why is Feature Engineering Important?
✔ Helps the model better understand patterns in data
✔ Improves accuracy and efficiency
✔ Reduces irrelevant noise in datasets

Key Feature Engineering Techniques
🔹 Feature Extraction – Deriving useful features from existing data
- Example: Converting birth date into age (age is more useful than raw birth date)

🔹 Feature Selection – Choosing the most relevant features to use
- Example: If predicting house prices, focus on size and location rather than irrelevant details

🔹 Feature Transformation – Modifying features to improve performance
- Example: Scaling house sizes and prices to the same numerical range for better model training


Feature Engineering for Structured & Unstructured Data
📊 Structured Data Example – Predicting House Prices
- Raw Data: Size, Location, Number of Rooms, Price
- Engineered Features: Price per Square Foot, Neighborhood Rating
📝 Unstructured Data Example – Sentiment Analysis on Customer Reviews
- Raw Data: Text Reviews
- Engineered Features: Word Frequency (TF-IDF), Sentiment Score

🖼 Image Data Example – AI-Powered Image Recognition
- Raw Data: Photos
- Engineered Features: Edges, Textures, Colors (Extracted via Neural Networks)


Real-World Application
Feature engineering is crucial for supervised learning, where labeled data helps predict outcomes.
Example: Spam detection – AI analyzes email text and extracts features like word frequency, sender credibility, and length of email to classify messages as spam or not spam.






  

  
