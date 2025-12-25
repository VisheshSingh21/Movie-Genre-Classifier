<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
   </head>
<body> 

<header>
    <h1>🎬 Movie Genre Classification</h1>
    <p>Predicting movie genres using Machine Learning and NLP</p>
    <div>
        <img class="badge" src="https://img.shields.io/badge/Python-3.10-blue">
        <img class="badge" src="https://img.shields.io/badge/Jupyter-Notebook-orange">
        <img class="badge" src="https://img.shields.io/badge/Streamlit-App-red">
    </div>
</header>

<section>
    <h2>📌 Project Overview</h2>
    <p>This project develops a <span class="highlight">machine learning model</span> to predict the genre of a movie based on its plot description. 
       The project covers preprocessing, feature extraction using <span class="highlight">TF-IDF</span>, model training with <span class="highlight">Naive Bayes, Logistic Regression, and SVM</span>, and evaluation with accuracy, precision, recall, F1-score, confusion matrix, feature importance, and misclassification analysis.</p>
</section>

<section>
    <h2>📁 Dataset</h2>
    <p>IMDb Genre Classification Dataset from Kaggle:</p>
    <a href="https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb" target="_blank">Genre Classification Dataset – IMDb</a>
    <table>
        <tr><th>Column</th><th>Description</th></tr>
        <tr><td>id</td><td>Unique movie ID</td></tr>
        <tr><td>genre</td><td>Movie genre label</td></tr>
        <tr><td>plot</td><td>Text plot description of the movie</td></tr>
    </table>
</section>

<section>
    <h2>📂 Project Structure</h2>
    <pre>
Movie-Genre-Classifier/


│   ├── svm_genre_model.pkl
│   └── tfidf_vectorizer.pkl

│   └── Movie_Genre_Classifier.ipynb

│   └── streamlit_app.py
├── requirements.txt
└── README.html
    </pre>
</section>

<section>
    <h2>⚙️ Installation</h2>
    <pre>
git clone &lt;your-repo-url&gt;
cd Movie-Genre-Classifier
pip install -r requirements.txt
jupyter notebook notebooks/Movie_Genre_Classifier.ipynb
streamlit run app/streamlit_app.py
    </pre>
</section>

<section>
    <h2>📊 Model Evaluation</h2>
    <p><strong>Accuracy Comparison:</strong></p>
   <p>See the evaluation graphs ploted in code section</p>
</section>

<section>
    <h2>🖥️ Streamlit Web App</h2>
    <p>Input any movie plot and get:</p>
    <ul>
        <li>Predicted genre instantly</li>
        <li>Top contributing words for prediction</li>
        <li>Accuracy and metrics visualization</li>
    </ul>
    <pre>
new_plot = [
    "A fearless police officer fights corruption and crime in the city"
]
Predicted Genre: Action / Thriller
    </pre>
</section>

<section>
    <h2>💡 Skills Demonstrated</h2>
    <ul>
        <li>Python & Jupyter Notebook</li>
        <li>NLP preprocessing (stopwords, lemmatization, TF-IDF)</li>
        <li>Text classification (Naive Bayes, Logistic Regression, SVM)</li>
        <li>Model evaluation (accuracy, F1-score, confusion matrix)</li>
        <li>Data visualization (bar charts, heatmaps, feature importance)</li>
        <li>Model saving/loading (pickle)</li>
        <li>Web deployment using Streamlit</li>
    </ul>
</section>

<section>
    <h2>🚀 Future Enhancements</h2>
    <ul>
        <li>Use deep learning models (LSTM, BERT) for better context understanding</li>
        <li>Handle multi-label classification for movies with multiple genres</li>
        <li>Add more visual analytics in the web app</li>
        <li>Deploy app on Heroku or Streamlit Cloud for public access</li>
    </ul>
</section>

<section>
    <h2>📞 Contact & Collaboration</h2>
    <p><strong>Author:</strong> Vishesh Singh</p>
    <p><strong>GitHub:</strong> <a href="https://github.com/VisheshSingh21" target="_blank">https://github.com/VisheshSingh21</a></p>
    <p><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/VisheshSingh" target="_blank">https://www.linkedin.com/in/VisheshSingh</a></p>
    <p>Feel free to explore, contribute, or use this project as a learning reference!</p>
</section>

<footer>
    &copy; 2025 Vishesh Singh | Movie Genre Classification Project
</footer>

</body>
</html>
