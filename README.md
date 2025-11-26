# 🎬 AlgoSuggest

<div align="center">

![Version](https://img.shields.io/badge/version-2.1-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)

**An intelligent movie recommendation engine powered by machine learning** ✨

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [Tech Stack](#-tech-stack)

</div>

---

## 🌟 Overview

**AlgoSuggest** is a production-ready movie recommendation system that uses **content-based filtering** to suggest movies based on your preferences. Built with machine learning algorithms and a sleek, modern UI, it delivers personalized recommendations in milliseconds.

### 🎯 What Makes It Special?

- 🧠 **Smart Recommendations**: Uses TF-IDF vectorization and cosine similarity
- 🔍 **Fuzzy Search**: Finds movies even with typos (e.g., "Avatr" → "Avatar")
- ⚡ **Lightning Fast**: Lives autocomplete with <300ms response time
- 🎨 **Premium UI**: Orange & Black theme with glowing effects inspired by Apple/Pixel design
- 🚀 **Production Ready**: Runs on Gunicorn with 4 workers for high concurrency
- 📊 **4800+ Movies**: TMDB 5000 Movies dataset included

---

## ✨ Features

### Core Functionality

| Feature | Description |
|---------|-------------|
| 🎭 **Title-Based Search** | Get recommendations based on a movie you already love |
| 🎪 **Genre Recommendations** | Discover top-rated movies by genre (Action, Sci-Fi, Drama, etc.) |
| 🔮 **Smart Autocomplete** | Real-time suggestions as you type |
| 🎯 **Fuzzy Matching** | Handles typos and approximate matches (40% similarity threshold) |
| ⭐ **Weighted Ratings** | Uses IMDB's weighted rating formula to rank movies by quality |

### User Experience

- 🌈 **Animated UI**: Smooth fade-ins and glow effects
- 📱 **Responsive Design**: Works perfectly on all screen sizes
- 🎨 **Modern Aesthetics**: Glassmorphism, backdrop blur, and gradient accents
- 🔔 **Smart Errors**: Helpful error messages when no results are found

---

## 🏗️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Lightweight web framework
- **Gunicorn** - Production WSGI server
- **scikit-learn** - Machine learning (TF-IDF, Cosine Similarity)
- **pandas** - Data manipulation
- **NumPy** - Numerical computing

### Frontend
- **React 18** (via CDN)
- **Tailwind CSS** - Utility-first styling
- **Babel** - JSX transpilation
- **Lucide Icons** - Beautiful SVG icons

### Data
- **TMDB 5000 Movies Dataset** (CSV format)

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- A modern web browser

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/algosuggest.git
cd algosuggest
```

### Step 2: Install Dependencies

```bash
pip3 install -r requirements.txt
```

**Or install manually:**

```bash
pip3 install pandas numpy scikit-learn flask flask-cors gunicorn
```

### Step 3: Add the Dataset

Place the `tmdb_5000_movies.csv` file in the project root directory. This file should contain columns: `title`, `genres`, `keywords`, `overview`, `vote_count`, `vote_average`.

> **📝 Note:** The first time you run the app, it will automatically generate model files (`cosine_sim.npz`, `indices.pkl`, `metadata_processed.pkl`). This takes ~30 seconds. These files are cached for future runs and are **not included in the repository** due to their size (77MB+).

---

## 🚀 Usage

### Running the Server

#### Production Mode (Recommended)

```bash
./start_server.sh
```

This will:
- ✅ Automatically kill any process on port 5000
- ✅ Start Gunicorn with 4 workers
- ✅ Pre-load the ML model for faster responses

#### Development Mode

```bash
python3 recommendation_engine.py
```

### Accessing the App

1. **Start the server** (see above)
2. **Open `index.html`** in your browser by double-clicking it
3. **Start discovering movies!** 🎉

---

## 🎨 Screenshots

### Main Interface
*Orange & Black theme with glowing effects*

### Autocomplete in Action
*Real-time suggestions as you type*

### Recommendations Display
*Clean, numbered list with smooth animations*

---

## 🧠 How It Works

### 1. Data Preprocessing
```python
# Extract genres and keywords from JSON strings
# Combine with movie overview to create "features soup"
features = genres + keywords + overview
```

### 2. Vectorization (TF-IDF)
```python
# Convert text features to numerical vectors
tfidf = TfidfVectorizer(stop_words='english', min_df=2)
tfidf_matrix = tfidf.fit_transform(df['features_soup'])
```

### 3. Similarity Calculation
```python
# Compute cosine similarity between all movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

### 4. Recommendation Generation
```python
# For a given movie, find top N most similar movies
sim_scores = sorted(cosine_sim[movie_idx], reverse=True)[1:N+1]
```

### 5. Weighted Rating (for Genre)
```python
# IMDB's formula for quality-based ranking
WR = (v / (v + m) × R) + (m / (v + m) × C)
# v = vote count, m = threshold, R = rating, C = mean rating
```

---

## 🛠️ API Endpoints

### 1. Get Title-Based Recommendations

**Endpoint:** `POST /api/recommend`

**Request:**
```json
{
  "title": "Avatar"
}
```

**Response:**
```json
{
  "recommendations": [
    "Aliens",
    "Mission to Mars",
    "Moonraker",
    ...
  ]
}
```

### 2. Get Genre Recommendations

**Endpoint:** `POST /api/recommend_genre`

**Request:**
```json
{
  "genre": "Sci-Fi"
}
```

**Response:**
```json
{
  "recommendations": [
    "Interstellar",
    "The Matrix",
    "Inception",
    ...
  ]
}
```

### 3. Autocomplete

**Endpoint:** `GET /api/autocomplete?query=avat`

**Response:**
```json
{
  "suggestions": [
    "Avatar",
    "Avatar: The Way of Water"
  ]
}
```

---

## 📂 Project Structure

```
algosuggest/
│
├── recommendation_engine.py  # 🧠 Backend ML engine + Flask API
├── index.html                # 🎨 Frontend React app (CDN)
├── start_server.sh           # 🚀 Production startup script
├── tmdb_5000_movies.csv      # 📊 Movie dataset
├── .gitignore                # 🚫 Git ignore rules
│
├── cosine_sim.npz            # 💾 Auto-generated similarity matrix (77MB)
├── indices.pkl               # 💾 Auto-generated movie indices
├── metadata_processed.pkl    # 💾 Auto-generated preprocessed data
│
└── README.md                 # 📖 You are here!
```

> **Note:** Files marked as "Auto-generated" are created on first run and excluded from Git.

---

## ⚙️ Configuration

### Backend Settings

Edit `recommendation_engine.py`:

```python
DATASET_PATH = './tmdb_5000_movies.csv'  # Path to dataset
COSINE_SIM_PATH = 'cosine_sim.pkl'       # Cache file
```

### Server Settings

Edit `start_server.sh`:

```bash
# Number of workers (adjust based on CPU cores)
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 recommendation_engine:app
```

### Frontend Settings

Edit `index.html` to change the API endpoint:

```javascript
const response = await fetch(`http://localhost:5000/api/recommend`, ...)
```

---

## 🎯 Genre Mapping

The system automatically handles variations:

| User Input | Maps To |
|------------|---------|
| `Sci-Fi` | `Science Fiction` |
| `Sci Fi` | `Science Fiction` |
| `SF` | `Science Fiction` |

---

## 🐛 Troubleshooting

### Port 5000 Already in Use

The `start_server.sh` script automatically handles this! It will:
```bash
# Find and kill any process using port 5000
lsof -t -i:5000 | xargs kill -9
```

### Frontend Can't Connect to Backend

1. ✅ Check if the server is running: `./start_server.sh`
2. ✅ Verify the URL in `index.html` is `http://localhost:5000`
3. ✅ Check browser console for CORS errors

### No Recommendations Found

- ✅ Check spelling (though fuzzy search should help!)
- ✅ Try searching with a different movie
- ✅ Verify the dataset is loaded correctly

---

## 🚀 Performance

- **Initial Load**: ~5 seconds (model loading)
- **Autocomplete**: <300ms
- **Recommendations**: <100ms (cached model)
- **Concurrent Users**: Supports 4+ simultaneous requests (Gunicorn)

---

## 🔮 Future Enhancements

- [ ] 🎭 Collaborative filtering (user-based recommendations)
- [ ] 🌐 Deploy to cloud (AWS, Heroku, Vercel)
- [ ] 📱 Mobile app (React Native)
- [ ] 🎬 Integrate with TMDB API for real-time data
- [ ] 👤 User accounts and watch history
- [ ] 🎯 Hybrid recommendation system
- [ ] 📊 Analytics dashboard

---

## 📄 License

MIT License - feel free to use this project for learning or commercial purposes!

---

## 🙏 Credits

- **TMDB** - Movie dataset
- **scikit-learn** - Machine learning library
- **Tailwind CSS** - Styling framework
- **React** - UI library
- Built with ❤️ by [Your Name]

---

## 🌟 Star This Repo!

If you found this project helpful, please give it a ⭐ on GitHub!

---

<div align="center">

**Made with 🎬 and ☕**

[⬆ Back to Top](#-algosuggest)

</div>
