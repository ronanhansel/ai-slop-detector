# AI Slop Detector Dashboard

A comprehensive real-time dashboard for analyzing political discourse and detecting toxic content, hate speech, and AI-generated text in social media comments.

![Dashboard Screenshot](https://img.shields.io/badge/Next.js-15-black?style=flat&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat&logo=typescript)
![Three.js](https://img.shields.io/badge/Three.js-r182-green?style=flat&logo=three.js)

## Overview

This dashboard visualizes and analyzes 181,000+ political discourse comments from social media, providing deep insights into sentiment patterns, hate speech prevalence, offensive content, and AI-generated text detection. Built with Next.js, TypeScript, and Three.js, it offers interactive 3D visualizations and real-time analysis tools.

## Features

### 📊 **Dashboard View**

#### Key Statistics

- **Total Comments**: 181,076 comments analyzed
- **Unique Users**: Track number of distinct commenters
- **Hate Speech Detection**: Count and percentage of hate speech
- **Offensive Content**: Identification of offensive language
- **Grok Interactions**: AI assistant (@grok) mentions and responses

#### Interactive 3D Visualization

- **Multi-dimensional scatter plot** powered by Three.js
  - **X-axis**: Toxicity Score (hate probability + offensive probability)
  - **Y-axis**: Rage-Bait Index (aggression + rage + anger + hate)
  - **Z-axis**: Sentiment (-1 to +1 scale)
- **Color-coded points**:
  - 🔴 Red: Hate speech detected
  - 🟠 Orange: Offensive content
  - 🔵 Blue: Negative sentiment
  - 🟢 Green: Positive sentiment
  - ⚪ Gray: Neutral
- **Interactive controls**:
  - Drag to rotate the 3D space
  - Hover over points to see comment details
  - Auto-rotation when not interacting
- **Performance optimized**: Displays up to 5,000 points simultaneously

#### Sentiment Distribution Chart

- Visual breakdown of positive, negative, and neutral comments
- Percentage and absolute counts
- Color-coded progress bars

#### Top Users Chart

- 10 most active commenters
- Activity metrics and visualization

### 👥 **User Analysis View**

#### Comprehensive User Table

- **Searchable**: Filter users by username
- **Sortable columns**:
  - Activity (total comment count)
  - Hate Score (average hate probability)
  - Offensive Score (average offensive probability)
- **User metrics**:
  - Total comments per user
  - Hate speech rate (%)
  - Offensive content rate (%)
  - Grok interactions count
  - User status badges:
    - 🔴 **FLAGGED**: User has posted hate speech
    - 🟡 **MONITOR**: High toxicity score (>30%)
    - 🟢 **NORMAL**: Clean record
- **Display limit**: Top 100 users shown
- **Total tracked**: All unique users from dataset

### 🧠 **Text Analyzer (Demo)**

A deterministic text analysis tool that evaluates content for:

#### Detection Capabilities

1. **Hate Speech Probability**

   - Keyword-based detection
   - Identifies hateful language patterns
   - Score: 0-95%

2. **Offensive Content Probability**

   - Profanity and aggressive language detection
   - Contextual analysis
   - Score: 0-95%

3. **AI-Generated Probability** 🤖

   - Detects AI-generated text patterns
   - Identifies common AI phrases ("as an ai", "language model", etc.)
   - Analyzes sentence structure and length
   - Detects formal/robotic vocabulary ("delve into", "utilize", "comprehensive")
   - Checks for overuse of formal connectors
   - Score: 0-95%

4. **Rage-Bait Index**

   - Composite score measuring inflammatory content
   - Combines toxicity, caps, and punctuation analysis
   - Scale: 0.0 - 1.0

5. **Sentiment Analysis**
   - Positive/Negative/Neutral classification
   - Confidence score
   - Real-time feedback

#### Text Features Analyzed

- Emoji count
- ALL CAPS words
- Hyperlink presence
- Exclamation marks
- Sentence length and complexity
- Formality markers

#### Interactive Interface

- Live text input
- Real-time analysis (800ms processing)
- Visual progress bars for all metrics
- Color-coded results

### 💾 **Data View**

#### Dataset Information

- **Source**: `final_merged_data_nlp.csv`
- **Total rows**: 181,076
- **Loaded comments**: 50,000 (optimized for performance)
- **Unique users**: Thousands of distinct commenters

#### Data Pipeline Console

- Live logging of data processing steps
- Real-time status updates:
  - CSV parsing
  - NLP feature extraction
  - Sentiment analysis computation
  - Hate speech probability calculation
  - Rage-bait index generation

#### Feature Columns

The dataset includes 60+ feature columns:

- **Identifiers**: commenter_id, comment_id, post_id
- **Content**: comment_content, cleaned_content
- **Metadata**: num_emojis, num_caps_words, contains_media, contains_link, tagged_grok, used_slang
- **NLP Labels**: sentiment_label, hate_label, offensive_label, irony_label
- **Probabilities**: sentiment_prob, hate_prob, offensive_prob, irony_prob
- **Empath Features**: aggression, anger, rage, violence, hate, negative_emotion, positive_emotion, politics, government, swearing_terms
- **LSA Dimensions**: 62 latent semantic analysis dimensions (lsa_dim_0 through lsa_dim_61)

#### NLP Pipeline

The data was processed using multiple state-of-the-art models:

1. **Sentiment Analysis**

   - Twitter-RoBERTa based model
   - Specialized for political discourse
   - Outputs: positive/negative/neutral with confidence

2. **Hate Speech Detection**

   - Fine-tuned transformer model
   - Binary classification: HATE / NOT-HATE
   - Probability scores for each comment

3. **Empath Features**
   - Lexical analysis framework
   - 194 emotional and topical categories
   - Specialized categories: rage, politics, violence, government

## Technology Stack

### Frontend

- **Next.js 15**: React framework with App Router
- **TypeScript 5**: Type-safe development
- **Three.js r182**: 3D visualization engine
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Icon library

### Data Processing

- **CSV Parsing**: Custom parser with 180K+ row support
- **Real-time API**: `/api/data` endpoint for data serving
- **Performance optimization**: Chunked loading, sampling for visualization

### Architecture

- **Server-side rendering**: Optimized initial page load
- **Client-side interactivity**: React hooks and state management
- **API Routes**: Next.js API handlers
- **Static assets**: CSV file served from `/public` directory

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-slop-detector.git
cd ai-slop-detector/dashboard

# Install dependencies
pnpm install

# Run development server
pnpm dev

# Build for production
pnpm build

# Start production server
pnpm start
```

## Usage

1. **Navigate tabs**: Use the sidebar to switch between Dashboard, Users, Analyze, and Data views
2. **Explore 3D visualization**: Drag to rotate, hover to inspect individual comments
3. **Search users**: Use the search bar in the Users tab to find specific commenters
4. **Analyze text**: Enter custom text in the Analyze tab to test toxicity detection
5. **View data pipeline**: Check the Data tab for processing logs and dataset information

## Dataset Structure

The CSV file (`final_merged_data_nlp.csv`) contains:

- **181,076 rows** (comments)
- **60+ columns** (features)
- **~50MB** file size
- **Political discourse** from social media

### Sample Comment Structure

```json
{
  "id": 1,
  "commenter_id": "user123",
  "comment_content": "Sample comment text...",
  "sentiment_label": "negative",
  "sentiment_prob": 0.87,
  "hate_label": "NOT-HATE",
  "hate_prob": 0.12,
  "offensive_label": "not-offensive",
  "offensive_prob": 0.23,
  "tagged_grok": false,
  "aggression": 0.45,
  "rage": 0.23,
  "politics": 0.89
}
```

## Performance Optimizations

- **CSV Sampling**: Loads first 50,000 rows for frontend display
- **3D Rendering**: Displays max 5,000 points with efficient BufferGeometry
- **Lazy Loading**: Data fetched on client-side after initial render
- **React Optimization**: useMemo, useCallback for expensive computations
- **StrictMode Handling**: Proper cleanup to prevent duplicate canvas rendering

## API Endpoints

### `GET /api/data`

Returns processed comment data and statistics.

**Response:**

```json
{
  "comments": [
    {
      "id": 1,
      "commenter_id": "user",
      "comment_content": "...",
      "sentiment_label": "negative",
      "hate_prob": 0.12,
      ...
    }
  ],
  "stats": {
    "totalComments": 50000,
    "uniqueUsers": 1234,
    "hateComments": 567,
    "offensiveComments": 890,
    "grokResponses": 255,
    "sentimentBreakdown": {
      "positive": 12000,
      "negative": 25000,
      "neutral": 13000
    },
    "topUsers": [
      { "user": "grok", "count": 255 },
      ...
    ]
  }
}
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- WebGL 2.0 required for 3D visualization

## Future Enhancements

- [ ] Real-time data streaming
- [ ] Advanced filtering and search
- [ ] Export functionality (CSV, JSON, PDF reports)
- [ ] Machine learning model integration
- [ ] User timeline analysis
- [ ] Network graph visualization
- [ ] Comparative analysis tools
- [ ] Custom date range filtering
- [ ] API key integration for live Twitter/X data
- [ ] Multi-platform support (Reddit, Facebook, etc.)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## Acknowledgments

- NLP models: HuggingFace Transformers
- Data processing: Python pandas, scikit-learn
- Visualization: Three.js community
- UI inspiration: Modern data dashboard designs

---

**Built with ❤️ for better online discourse analysis**
