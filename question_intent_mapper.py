import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QuestionIntentMapper:
    def __init__(self):
        """Initialize the Question & Intent Mapper"""
        # Force CPU device for deployment compatibility
        import torch
        device = 'cpu'  # Force CPU for cloud deployment
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Question templates for local generation
        self.question_templates = [
            "What are the best {topic} tools for {industry}?",
            "How do I choose the right {topic} solution for {industry}?", 
            "What are the top {topic} platforms in {industry}?",
            "What {topic} features should I look for in {industry}?",
            "How does {topic} work in the {industry} industry?",
            "What are the benefits of {topic} for {industry}?",
            "What are common {topic} challenges in {industry}?",
            "How much does {topic} cost for {industry} businesses?",
            "What are {topic} best practices for {industry}?",
            "How to implement {topic} in {industry}?",
            "What are {topic} alternatives for {industry}?",
            "Why is {topic} important for {industry}?",
            "How long does it take to see results from {topic}?",
            "What are the latest trends in {topic} for {industry}?",
            "How to measure {topic} success in {industry}?",
            "Can small businesses benefit from {topic}?",
            "How is {topic} different from traditional approaches?",
            "What industries benefit most from {topic}?",
            "What are the ROI expectations for {topic} in {industry}?",
            "How to get started with {topic} in {industry}?"
        ]
    
    def generate_questions(self, topics, industry, brand_name, competitors, product_features, num_questions=100):
        """
        Generate likely AI prompts users would ask related to the topics
        
        Args:
            topics (list): Main topics related to the brand/product
            industry (str): Industry category
            brand_name (str): Brand name
            competitors (list): List of competitor names
            product_features (list): Key features of the product
            num_questions (int): Number of questions to generate
            
        Returns:
            list: Generated questions
        """
        # Fallback to template-based generation
        return self._generate_questions_with_templates(topics, industry, brand_name, competitors, product_features, num_questions)
    
    def _generate_questions_with_templates(self, topics, industry, brand_name, competitors, product_features, num_questions):
        """Generate questions using templates"""
        all_questions = []
        
        # Generate topic-based questions from templates
        for topic in topics:
            for template in self.question_templates:
                if "{related_topic}" in template:
                    # For comparison questions, use other topics or competitors
                    other_topics = [t for t in topics if t != topic]
                    if other_topics:
                        related_topic = np.random.choice(other_topics)
                        question = template.format(topic=topic, related_topic=related_topic, industry=industry)
                        all_questions.append(question)
                    
                    # Also create competitor comparison questions
                    for competitor in competitors:
                        question = f"How does {brand_name} compare to {competitor} for {topic}?"
                        all_questions.append(question)
                else:
                    # Regular question templates
                    question = template.format(topic=topic, industry=industry)
                    all_questions.append(question)
        
        # Generate feature-specific questions
        for feature in product_features:
            all_questions.append(f"How does {feature} work in {brand_name}?")
            all_questions.append(f"What are the benefits of {feature}?")
            all_questions.append(f"How is {feature} in {brand_name} different from competitors?")
        
        # Generate brand-specific questions
        all_questions.append(f"What is {brand_name}?")
        all_questions.append(f"How much does {brand_name} cost?")
        all_questions.append(f"Is {brand_name} worth it?")
        all_questions.append(f"What are alternatives to {brand_name}?")
        
        # Shuffle and limit to requested number
        np.random.shuffle(all_questions)
        return all_questions[:num_questions]
    
    def cluster_questions(self, questions, n_clusters=None, min_cluster_size=5):
        """
        Cluster questions into intent groups
        
        Args:
            questions (list): List of questions to cluster
            n_clusters (int): Number of clusters for KMeans (if None, use HDBSCAN)
            min_cluster_size (int): Minimum cluster size for HDBSCAN
            
        Returns:
            dict: Clustered questions with intent labels
        """
        # Embed questions
        print("Embedding questions...")
        embeddings = self.embedding_model.encode(questions)
        
        # Choose clustering method
        if n_clusters:
            # Use KMeans for fixed number of clusters
            print(f"Clustering with KMeans (k={n_clusters})...")
            clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
            labels = clustering.labels_
        else:
            # Use UMAP + HDBSCAN for automatic clustering
            print("Reducing dimensions with UMAP...")
            umap_embeddings = umap.UMAP(
                n_neighbors=15, 
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            ).fit_transform(embeddings)
            
            print(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='euclidean',
                gen_min_span_tree=True,
                prediction_data=True
            ).fit(umap_embeddings)
            
            labels = clusterer.labels_
        
        # Organize questions by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(questions[i])
        
        # Generate intent labels for each cluster
        result = {}
        for label, cluster_questions in clusters.items():
            if label == -1:
                # Skip noise cluster from HDBSCAN
                continue
                
            # Generate an intent label
            intent_label = self._generate_intent_label(cluster_questions)
            
            # Store cluster with intent label
            result[intent_label] = cluster_questions
        
        return result
    
    def _generate_intent_label(self, questions):
        """Generate an intent label for a cluster of questions"""
        # Extract common keywords
        all_text = " ".join(questions)
        
        # Remove question words and common stop words
        cleaned_text = re.sub(r'\b(what|how|why|is|are|do|does|can|should|would|will|the|a|an|in|for|to|of|with|on|at)\b', 
                             '', all_text, flags=re.IGNORECASE)
        
        # Extract most common words
        words = cleaned_text.lower().split()
        word_freq = pd.Series(words).value_counts().head(3)
        
        # Create intent label
        if len(word_freq) > 0:
            intent_words = word_freq.index.tolist()
            intent_label = "_".join(intent_words)
            return intent_label
        else:
            # Fallback to using the first question's first few words
            first_q = questions[0]
            words = first_q.split()[:3]
            return "_".join(words).lower().replace("?", "").replace(",", "")
    
    def visualize_clusters(self, questions, clusters):
        """
        Visualize question clusters
        
        Args:
            questions (list): List of all questions
            clusters (dict): Clustered questions with intent labels
            
        Returns:
            None (displays plot)
        """
        # Embed questions
        embeddings = self.embedding_model.encode(questions)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'question': questions,
        })
        
        # Add cluster labels
        df['cluster'] = 'unclustered'
        for intent, cluster_questions in clusters.items():
            for q in cluster_questions:
                df.loc[df['question'] == q, 'cluster'] = intent
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis')
        plt.title('Question Clusters by Intent')
        plt.xlabel('PCA Dimension 1')
        plt.ylabel('PCA Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('question_clusters.png')
        plt.show()
    
    def export_results(self, clusters, output_file='question_intents.json'):
        """
        Export clustered questions to JSON
        
        Args:
            clusters (dict): Clustered questions with intent labels
            output_file (str): Output filename
            
        Returns:
            None
        """
        # Format results
        results = {
            "intents": [],
            "metadata": {
                "total_questions": sum(len(qs) for qs in clusters.values()),
                "total_intents": len(clusters),
                "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Add each intent with its questions
        for intent, questions in clusters.items():
            results["intents"].append({
                "intent": intent,
                "questions": questions,
                "count": len(questions)
            })
        
        # Sort intents by question count
        results["intents"] = sorted(results["intents"], key=lambda x: x["count"], reverse=True)
        
        # Export to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Exported {results['metadata']['total_questions']} questions across {results['metadata']['total_intents']} intents to {output_file}")

# Example usage
if __name__ == "__main__":
    mapper = QuestionIntentMapper()
    
    # Example data
    topics = ["AI content optimization", "LLM SEO", "semantic search", "content discoverability"]
    industry = "digital marketing"
    brand_name = "AIO Search"
    competitors = ["Clearscope", "MarketMuse", "Surfer SEO"]
    product_features = ["Content Optimizer", "AI Visibility Checker", "Question Mapper", "AI Sitemap"]
    
    # Generate questions
    questions = mapper.generate_questions(
        topics=topics,
        industry=industry,
        brand_name=brand_name,
        competitors=competitors,
        product_features=product_features,
        num_questions=50  # Small number for demo purposes
    )
    
    # Cluster questions
    clusters = mapper.cluster_questions(questions, min_cluster_size=3)
    
    # Visualize clusters
    mapper.visualize_clusters(questions, clusters)
    
    # Export results
    mapper.export_results(clusters)