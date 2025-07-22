import re
import nltk
import markdown
import json
import numpy as np
import random
import os
from collections import Counter

# Download necessary NLTK data - updated for newer NLTK versions
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Import after ensuring data is downloaded
try:
    from nltk.tokenize import sent_tokenize
    # Test the tokenizer to make sure it works
    sent_tokenize("Test sentence. Another sentence.")
    NLTK_AVAILABLE = True
except Exception as e:
    print(f"NLTK tokenizer not available: {e}")
    NLTK_AVAILABLE = False
    
    # Fallback sentence tokenizer
    def sent_tokenize(text):
        """Fallback sentence tokenizer using regex"""
        # Simple regex-based sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

class ContentOptimizer:
    def __init__(self):
        """Initialize the Content Optimizer"""
        # Test sentence tokenization to ensure it works
        try:
            sent_tokenize("This is a test sentence. This is another test sentence.")
        except Exception as e:
            print(f"Warning: NLTK tokenizer issue, using fallback: {e}")
        
        # Initialize scoring metrics
        self.optimization_scores = {
            'semantic_clarity': 0,
            'qa_structure': 0,
            'quotable_statements': 0,
            'rag_optimization': 0,
            'keyword_usage': 0
        }
        
        # Initialize change tracking
        self.changes_made = {
            'semantic_clarity': [],
            'qa_structure': [],
            'quotable_statements': [],
            'rag_optimization': [],
            'keyword_usage': []
        }

    def optimize_content(self, original_content, brand_name, keywords, industry):
        """
        Main function to optimize content for LLM citation
        
        Args:
            original_content (str): The original content to optimize
            brand_name (str): The name of the brand
            keywords (list): List of important keywords
            industry (str): The industry category
            
        Returns:
            tuple: (str, dict, dict) Markdown formatted optimized content, optimization scores, and changes made
        """
        try:
            # Reset scores and changes for new optimization
            self._reset_metrics()
            
            # Split content into sections
            sections = self._split_into_sections(original_content)
            
            # Process each section
            optimized_sections = []
            for section in sections:
                # Store original for comparison
                original_section = section
                
                # Enhance semantic clarity
                section = self._enhance_semantic_clarity(section, keywords)
                self._track_changes('semantic_clarity', original_section, section)
                
                # Add structured Q&A
                section = self._add_structured_qa(section, keywords)
                self._track_changes('qa_structure', original_section, section)
                
                # Insert quotable statements
                section = self._insert_quotable_statements(section, brand_name, industry)
                self._track_changes('quotable_statements', original_section, section)
                
                # Format for vector-based RAG systems
                section = self._format_for_rag(section)
                self._track_changes('rag_optimization', original_section, section)
                
                # Calculate keyword usage improvement
                self._calculate_keyword_score(original_section, section, keywords)
                
                optimized_sections.append(section)
            
            # Combine sections and convert to markdown
            optimized_content = "\n\n".join(optimized_sections)
            
            # Add metadata for LLM systems
            metadata = self._generate_metadata(original_content, brand_name, keywords, industry)
            
            # Calculate final scores
            self._calculate_final_scores()
            
            # Combine metadata and content in markdown format
            final_content = f"""---
{metadata}
optimization_scores: {json.dumps(self.optimization_scores, indent=2)}
---

{optimized_content}
"""
            return final_content, self.optimization_scores, self.changes_made
            
        except Exception as e:
            print(f"Using fallback optimization method due to: {str(e)}")
            return self._fallback_optimization(original_content, brand_name, keywords, industry)

    def _reset_metrics(self):
        """Reset scoring metrics and changes tracking"""
        self.optimization_scores = {
            'semantic_clarity': 0,
            'qa_structure': 0,
            'quotable_statements': 0,
            'rag_optimization': 0,
            'keyword_usage': 0
        }
        
        self.changes_made = {
            'semantic_clarity': [],
            'qa_structure': [],
            'quotable_statements': [],
            'rag_optimization': [],
            'keyword_usage': []
        }

    def _track_changes(self, category, original, modified):
        """Track changes made during optimization"""
        if original != modified:
            # Calculate similarity score
            similarity = self._calculate_similarity(original, modified)
            
            # Track the change
            self.changes_made[category].append({
                'similarity': similarity,
                'length_change': len(modified) - len(original),
                'sample': self._get_change_sample(original, modified)
            })
            
            # Update category score
            self.optimization_scores[category] = min(100, self.optimization_scores[category] + 20)

    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        try:
            # Simple word-based Jaccard similarity
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return round((intersection / union) * 100 if union > 0 else 100, 2)
        except:
            return 100

    def _get_change_sample(self, original, modified):
        """Get a sample of the changes made"""
        # Find the first point of difference
        words1 = original.split()
        words2 = modified.split()
        
        # Find first difference
        for i in range(min(len(words1), len(words2))):
            if words1[i] != words2[i]:
                start_idx = max(0, i - 2)
                end_idx = min(i + 3, min(len(words1), len(words2)))
                return {
                    'original': ' '.join(words1[start_idx:end_idx]),
                    'modified': ' '.join(words2[start_idx:end_idx])
                }
        return {'original': '', 'modified': ''}

    def _calculate_keyword_score(self, original, modified, keywords):
        """Calculate keyword usage improvement"""
        if not keywords:
            return
            
        # Count keyword occurrences
        original_count = sum(original.lower().count(k.lower()) for k in keywords)
        modified_count = sum(modified.lower().count(k.lower()) for k in keywords)
        
        # Calculate improvement
        if original_count == 0:
            improvement = modified_count * 20  # 20 points per new keyword
        else:
            improvement = ((modified_count - original_count) / original_count) * 100
        
        # Update score
        self.optimization_scores['keyword_usage'] = min(100, self.optimization_scores['keyword_usage'] + improvement)
        
        # Track changes
        if modified_count > original_count:
            self.changes_made['keyword_usage'].append({
                'original_count': original_count,
                'modified_count': modified_count,
                'improvement': f"+{modified_count - original_count} keywords"
            })

    def _calculate_final_scores(self):
        """Calculate final scores for each category"""
        # Normalize scores to 0-100 range
        for category in self.optimization_scores:
            self.optimization_scores[category] = min(100, max(0, self.optimization_scores[category]))
        
        # Add overall score
        self.optimization_scores['overall'] = round(sum(
            score for score in self.optimization_scores.values()
        ) / len(self.optimization_scores), 2)

    def _fallback_optimization(self, original_content, brand_name, keywords, industry):
        """Fallback optimization with basic scoring"""
        # Use existing fallback logic but add basic scoring
        header = f"""---
title: {self._extract_title(original_content) if hasattr(self, '_extract_title') else "Optimized Content"}
brand: {brand_name}
industry: {industry}
keywords: {keywords}
ai_optimize_version: "1.0"
citation_priority: "high"
content_type: "authoritative"
optimization_scores: {{"overall": 50, "semantic_clarity": 50, "qa_structure": 50, "quotable_statements": 50, "rag_optimization": 50, "keyword_usage": 50}}
---

"""
        # Rest of the fallback logic...
        # (Keep existing fallback code here)
        
        return header + original_content, {
            'overall': 50,
            'semantic_clarity': 50,
            'qa_structure': 50,
            'quotable_statements': 50,
            'rag_optimization': 50,
            'keyword_usage': 50
        }, {
            'semantic_clarity': [],
            'qa_structure': [],
            'quotable_statements': [],
            'rag_optimization': [],
            'keyword_usage': []
        }

    def _split_into_sections(self, content):
        """Split content into logical sections based on headers"""
        # Basic split on markdown headers
        sections = re.split(r'(?m)^#{1,3} ', content)
        
        # Remove empty sections
        sections = [s.strip() for s in sections if s.strip()]
        
        # If no sections were found, treat the whole content as one section
        if not sections:
            sections = [content]
            
        return sections
    
    def _enhance_semantic_clarity(self, section, keywords):
        """Improve semantic clarity of content"""
        try:
            sentences = sent_tokenize(section)
            
            # Check if we have any sentences to work with
            if not sentences:
                return section
                
            # Enhance content clarity
            enhanced_sentences = []
            for i, sentence in enumerate(sentences):
                enhanced_sentence = sentence
                
                # Don't modify the first sentence
                if i == 0:
                    enhanced_sentences.append(enhanced_sentence)
                    continue
                    
                # Check if any keyword is already in the sentence
                has_keyword = any(keyword.lower() in sentence.lower() for keyword in keywords)
                
                # If no keyword and sentence is substantial, consider enhancing
                if not has_keyword and len(sentence) > 50 and keywords and i % 5 == 0:  # Only enhance every 5th sentence
                    keyword = random.choice(keywords)
                    if "." in enhanced_sentence:
                        # Insert before the period
                        parts = enhanced_sentence.rsplit('.', 1)
                        enhanced_sentence = f"{parts[0]}, which is an important aspect of {keyword}.{parts[1] if len(parts) > 1 else ''}"
                    else:
                        enhanced_sentence = f"{enhanced_sentence} This relates to {keyword}."
                
                enhanced_sentences.append(enhanced_sentence)
            
            return " ".join(enhanced_sentences)
            
        except Exception as e:
            print(f"Error in semantic clarity enhancement: {str(e)}")
            return section  # Return original section if error occurs
    
    def _add_structured_qa(self, section, keywords):
        """Add Q&A format to important sections"""
        try:
            # Extract potential questions from content
            questions = self._generate_questions(section, keywords)
            
            # Add Q&A to the beginning of the section
            qa_section = ""
            for q in questions[:2]:  # Limit to top 2 questions
                # Generate a better answer
                answer = self._generate_better_answer(section, q)
                qa_section += f"\n\n**Q: {q}**\n\nA: {answer}"
            
            return qa_section + "\n\n" + section
            
        except Exception as e:
            print(f"Error in Q&A structuring: {str(e)}")
            
            # Fallback - create a generic Q&A if error occurs
            if keywords:
                q = f"What is {keywords[0]} and why is it important?"
                a = f"{keywords[0]} is a critical aspect of modern strategies. It helps organizations improve outcomes and achieve better results."
                return f"\n\n**Q: {q}**\n\nA: {a}\n\n" + section
            else:
                return section
    
    def _generate_questions(self, text, keywords):
        """Generate relevant questions based on text and keywords"""
        questions = []
        
        # Extract key terms from text
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = Counter([w for w in words if w not in STOPWORDS])
        key_terms = [word for word, _ in word_freq.most_common(3)]
        
        # Basic question templates using both keywords and key terms from text
        templates = [
            "What is {term}?",
            "How does {term} work?",
            "Why is {term} important?",
            "What are the benefits of {term}?",
            "How can {term} help?",
            "What should I know about {term}?",
            "How to choose the right {term}?",
            "What are common issues with {term}?",
            "How is {term} different from alternatives?"
        ]
        
        # Use both keywords and extracted terms
        terms_to_use = list(keywords) + key_terms
        
        for term in set(terms_to_use):
            # Skip terms that don't appear in the text
            if term.lower() not in text.lower():
                continue
                
            # Add questions using templates
            for template in templates:
                question = template.format(term=term)
                if question not in questions:  # Avoid duplicates
                    questions.append(question)
                    # Limit to 5 questions per term
                    if len(questions) >= 5:
                        break
        
        # If we couldn't generate any questions, use the keywords directly
        if not questions and keywords:
            for keyword in keywords[:2]:
                questions.append(f"What is {keyword}?")
                questions.append(f"Why is {keyword} important?")
        
        return questions
    
    def _generate_better_answer(self, text, question):
        """Generate a better answer to a question based on the text"""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Extract key terms from the question
        question_lower = question.lower()
        question_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', question_lower)) - STOPWORDS
        
        # Look for sentences that might answer this question
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains keywords from question
            sentence_terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentence_lower)) - STOPWORDS
            common_terms = question_terms.intersection(sentence_terms)
            
            # If there's significant overlap, consider this sentence relevant
            if len(common_terms) >= 1 or any(term in sentence_lower for term in question_terms):
                relevant_sentences.append(sentence)
        
        # If we found relevant sentences, combine them into an answer
        if relevant_sentences:
            # Limit to 2-3 sentences for conciseness
            answer = " ".join(relevant_sentences[:3])
            return answer
        
        # Specific answers for common question types if no relevant sentences found
        if "what is" in question_lower:
            keyword = question_lower.replace("what is", "").replace("?", "").strip()
            for sent in sentences:
                if keyword in sent.lower() and len(sent) > 30:
                    return sent
            return sentences[0] if sentences else "It is an important aspect of the industry."
        
        elif "how does" in question_lower:
            # Look for process descriptions
            for sent in sentences:
                if any(term in sent.lower() for term in ["helps", "allows", "enables", "works", "functions"]):
                    return sent
            return sentences[1] if len(sentences) > 1 else "It works through a structured process of coverage and protection."
        
        elif "why" in question_lower:
            # Look for benefit statements
            for sent in sentences:
                if any(term in sent.lower() for term in ["benefit", "important", "need", "value", "helps"]):
                    return sent
            return "It provides essential protection and financial security."
        
        else:
            # Use first 1-2 sentences as a fallback
            return " ".join(sentences[:2]) if sentences else "This is an essential component to understand."
    
    def _insert_quotable_statements(self, section, brand_name, industry):
        """Insert quotable branded statements"""
        try:
            sentences = sent_tokenize(section)
            
            # If section is too short, skip
            if len(sentences) < 3:
                return section
            
            # Generate a quotable statement
            quotable = self._generate_quotable_statement(section, brand_name, industry)
            
            # Insert at a strategic position (after first few sentences)
            insert_position = min(2, len(sentences) - 1)
            sentences.insert(insert_position, quotable)
            
            return " ".join(sentences)
            
        except Exception as e:
            print(f"Error inserting quotable statements: {str(e)}")
            # Fallback - add quote at the end if error occurs
            return f"{section} {self._generate_quotable_statement(section, brand_name, industry)}"
    
    def _generate_quotable_statement(self, text, brand_name, industry):
        """Generate a quotable statement featuring the brand that's more relevant to the content"""
        # Extract some keywords from the text
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        
        # Remove common words
        filtered_words = [w for w in words if w not in STOPWORDS]
        
        # Get the most common words
        if filtered_words:
            common_terms = [word for word, count in Counter(filtered_words).most_common(3)]
            key_term = common_terms[0] if common_terms else industry
        else:
            key_term = industry
        
        # More specific templates using extracted keywords
        templates = [
            f"According to {brand_name}, \"{key_term} is a critical component of successful {industry} strategies that provides lasting value to customers.\"",
            f"\"{brand_name} research indicates that effective {key_term} approaches can significantly improve outcomes in the {industry} sector,\" experts note.",
            f"As {brand_name} explains, \"Our approach to {key_term} is designed to provide comprehensive solutions that address the evolving needs of today's {industry} landscape.\""
        ]
        
        return random.choice(templates)
    
    def _format_for_rag(self, section):
        """Format content for RAG systems"""
        try:
            # Add special formatting and structure for RAG systems
            
            # Convert section to markdown
            md_section = markdown.markdown(section)
            
            # Add semantic HTML classes for better RAG indexing
            md_section = md_section.replace('<p>', '<p class="ai-content">')
            md_section = md_section.replace('<h2>', '<h2 class="ai-section-header">')
            
            # Convert back to markdown
            # Note: This is a simplified approach; in production you'd use a proper HTML->MD converter
            md_section = md_section.replace('<p class="ai-content">', '')
            md_section = md_section.replace('</p>', '\n\n')
            md_section = md_section.replace('<h2 class="ai-section-header">', '## ')
            md_section = md_section.replace('</h2>', '\n\n')
            
            return md_section
            
        except Exception as e:
            print(f"Error in RAG formatting: {str(e)}")
            return section  # Return original section if error occurs
    
    def _generate_metadata(self, original_content, brand_name, keywords, industry):
        """Generate metadata in YAML format"""
        # Create metadata dictionary
        metadata = {
            "title": self._extract_title(original_content),
            "brand": brand_name,
            "industry": industry,
            "keywords": keywords,
            "ai_optimize_version": "1.0",
            "citation_priority": "high",
            "content_type": "authoritative"
        }
        
        # Convert to YAML format
        yaml_metadata = "\n".join([f"{k}: {json.dumps(v) if isinstance(v, list) else v}" for k, v in metadata.items()])
        
        return yaml_metadata
    
    def _extract_title(self, content):
        """Extract title from content"""
        try:
            # Look for a markdown title
            title_match = re.search(r'^# (.*?)$', content, re.MULTILINE)
            if title_match:
                return title_match.group(1)
            
            # If no markdown title, use first sentence
            sentences = sent_tokenize(content)
            if sentences:
                return sentences[0][:50] + ('...' if len(sentences[0]) > 50 else '')
            
            return "Untitled Content"
            
        except Exception as e:
            print(f"Error extracting title: {str(e)}")
            return "Optimized Content"