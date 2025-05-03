import pandas as pd
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
import json
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
from .models import Assessment

load_dotenv()

class SHLRecommender:
    def __init__(self, data_path: str = "data/assessments.csv"):
        """Initializes the recommender system with enough sarcasm to numb the pain."""
        try:
            self.df = pd.read_csv(data_path)
            print(f"\nLoaded {len(self.df)} assessments. Let the disappointment begin.")
            
            # Initialize models with appropriate error handling
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.llm = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            # Pre-cache common queries
            self._warmup_cache()
            
        except Exception as e:
            raise RuntimeError(f"Initialization failed. Reason: {e}. Solution: Try harder.") from e

    def _warmup_cache(self):
        """Pre-cache common queries because users are predictable."""
        common_queries = [
            "Java developer test",
            "Manager personality assessment",
            "Technical test under 30 minutes"
        ]
        for q in common_queries:
            self.parse_query_with_gemini(q)

    @lru_cache(maxsize=1000)
    def parse_query_with_gemini(self, query: str) -> dict:
        """
        Uses Gemini to extract requirements from user's word salad.
        Returns: {
            'duration_minutes': int | None,
            'behavioral': bool,
            'technical': bool,
            'job_role': str,
            'cleaned_query': str,
            'source': 'gemini' | 'fallback'
        }
        """
        prompt = f"""
        Extract JSON with these fields from this assessment request:
        - duration_minutes: extract time or null
        - behavioral: true for culture/personality/team skills
        - technical: true for coding/technical skills
        - job_role: main job title
        - cleaned_query: simplified search query
        
        Input: "{query}"
        
        Respond ONLY with valid JSON like:
        {{
            "duration_minutes": 60,
            "behavioral": true,
            "technical": false,
            "job_role": "COO",
            "cleaned_query": "COO cultural fit assessment"
        }}
        """
        
        try:
            response = self.llm.generate_content(prompt)
            json_str = response.text.strip().replace('```json', '').replace('```', '')
            result = json.loads(json_str)
            result['source'] = 'gemini'
            return result
        except Exception as e:
            print(f"Gemini parsing failed. Falling back to regex. Error: {e}")
            return {**self._regex_parse_fallback(query), 'source': 'fallback'}

    def _regex_parse_fallback(self, query: str) -> dict:
        """Poor man's parser for when Gemini fails us."""
        # Duration extraction (handles most common formats)
        duration = None
        if match := re.search(r'(\d+)\s*(min|minutes?|m|hour|hrs?)\b', query, re.IGNORECASE):
            duration = int(match.group(1))
            if match.group(2)[0].lower() in ('h', 'r'):  # hours
                duration *= 60

        # Behavioral detection
        behavioral = bool(re.search(
            r'culture|cultural|fit|behavior|personality|collaborat|team|soft skill|communicat',
            query, re.IGNORECASE
        ))

        # Technical detection
        technical = bool(re.search(
            r'technical|code|coding|programming|java|python|sql|developer|engineer|tech',
            query, re.IGNORECASE
        ))

        # Job role extraction
        job_role = ""
        if match := re.search(r'\b(for|hire|looking)\s+(a|an)?\s*(.+?)\s+(position|role|job|test)', query, re.IGNORECASE):
            job_role = match.group(3)

        # Query cleaning
        cleaned = re.sub(r'\b(assess|test|exam|evaluat|looking|for)\w*\b', '', query, flags=re.IGNORECASE)
        cleaned = ' '.join([word for word in cleaned.split() if len(word) > 2])  # Remove small words

        return {
            'duration_minutes': duration,
            'behavioral': behavioral,
            'technical': technical,
            'job_role': job_role,
            'cleaned_query': cleaned.strip() or query  # Fallback to original if empty
        }

    def _enhance_query(self, parsed_query: dict) -> str:
        """Injects steroids into the query."""
        enhancements = []
        if parsed_query['behavioral']:
            enhancements.extend(['behavioral', 'teamwork', 'cultural fit'])
        if parsed_query['technical']:
            enhancements.extend(['technical', 'coding', 'programming'])
        if parsed_query['job_role']:
            enhancements.append(parsed_query['job_role'])

        return f"{parsed_query['cleaned_query']} {' '.join(enhancements)}".strip()

    def _semantic_search(self, enhanced_query: str, top_k: int = 20, threshold: float = 0.3) -> pd.DataFrame:
        """Finds assessments that don't completely suck."""
        query_embedding = self.model.encode(enhanced_query, convert_to_tensor=True)
        
        combined_texts = self.df.apply(
            lambda row: f"{row['Name']}. {row['Description']}. Types: {row['Test Types']}. Level: {row['Job Levels']}. Duration: {row['Assessment Length (Minutes)']}min", 
            axis=1
        ).tolist()
        
        assessment_embeddings = self.model.encode(combined_texts, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, assessment_embeddings)[0]
        
        # Filter and sort results
        scored_indices = [(i, float(score)) for i, score in enumerate(similarities) if score > threshold]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_indices:
            print("\nSemantic search found nothing. Maybe try a real query?")
            return pd.DataFrame()
            
        return self.df.iloc[[i for i, _ in scored_indices[:top_k]]]

    def _filter_assessments(self, df: pd.DataFrame, parsed_query: dict) -> pd.DataFrame:
        """Applies all filters with some common sense."""
        if df.empty:
            return df
            
        # Duration filter with tolerance
        if parsed_query['duration_minutes']:
            tolerance = min(30, parsed_query['duration_minutes'] * 0.5)  # Dynamic tolerance
            df = df[
                (df['Assessment Length (Minutes)'] >= max(0, parsed_query['duration_minutes'] - tolerance)) & 
                (df['Assessment Length (Minutes)'] <= parsed_query['duration_minutes'] + tolerance)
            ]
        
        # Type filters
        type_filters = []
        if parsed_query['behavioral']:
            type_filters.append(r'B|P')  # Behavioral or Personality
        if parsed_query['technical']:
            type_filters.append(r'K|T')  # Knowledge or Technical
            
        if type_filters:
            df = df[df['Test Types'].str.contains('|'.join(type_filters), regex=True, case=False)]
            
        return df

    def _rerank_with_llm(self, query: str, assessments: pd.DataFrame) -> pd.DataFrame:
        """The final insult - let an LLM fix our mistakes."""
        if len(assessments) <= 1:  # No point reranking 0 or 1 items
            return assessments
            
        prompt = f"""
        As an HR expert, rank these assessments (1=best) for this query:
        "{query}"
        
        Considerations:
        - Match job level and requirements
        - Prefer shorter durations
        - Prioritize behavioral/cultural fit when mentioned
        
        {assessments[['Name', 'Description', 'Test Types', 'Job Levels', 'Assessment Length (Minutes)']].to_markdown()}
        
        Return ONLY a comma-separated list of indices in order (e.g., "3,1,2").
        """
        
        try:
            response = self.llm.generate_content(prompt)
            indices = [int(i.strip()) for i in response.text.split(',') if i.strip().isdigit()]
            valid_indices = [i for i in indices if i < len(assessments)]
            return assessments.iloc[valid_indices] if valid_indices else assessments
        except Exception as e:
            print(f"\nLLM ranking failed. Using default order. Error: {e}")
            return assessments

    def recommend(self, raw_query: str) -> List[Assessment]:
        """The main attraction. Manages expectations while delivering mediocrity."""
        print(f"\nProcessing query: '{raw_query}'")
        
        # Step 1: Parse the query (with Gemini or fallback)
        parsed_query = self.parse_query_with_gemini(raw_query)
        print(f"Parsed: {parsed_query}")
        
        # Step 2: Enhance the query
        enhanced_query = self._enhance_query(parsed_query)
        print(f"Enhanced query: '{enhanced_query}'")
        
        # Step 3: Semantic search
        semantic_results = self._semantic_search(enhanced_query)
        if semantic_results.empty:
            print("No results found. Maybe try LinkedIn instead?")
            return []
        
        # Step 4: Apply filters
        filtered = self._filter_assessments(semantic_results, parsed_query)
        if filtered.empty:
            print("Filters too strict. Relax your requirements.")
            return []
        
        # Step 5: LLM re-ranking
        reranked = self._rerank_with_llm(enhanced_query, filtered.head(20))
        
        # Step 6: Format results (with duplicate removal)
        seen = set()
        recommendations = []
        for _, row in reranked.iterrows():
            if row['Name'] not in seen:
                seen.add(row['Name'])
                recommendations.append(
                    Assessment(
                        name=row['Name'],
                        url=row['URL'],
                        remote_support=row['Remote'],
                        adaptive_support=row['Adaptive'],
                        duration=row['Assessment Length (Minutes)'],
                        test_type=row['Test Types']
                    )
                )
            if len(recommendations) >= 10:
                break
                
        print(f"\nReturning {len(recommendations)} recommendations. Lower your expectations.")
        return recommendations