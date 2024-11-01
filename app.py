import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import pinterest
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

class MemeFinder:
    def __init__(self):
        # Load CLIP model for image-text matching
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Pinterest API setup 
        self.pinterest_client = pinterest.Pinterest(
            # Replace with your Pinterest API credentials
            access_token=os.getenv('PINTEREST_ACCESS_TOKEN')
        )
        
        # Local meme database (can be expanded)
        self.meme_database = [
            {
                "id": 1, 
                "url": "https://example.com/meme1.jpg", 
                "keywords": ["funny", "cat", "internet"]
            },
            # More memes can be added here
        ]

    def search_memes(self, query):
        """
        Search memes using multiple strategies:
        1. Local database keyword matching
        2. Pinterest API search
        3. CLIP semantic search
        """
        # 1. Local database search
        local_results = [
            meme for meme in self.meme_database 
            if any(query.lower() in keyword.lower() for keyword in meme['keywords'])
        ]
        
        # 2. Pinterest API search
        pinterest_results = self._search_pinterest(query)
        
        # 3. CLIP semantic search
        semantic_results = self._semantic_search(query)
        
        # Combine and deduplicate results
        combined_results = local_results + pinterest_results + semantic_results
        return list({result['url']: result for result in combined_results}.values())

    def _search_pinterest(self, query):
        """Search memes on Pinterest"""
        try:
            pins = self.pinterest_client.search(query, limit=10)
            return [
                {
                    "url": pin.image_large_url, 
                    "keywords": [query]
                } for pin in pins
            ]
        except Exception as e:
            print(f"Pinterest search error: {e}")
            return []

    def _semantic_search(self, query):
        """Use CLIP for semantic image-text matching"""
        try:
            # Encode text query
            text_inputs = self.processor(
                text=[query], 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            text_features = self.model.get_text_features(**text_inputs)
            
            # Compare with meme database images
            results = []
            for meme in self.meme_database:
                # Download and process image
                image = Image.open(requests.get(meme['url'], stream=True).raw)
                image_inputs = self.processor(
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)
                
                image_features = self.model.get_image_features(**image_inputs)
                
                # Compute similarity
                similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
                
                if similarity.item() > 0.5:  # Threshold for semantic match
                    results.append({
                        "url": meme['url'],
                        "similarity": similarity.item()
                    })
            
            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

app = Flask(__name__)
CORS(app)
meme_finder = MemeFinder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    results = meme_finder.search_memes(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
