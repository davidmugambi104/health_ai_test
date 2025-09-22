from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import re
from rapidfuzz import process, fuzz
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Constants
MIN_SYMPTOMS_REQUIRED = 1
MAX_SYMPTOMS_ALLOWED = 10
MIN_MATCHED_SYMPTOMS = 1
MIN_CONFIDENCE_THRESHOLD = 10.0
TOP_N_DISEASES = 5
FUZZY_MATCH_THRESHOLD = 60

class SymptomMatcher:
    """Handles symptom normalization, matching, and severity weighting"""
    
    def __init__(self):
        self.symptom_severity = {}
        self.canonical_symptoms = set()
        self.symptom_mapping = {}
        self._load_severity_data()
        
    def _load_severity_data(self) -> None:
        """Load symptom severity data from CSV"""
        try:
            base_dir = Path(__file__).parent
            severity_path = base_dir / "datasets" / "Symptom-severity.csv"
            severity_df = pd.read_csv(severity_path)
            
            for _, row in severity_df.iterrows():
                symptom = self._normalize_name(row['Symptom'])
                self.symptom_severity[symptom] = row['weight']
                self.canonical_symptoms.add(symptom)
                
            logger.info(f"Loaded {len(self.symptom_severity)} symptoms with severity data")
            
        except Exception as e:
            logger.error(f"Error loading symptom severity data: {e}")
            raise
            
    def _normalize_name(self, name: str) -> str:
        """Normalize symptom/disease names to a standard format"""
        if not isinstance(name, str) or pd.isna(name):
            return ""
        
        # Convert to lowercase, remove punctuation, and replace spaces with underscores
        name = name.lower().strip()
        name = re.sub(r'[^a-z0-9\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name
    
    def add_symptom(self, symptom: str) -> str:
        """
        Add a symptom to the canonical list with default severity if not present
        Returns the normalized symptom name
        """
        normalized = self._normalize_name(symptom)
        if normalized and normalized not in self.canonical_symptoms:
            self.canonical_symptoms.add(normalized)
            if normalized not in self.symptom_severity:
                self.symptom_severity[normalized] = 1  # Default severity
                logger.info(f"Added symptom '{normalized}' with default severity 1")
        return normalized
        
    def map_to_canonical(self, symptom: str) -> Optional[str]:
        """
        Map an input symptom to its canonical form using fuzzy matching
        Returns the canonical symptom name if a match is found above threshold
        """
        if not symptom:
            return None
            
        normalized = self._normalize_name(symptom)
        
        # Exact match check
        if normalized in self.canonical_symptoms:
            return normalized
            
        # Check if it's a close match with underscores/spaces
        alt_form = normalized.replace('_', ' ') if '_' in normalized else normalized.replace(' ', '_')
        if alt_form in self.canonical_symptoms:
            return alt_form
            
        # Fuzzy match against known symptoms
        if self.canonical_symptoms:
            match, score, _ = process.extractOne(
                normalized, 
                list(self.canonical_symptoms), 
                scorer=fuzz.token_sort_ratio
            )
            
            if score >= FUZZY_MATCH_THRESHOLD:
                return match
                
        return None
        
    def get_severity(self, symptom: str) -> int:
        """Get severity weight for a symptom, default to 1 if not found"""
        return self.symptom_severity.get(symptom, 1)


class DiseaseDatabase:
    """Manages disease data including symptoms, descriptions, and precautions"""
    
    def __init__(self, symptom_matcher: SymptomMatcher):
        self.symptom_matcher = symptom_matcher
        self.diseases = {}  # disease_name -> disease_data
        self._load_data()
        
    def _load_data(self) -> None:
        """Load and preprocess all disease-related datasets"""
        try:
            base_dir = Path(__file__).parent
            datasets_dir = base_dir / "datasets"
            
            # Load main dataset
            df = pd.read_csv(datasets_dir / "dataset.csv")
            desc_df = pd.read_csv(datasets_dir / "symptom_Description.csv")
            prec_df = pd.read_csv(datasets_dir / "symptom_precaution.csv")
            
            # Preprocess descriptions
            descriptions = {}
            for _, row in desc_df.iterrows():
                disease = self.symptom_matcher._normalize_name(row['Disease'])
                descriptions[disease] = row['Description']
            
            # Preprocess precautions
            precautions = {}
            for _, row in prec_df.iterrows():
                disease = self.symptom_matcher._normalize_name(row['Disease'])
                prec_list = []
                for i in range(1, 5):
                    col_name = f'Precaution_{i}'
                    if col_name in row and pd.notna(row[col_name]):
                        prec_list.append(str(row[col_name]).strip())
                precautions[disease] = prec_list
            
            # First pass: collect all unique symptoms from the dataset
            all_symptoms_set = set()
            for _, row in df.iterrows():
                for col in df.columns[1:]:  # Skip Disease column
                    if pd.notna(row[col]):
                        symptom = str(row[col]).strip()
                        all_symptoms_set.add(symptom)
            
            # Add all symptoms to the matcher
            for symptom in all_symptoms_set:
                self.symptom_matcher.add_symptom(symptom)
            
            # Second pass: process diseases and their symptoms
            for _, row in df.iterrows():
                disease = self.symptom_matcher._normalize_name(row['Disease'])
                
                # Collect all symptoms for this disease
                symptoms = set()
                for col in df.columns[1:]:  # Skip Disease column
                    if pd.notna(row[col]):
                        symptom_str = str(row[col]).strip()
                        canonical_symptom = self.symptom_matcher.map_to_canonical(symptom_str)
                        if canonical_symptom:
                            symptoms.add(canonical_symptom)
                
                # Calculate total possible severity for this disease
                total_severity = sum(self.symptom_matcher.get_severity(s) for s in symptoms)
                
                # Store disease data
                if disease in self.diseases:
                    # Merge symptoms if disease appears multiple times
                    self.diseases[disease]['symptoms'].update(symptoms)
                    # Recalculate total severity
                    self.diseases[disease]['total_severity'] = sum(
                        self.symptom_matcher.get_severity(s) 
                        for s in self.diseases[disease]['symptoms']
                    )
                else:
                    self.diseases[disease] = {
                        'symptoms': symptoms,
                        'total_severity': total_severity,
                        'description': descriptions.get(disease, "No description available"),
                        'precautions': precautions.get(disease, [])
                    }
                    
            logger.info(f"Loaded {len(self.diseases)} diseases with {len(all_symptoms_set)} unique symptoms")
            
            # Log some sample symptoms for debugging
            sample_symptoms = list(all_symptoms_set)[:10]
            logger.info(f"Sample symptoms in dataset: {sample_symptoms}")
            
        except Exception as e:
            logger.error(f"Error loading disease data: {e}")
            raise
    
    def get_all_symptoms(self) -> List[str]:
        """Get all unique canonical symptoms across all diseases"""
        all_symptoms = set()
        for disease_data in self.diseases.values():
            all_symptoms.update(disease_data['symptoms'])
        return sorted(list(all_symptoms))
    
    def find_matching_diseases(self, user_symptoms: List[str]) -> List[Dict]:
        """
        Find diseases that match the provided symptoms, with confidence scoring
        """
        results = []
        user_symptoms_set = set(user_symptoms)
        
        for disease, data in self.diseases.items():
            disease_symptoms = data['symptoms']
            
            # Find matching symptoms
            matched_symptoms = user_symptoms_set.intersection(disease_symptoms)
            
            if len(matched_symptoms) < MIN_MATCHED_SYMPTOMS:
                continue
                
            # Calculate confidence score
            matched_severity = sum(self.symptom_matcher.get_severity(s) for s in matched_symptoms)
            total_severity = data['total_severity']
            
            if total_severity == 0:
                continue
                
            confidence = (matched_severity / total_severity) * 100
            
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                continue
                
            # Find unmatched symptoms from the disease perspective
            unmatched_symptoms = disease_symptoms - user_symptoms_set
            
            results.append({
                'disease': disease.replace('_', ' ').title(),
                'confidence': round(confidence, 2),
                'matched_symptoms': [s.replace('_', ' ').title() for s in matched_symptoms],
                'unmatched_symptoms': [s.replace('_', ' ').title() for s in unmatched_symptoms],
                'description': data['description'],
                'precautions': data['precautions'],
                'severity_score': total_severity
            })
        
        # Sort by number of matched symptoms (descending) then by confidence (descending)
        results.sort(key=lambda x: (len(x['matched_symptoms']), x['confidence']), reverse=True)
        
        return results[:TOP_N_DISEASES]


# Initialize components
try:
    symptom_matcher = SymptomMatcher()
    disease_db = DiseaseDatabase(symptom_matcher)
    logger.info("API components initialized successfully")
    
    # Log all available symptoms for debugging
    all_symptoms = disease_db.get_all_symptoms()
    logger.info(f"Total available symptoms: {len(all_symptoms)}")
    logger.info(f"First 20 symptoms: {all_symptoms[:20]}")
except Exception as e:
    logger.error(f"Failed to initialize API components: {e}")
    raise


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict diseases based on symptoms
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            symptoms:
              type: array
              items:
                type: string
              description: List of user symptoms
              example: ["itching", "skin_rash", "nodal_skin_eruptions"]
    responses:
      200:
        description: Successful prediction
      400:
        description: Invalid input
    """
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({
                "success": False,
                "error": "Missing symptoms in request body"
            }), 400
        
        input_symptoms = data['symptoms']
        
        # Validate input
        if not isinstance(input_symptoms, list):
            return jsonify({
                "success": False,
                "error": "Symptoms must be provided as a list"
            }), 400
            
        if len(input_symptoms) < MIN_SYMPTOMS_REQUIRED:
            return jsonify({
                "success": False,
                "error": f"At least {MIN_SYMPTOMS_REQUIRED} symptom is required for prediction"
            }), 400
            
        if len(input_symptoms) > MAX_SYMPTOMS_ALLOWED:
            return jsonify({
                "success": False,
                "error": f"Maximum {MAX_SYMPTOMS_ALLOWED} symptoms allowed for prediction"
            }), 400
        
        # Log the received symptoms for debugging
        logger.info(f"Received symptoms: {input_symptoms}")
        
        # Normalize and map symptoms to canonical form
        canonical_symptoms = []
        unrecognized_symptoms = []
        for symptom in input_symptoms:
            canonical = symptom_matcher.map_to_canonical(symptom)
            if canonical:
                canonical_symptoms.append(canonical)
            else:
                unrecognized_symptoms.append(symptom)
        
        # Log the mapping results
        logger.info(f"Canonical symptoms: {canonical_symptoms}")
        logger.info(f"Unrecognized symptoms: {unrecognized_symptoms}")
        
        if unrecognized_symptoms:
            # Try to suggest similar symptoms
            suggestions = {}
            for symptom in unrecognized_symptoms:
                normalized = symptom_matcher._normalize_name(symptom)
                if symptom_matcher.canonical_symptoms:
                    matches = process.extract(
                        normalized, 
                        list(symptom_matcher.canonical_symptoms), 
                        scorer=fuzz.token_sort_ratio,
                        limit=3
                    )
                    suggestions[symptom] = [match[0] for match in matches if match[1] > 50]
            
            return jsonify({
                "success": False,
                "error": f"The following symptoms could not be recognized: {unrecognized_symptoms}",
                "suggestions": suggestions,
                "message": "Please try the suggested alternatives or check the /symptoms endpoint for a full list"
            }), 400
        
        # Find matching diseases
        matched_diseases = disease_db.find_matching_diseases(canonical_symptoms)
        
        # Calculate health score based on symptom severity
        total_severity = sum(symptom_matcher.get_severity(s) for s in canonical_symptoms)
        avg_severity = total_severity / len(canonical_symptoms) if canonical_symptoms else 0
        health_score = max(0, 100 - min(100, avg_severity * 10))
        
        response = {
            "input_symptoms": [s.replace('_', ' ').title() for s in canonical_symptoms],
            "health_score": round(health_score, 2),
            "predictions": matched_diseases,
            "recommendation": "Consult a healthcare professional for accurate diagnosis and treatment"
        }
        
        return jsonify({
            "success": True, 
            "data": response, 
            "meta": {
                "predictions_found": len(matched_diseases),
                "symptoms_provided": len(input_symptoms),
                "symptoms_matched": len(canonical_symptoms)
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error during prediction"
        }), 500


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """
    Get all known symptoms
    ---
    responses:
      200:
        description: List of all known symptoms
    """
    try:
        symptoms = disease_db.get_all_symptoms()
        formatted_symptoms = [s.replace('_', ' ').title() for s in symptoms]
        return jsonify({"success": True, "data": formatted_symptoms})
    except Exception as e:
        logger.error(f"Error fetching symptoms: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve symptoms list"
        }), 500


@app.route("/diseases", methods=["GET"])
def get_diseases():
    """
    Get all known diseases
    ---
    responses:
      200:
        description: List of all known diseases
    """
    try:
        diseases = list(disease_db.diseases.keys())
        formatted_diseases = [d.replace('_', ' ').title() for d in diseases]
        return jsonify({"success": True, "data": formatted_diseases})
    except Exception as e:
        logger.error(f"Error fetching diseases: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve diseases list"
        }), 500


@app.route("/symptom_severity/<symptom>", methods=["GET"])
def get_symptom_severity(symptom):
    """
    Get severity of a specific symptom
    ---
    parameters:
      - name: symptom
        in: path
        type: string
        required: true
    responses:
      200:
        description: Severity information for the symptom
      404:
        description: Symptom not found
    """
    try:
        canonical_symptom = symptom_matcher.map_to_canonical(symptom)
        if not canonical_symptom:
            return jsonify({
                "success": False,
                "error": "Symptom not found"
            }), 404
            
        severity = symptom_matcher.get_severity(canonical_symptom)
        return jsonify({
            "success": True, 
            "symptom": canonical_symptom.replace('_', ' ').title(), 
            "severity": severity
        })
    except Exception as e:
        logger.error(f"Error fetching symptom severity: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to retrieve symptom severity"
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    ---
    responses:
      200:
        description: API status
    """
    return jsonify({
        "success": True,
        "status": "healthy",
        "diseases_loaded": len(disease_db.diseases),
        "symptoms_loaded": len(symptom_matcher.canonical_symptoms)
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'templates/index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('templates', filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)