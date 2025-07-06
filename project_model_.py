# MedGuardian AI Models - Training and Inference
# ai_models.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CardiovascularRiskModel:
    """AI model for cardiovascular disease risk prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age', 'gender', 'chest_pain_type', 'resting_bp', 'cholesterol',
            'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate', 'exercise_angina',
            'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia',
            'bmi', 'smoking', 'family_history'
        ]
        
    def create_model(self, input_dim: int) -> keras.Model:
        """Create deep neural network for cardiovascular risk prediction"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def generate_synthetic_data(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic cardiovascular data for training"""
        np.random.seed(42)
        
        # Generate realistic cardiovascular risk factors
        data = {
            'age': np.random.normal(55, 15, n_samples).clip(20, 90),
            'gender': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
            'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),
            'resting_bp': np.random.normal(130, 20, n_samples).clip(80, 200),
            'cholesterol': np.random.normal(245, 50, n_samples).clip(100, 400),
            'fasting_blood_sugar': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'rest_ecg': np.random.choice([0, 1, 2], n_samples),
            'max_heart_rate': np.random.normal(150, 25, n_samples).clip(70, 200),
            'exercise_angina': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'st_depression': np.random.exponential(1, n_samples).clip(0, 6),
            'st_slope': np.random.choice([0, 1, 2], n_samples),
            'num_major_vessels': np.random.choice([0, 1, 2, 3], n_samples),
            'thalassemia': np.random.choice([0, 1, 2, 3], n_samples),
            'bmi': np.random.normal(26, 5, n_samples).clip(15, 50),
            'smoking': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Generate realistic target variable based on risk factors
        risk_score = (
            0.02 * df['age'] +
            0.3 * df['gender'] +
            0.15 * df['chest_pain_type'] +
            0.01 * (df['resting_bp'] - 120) +
            0.005 * (df['cholesterol'] - 200) +
            0.2 * df['fasting_blood_sugar'] +
            0.1 * df['exercise_angina'] +
            0.3 * df['st_depression'] +
            0.5 * df['smoking'] +
            0.3 * df['family_history'] +
            0.05 * (df['bmi'] - 25).clip(0, np.inf) +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Convert to binary classification
        target = (risk_score > np.percentile(risk_score, 70)).astype(int)
        
        return df, target
    
    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Train the cardiovascular risk prediction model"""
        logger.info("Training cardiovascular risk model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self.create_model(X_train_scaled.shape[1])
        
        # Early stopping and model checkpointing
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate model
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': np.mean(y_pred.flatten() == y_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred.flatten())
        }
        
        logger.info(f"Model trained. Accuracy: {metrics['accuracy']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
        
        return metrics
    
    def predict(self, features: Dict) -> Dict:
        """Predict cardiovascular risk for a patient"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert features to DataFrame
        feature_vector = pd.DataFrame([features])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in feature_vector.columns:
                feature_vector[col] = 0
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector[self.feature_columns])
        
        # Make prediction
        risk_probability = self.model.predict(feature_vector_scaled)[0][0]
        
        return {
            'risk_probability': float(risk_probability),
            'risk_level': self._get_risk_level(risk_probability),
            'confidence': float(abs(risk_probability - 0.5) * 2)
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability < 0.3:
            return 'low'
        elif probability < 0.6:
            return 'medium'
        elif probability < 0.8:
            return 'high'
        else:
            return 'critical'

class DiabetesRiskModel:
    """AI model for diabetes risk prediction"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = [
            'glucose', 'blood_pressure', 'skin_thickness', 'insulin',
            'bmi', 'diabetes_pedigree', 'age', 'pregnancies'
        ]
    
    def generate_synthetic_data(self, n_samples: int = 3000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic diabetes data"""
        np.random.seed(42)
        
        data = {
            'pregnancies': np.random.poisson(2, n_samples).clip(0, 15),
            'glucose': np.random.normal(120, 30, n_samples).clip(50, 250),
            'blood_pressure': np.random.normal(70, 15, n_samples).clip(40, 120),
            'skin_thickness': np.random.normal(25, 10, n_samples).clip(0, 60),
            'insulin': np.random.exponential(100, n_samples).clip(0, 500),
            'bmi': np.random.normal(28, 8, n_samples).clip(15, 60),
            'diabetes_pedigree': np.random.exponential(0.5, n_samples).clip(0, 3),
            'age': np.random.gamma(2, 15, n_samples).clip(18, 80)
        }
        
        df = pd.DataFrame(data)
        
        # Generate target based on realistic diabetes risk factors
        risk_score = (
            0.01 * df['glucose'] +
            0.05 * df['bmi'] +
            0.02 * df['age'] +
            0.1 * df['diabetes_pedigree'] +
            0.001 * df['insulin'] +
            np.random.normal(0, 0.3, n_samples)
        )
        
        target = (risk_score > np.percentile(risk_score, 65)).astype(int)
        
        return df, target
    
    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Train diabetes risk model"""
        logger.info("Training diabetes risk model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': self.model.score(X_test_scaled, y_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        logger.info(f"Diabetes model trained. Accuracy: {metrics['accuracy']:.3f}")
        return metrics

class MedicalImageAnalyzer:
    """AI model for medical image analysis"""
    
    def __init__(self):
        self.model = None
        self.image_size = (224, 224)
        self.class_names = ['Normal', 'Abnormal']
    
    def create_cnn_model(self) -> keras.Model:
        """Create CNN model for medical image classification"""
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess medical image for analysis"""
        try:
            # Load and resize image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    def analyze_chest_xray(self, image_path: str) -> Dict:
        """Analyze chest X-ray for abnormalities"""
        if self.model is None:
            # Load pre-trained model or create new one
            self.model = self.create_cnn_model()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Generate detailed analysis
            analysis = {
                'classification': self.class_names[predicted_class],
                'confidence': float(confidence),
                'abnormality_probability': float(predictions[0][1]) if len(predictions[0]) > 1 else 0.0,
                'recommendations': self._generate_xray_recommendations(predicted_class, confidence)
