#!/usr/bin/env python3
"""Test du yaw en temps réel avec affichage vidéo."""

import cv2
import numpy as np
import sys
from pathlib import Path

# Ajouter dynamiquement le dossier racine du projet à sys.path pour les imports locaux
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.fr_core.landmark_onnx import LandmarkDetectorONNX

def main():
    detector = LandmarkDetectorONNX()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return
    
    print('Test Yaw en temps réel avec vidéo:')
    print('- Tournez la tête à GAUCHE: yaw devrait être POSITIF (+)')
    print('- Tournez la tête à DROITE: yaw devrait être NEGATIF (-)')
    print('- Face à la caméra: yaw devrait être proche de 0')
    print('Appuyez sur q pour quitter\n')
    
    cv2.namedWindow('Test Yaw', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Test Yaw', 800, 600)
    
    # Rotation de l'image : 0=aucune, 1=90° sens horaire, 2=180°, 3=270° sens horaire (90° anti-horaire)
    # Logitech C920 filme en paysage (landscape) : pas de rotation nécessaire
    rotation_mode = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Appliquer la rotation si nécessaire
        if rotation_mode == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_mode == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_mode == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        result = detector.process_frame(frame, compute_pose=True)
        
        if result and result['pose']:
            yaw, pitch, roll = result['pose']
            
            # Dessiner la bbox
            if result['bbox']:
                x, y, w, h = result['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Afficher les angles en grand
            h_frame, w_frame = frame.shape[:2]
            
            # Fond semi-transparent pour le texte
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (w_frame-10, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Titre
            cv2.putText(frame, "TEST YAW EN TEMPS REEL", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Yaw avec couleur selon valeur
            yaw_color = (0, 255, 0) if abs(yaw) < 15 else (0, 165, 255)
            cv2.putText(frame, f"YAW:   {yaw:+7.1f} deg", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, yaw_color, 2)
            
            # Pitch
            cv2.putText(frame, f"PITCH: {pitch:+7.1f} deg", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Roll
            cv2.putText(frame, f"ROLL:  {roll:+7.1f} deg", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Indicateur direction
            direction_text = ""
            direction_color = (255, 255, 255)
            if yaw > 15:
                direction_text = "← GAUCHE"
                direction_color = (0, 255, 255)
            elif yaw < -15:
                direction_text = "DROITE →"
                direction_color = (255, 0, 255)
            else:
                direction_text = "↑ FRONTAL"
                direction_color = (0, 255, 0)
            
            cv2.putText(frame, direction_text, (w_frame - 250, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, direction_color, 2)
        else:
            # Pas de visage détecté
            cv2.putText(frame, "Aucun visage detecte", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Appuyez sur 'q' pour quitter", (20, h_frame - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Test Yaw', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print('\nTest terminé')

if __name__ == "__main__":
    main()
