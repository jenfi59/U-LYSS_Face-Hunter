#!/usr/bin/env python3
"""
Script d'enrollment avec interface tactile complète
Compatible écran tactile sans clavier ni souris
"""

import sys
import os
import subprocess
from pathlib import Path
import cv2
import numpy as np
import re

print("[DEBUG] Script démarré")

# Configuration
PROJECT_DIR = Path(__file__).parent.absolute()
os.chdir(PROJECT_DIR)
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'

print("[DEBUG] Configuration effectuée")

class TouchscreenUI:
    """Interface tactile pour l'enrollment"""
    
    def __init__(self):
        print("[DEBUG] Initialisation TouchscreenUI")
        self.window_name = "D-Face Hunter - Enrollment"
        self.screen_width = 720   # Format smartphone portrait
        self.screen_height = 1440
        self.selected_camera = 5  # Par défaut caméra arrière
        self.username = ""
        self.keyboard_visible = False
        self.keys = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', '-'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', '_', 'DEL', 'OK']
        ]
        self.current_step = 'camera_selection'
        self.button_color = (168, 9, 9)  # BGR pour #0909A8 (bleu marin)
        self.selected_color = (100, 200, 100)  # Vert pour sélection
        self.text_color = (255, 255, 255)  # Blanc
        
        # Désactiver la mise en veille automatique
        try:
            self.disable_sleep()
        except Exception as e:
            print(f"[WARNING] Erreur dans disable_sleep: {e}")
    
    def disable_sleep(self):
        """Désactiver la mise en veille automatique"""
        try:
            # Désactiver la suspension automatique (GNOME/systemd)
            subprocess.run(['systemctl', '--user', 'mask', 'sleep.target'], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Désactiver l'écran de veille (X11)
            subprocess.run(['xset', 's', 'off'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['xset', '-dpms'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['xset', 's', 'noblank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[INFO] Mise en veille automatique désactivée")
        except Exception as e:
            print(f"[WARNING] Impossible de désactiver la mise en veille: {e}")
    
    def enable_sleep(self):
        """Réactiver la mise en veille automatique"""
        try:
            subprocess.run(['systemctl', '--user', 'unmask', 'sleep.target'], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['xset', 's', 'on'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['xset', '+dpms'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[INFO] Mise en veille automatique réactivée")
        except Exception as e:
            print(f"[WARNING] Impossible de réactiver la mise en veille: {e}")
        
    def main_menu_screen(self):
        """Menu principal avec 4 options"""
        # Détruire et recréer la fenêtre pour forcer le bon ratio portrait
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        cv2.waitKey(1)  # Forcer la mise à jour
        
        img = self.create_blank_screen()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Titre principal
        title = "D-Face Hunter v1.2.1"
        text_size = cv2.getTextSize(title, font, 1.3, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(img, title, (text_x, 150), font, 1.3, (0, 0, 0), 3)
        
        # Sous-titre
        subtitle = "Menu Principal"
        text_size = cv2.getTextSize(subtitle, font, 0.9, 2)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(img, subtitle, (text_x, 220), font, 0.9, (100, 100, 100), 2)
        
        # Boutons du menu
        button_width = 600
        button_height = 150
        button_x = 60
        y_start = 350
        spacing = 180
        
        menu_items = [
            ("ENROLLMENT", 'enrollment', "Enroler un nouveau visage"),
            ("VALIDATION", 'validation', "Verifier une identite"),
            ("GESTION", 'manage', "Gerer les modeles"),
            ("QUITTER", 'quit', "Fermer l'application")
        ]
        
        for idx, (title, action, desc) in enumerate(menu_items):
            btn_y = y_start + idx * spacing
            self.draw_button(img, button_x, btn_y, button_width, button_height, title)
            # Description sous le bouton
            desc_y = btn_y + button_height - 20
            cv2.putText(img, desc, (button_x + 30, desc_y), font, 0.55, (150, 150, 150), 1)
        
        cv2.imshow(self.window_name, img)
        
        selected_action = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for idx, (title, action, desc) in enumerate(menu_items):
                    btn_y = y_start + idx * spacing
                    if button_x <= x <= button_x + button_width and btn_y <= y <= btn_y + button_height:
                        selected_action[0] = action
                        cv2.setMouseCallback(self.window_name, lambda *args: None)
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
        while selected_action[0] is None:
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                selected_action[0] = 'quit'
                break
        
        return selected_action[0]
        
    def loading_screen(self, message="Enrollment en cours..."):
        """Écran de chargement animé pendant le traitement"""
        img = self.create_blank_screen()
        
        # Fond bleu foncé
        img[:] = (50, 50, 50)
        
        # Titre
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = message
        text_size = cv2.getTextSize(title, font, 1.2, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(img, title, (text_x, 300), font, 1.2, (255, 255, 255), 3)
        
        # Message d'attente
        wait_msg = "Veuillez patienter..."
        text_size = cv2.getTextSize(wait_msg, font, 0.9, 2)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(img, wait_msg, (text_x, 500), font, 0.9, (200, 200, 200), 2)
        
        # Barre de progression animée
        bar_width = 500
        bar_height = 30
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = 700
        
        # Fond de la barre
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)
        
    def animate_loading(self, process, total_phases=3):
        """Anime l'écran de chargement pendant l'exécution du process"""
        import time
        
        # S'assurer que la fenêtre existe
        try:
            cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        
        phases = [
            "Phase 1/3: Capture automatique...",
            "Phase 2/3: Validation manuelle...",
            "Phase 3/3: Verification DTW..."
        ]
        
        phase_idx = 0
        progress = 0
        start_time = time.time()
        
        while process.poll() is None:  # Tant que le process tourne
            # Créer l'écran
            img = self.create_blank_screen()
            img[:] = (50, 50, 50)  # Fond gris foncé
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Titre principal
            title = "Enrollment en cours"
            text_size = cv2.getTextSize(title, font, 1.2, 3)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(img, title, (text_x, 250), font, 1.2, (255, 255, 255), 3)
            
            # Phase actuelle (estimation basée sur le temps)
            elapsed = time.time() - start_time
            if elapsed < 15:
                phase_idx = 0
            elif elapsed < 35:
                phase_idx = 1
            else:
                phase_idx = 2
            
            phase_msg = phases[min(phase_idx, len(phases)-1)]
            text_size = cv2.getTextSize(phase_msg, font, 0.8, 2)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(img, phase_msg, (text_x, 450), font, 0.8, (100, 200, 255), 2)
            
            # Barre de progression animée
            bar_width = 500
            bar_height = 30
            bar_x = (self.screen_width - bar_width) // 2
            bar_y = 700
            
            # Fond de la barre
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (100, 100, 100), -1)
            
            # Progression (animation infinie)
            progress = (progress + 5) % bar_width
            fill_width = min(150, progress)
            if progress > bar_width - 150:
                fill_width = bar_width - progress + 150
            
            cv2.rectangle(img, (bar_x + progress - fill_width, bar_y), 
                         (bar_x + progress, bar_y + bar_height), 
                         self.button_color, -1)
            
            # Bordure
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (255, 255, 255), 2)
            
            # Message info
            info_msg = "Suivez les instructions à l'écran"
            text_size = cv2.getTextSize(info_msg, font, 0.6, 2)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(img, info_msg, (text_x, 950), font, 0.6, (180, 180, 180), 2)
            
            try:
                cv2.imshow(self.window_name, img)
                cv2.waitKey(100)  # 100ms par frame = 10 FPS
            except:
                # Recréer la fenêtre si nécessaire
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
                cv2.imshow(self.window_name, img)
                cv2.waitKey(100)
        
        # Après la fin du process, s'assurer que la fenêtre existe toujours
        try:
            cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        
    def create_blank_screen(self):
        """Créer un écran blanc pour l'interface"""
        return np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * 255
    
    def draw_button(self, img, x, y, w, h, text, color=None, text_color=None, selected=False):
        """Dessiner un bouton"""
        if color is None:
            color = self.button_color
        if text_color is None:
            text_color = self.text_color
            
        if selected:
            color = self.selected_color
            
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
    def camera_selection_screen(self):
        """Écran de sélection de caméra - format portrait"""
        img = self.create_blank_screen()
        
        # Titre centré en haut
        title = "Selectionner Camera"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(title, font, 1.2, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(img, title, (text_x, 100), font, 1.2, (0, 0, 0), 3)
        
        # Boutons verticaux pour format portrait
        button_width = 600
        button_height = 180
        button_x = (self.screen_width - button_width) // 2
        
        # Bouton Caméra Arrière
        rear_y = 300
        self.draw_button(img, button_x, rear_y, button_width, button_height, 
                        "Camera Arriere", selected=(self.selected_camera == 5))
        
        # Bouton Caméra Avant
        front_y = 550
        self.draw_button(img, button_x, front_y, button_width, button_height,
                        "Camera Avant", selected=(self.selected_camera == 6))
        
        # Bouton Continuer en bas
        continue_y = 1200
        self.draw_button(img, button_x, continue_y, button_width, button_height,
                        "CONTINUER")
        
        cv2.imshow(self.window_name, img)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Vérifier clic sur Caméra Arrière
                if button_x <= x <= button_x + button_width and rear_y <= y <= rear_y + button_height:
                    self.selected_camera = 5
                    self.camera_selection_screen()
                # Vérifier clic sur Caméra Avant
                elif button_x <= x <= button_x + button_width and front_y <= y <= front_y + button_height:
                    self.selected_camera = 6
                    self.camera_selection_screen()
                # Vérifier clic sur Continuer
                elif button_x <= x <= button_x + button_width and continue_y <= y <= continue_y + button_height:
                    self.current_step = 'username_input'
                    cv2.setMouseCallback(self.window_name, lambda *args: None)
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
    def draw_keyboard(self, img, start_y):
        """Dessiner le clavier virtuel - adapté format portrait"""
        key_width = 60
        key_height = 70
        margin = 8
        start_x = 30
        
        for row_idx, row in enumerate(self.keys):
            y = start_y + row_idx * (key_height + margin)
            for col_idx, key in enumerate(row):
                x = start_x + col_idx * (key_width + margin)
                
                # Couleur spéciale pour DEL et OK
                if key in ['DEL', 'OK']:
                    color = (50, 50, 200) if key == 'DEL' else (50, 200, 50)
                else:
                    color = self.button_color
                
                self.draw_button(img, x, y, key_width, key_height, key, color=color)
    
    def username_input_screen(self):
        """Écran de saisie du nom d'utilisateur avec clavier virtuel - format portrait"""
        img = self.create_blank_screen()
        
        # Titre
        title = "Nom Utilisateur"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(title, font, 1.0, 2)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(img, title, (text_x, 80), font, 1.0, (0, 0, 0), 2)
        
        # Zone de texte pour afficher le nom saisi
        text_box_y = 130
        cv2.rectangle(img, (40, text_box_y), (680, text_box_y + 70), (200, 200, 200), -1)
        cv2.rectangle(img, (40, text_box_y), (680, text_box_y + 70), (0, 0, 0), 3)
        
        # Afficher le texte saisi
        if self.username:
            cv2.putText(img, self.username, (60, text_box_y + 50), font, 1.0, (0, 0, 0), 2)
        
        # Clavier virtuel en bas
        keyboard_start_y = 800
        self.draw_keyboard(img, keyboard_start_y)
        
        cv2.imshow(self.window_name, img)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                key_width = 60
                key_height = 70
                margin = 8
                start_x = 30
                keyboard_start_y = 800
                
                for row_idx, row in enumerate(self.keys):
                    key_y = keyboard_start_y + row_idx * (key_height + margin)
                    for col_idx, key in enumerate(row):
                        key_x = start_x + col_idx * (key_width + margin)
                        
                        if key_x <= x <= key_x + key_width and key_y <= y <= key_y + key_height:
                            if key == 'DEL':
                                self.username = self.username[:-1]
                            elif key == 'OK':
                                if self.username:
                                    self.current_step = 'confirm_start'
                                    cv2.setMouseCallback(self.window_name, lambda *args: None)
                                    return
                            else:
                                if len(self.username) < 20:
                                    self.username += key
                            
                            self.username_input_screen()
                            return
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
    
    def confirm_start_screen(self):
        """Écran de confirmation avant démarrage - format portrait"""
        img = self.create_blank_screen()
        
        # Titre
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "Confirmation"
        text_size = cv2.getTextSize(title, font, 1.2, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(img, title, (text_x, 100), font, 1.2, (0, 0, 0), 3)
        
        # Informations sur deux lignes
        info_y = 250
        cv2.putText(img, f"Utilisateur: {self.username}", (80, info_y), font, 0.9, (0, 0, 0), 2)
        
        camera_name = "Arriere" if self.selected_camera == 5 else "Avant"
        cv2.putText(img, f"Camera: {camera_name}", (80, info_y + 60), font, 0.9, (0, 0, 0), 2)
        
        # Instructions compactes
        instructions = [
            "L'enrollment va demarrer.",
            "Suivez les instructions",
            "a l'ecran pour capturer",
            "vos caracteristiques faciales."
        ]
        
        inst_y = 450
        for idx, line in enumerate(instructions):
            cv2.putText(img, line, (80, inst_y + idx * 50), font, 0.7, (50, 50, 50), 2)
        
        # Bouton DEMARRER
        button_width = 600
        button_height = 120
        button_x = (self.screen_width - button_width) // 2
        button_y = 1200
        
        self.draw_button(img, button_x, button_y, button_width, button_height, "DEMARRER")
        
        cv2.imshow(self.window_name, img)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
                    self.current_step = 'enrollment_running'
                    cv2.setMouseCallback(self.window_name, lambda *args: None)
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
    
    def parse_validation_output(self, output):
        """Parser la sortie de la validation pour extraire les résultats DTW"""
        result = {
            'success': False,
            'dtw_distance': None,
            'verified': False,
            'message': ''
        }
        
        # Rechercher la distance DTW
        dtw_match = re.search(r'DTW Distance\s*:\s*([\d.]+)', output)
        if dtw_match:
            result['dtw_distance'] = float(dtw_match.group(1))
        
        # Vérifier si vérifié
        if 'Verified : YES' in output or 'VERIFIED' in output.upper():
            result['verified'] = True
            result['success'] = True
            result['message'] = 'Verification reussie!'
        elif 'Verified : NO' in output or 'NOT VERIFIED' in output.upper():
            result['verified'] = False
            result['message'] = 'Verification echouee'
        else:
            result['message'] = 'Enrollment termine'
            result['success'] = True
        
        return result
    
    def final_result_screen(self, result):
        """Écran des résultats finaux avec options - format portrait"""
        img = self.create_blank_screen()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Titre
        title = "Resultats"
        text_size = cv2.getTextSize(title, font, 1.2, 3)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        cv2.putText(img, title, (text_x, 100), font, 1.2, (0, 0, 0), 3)
        
        # Message résultat
        message = result.get('message', 'Enrollment termine')
        text_size = cv2.getTextSize(message, font, 0.9, 2)[0]
        text_x = (self.screen_width - text_size[0]) // 2
        color = (0, 200, 0) if result.get('success') else (0, 0, 200)
        cv2.putText(img, message, (text_x, 200), font, 0.9, color, 2)
        
        # Distance DTW si disponible
        if result.get('dtw_distance') is not None:
            dtw_text = f"Distance DTW: {result['dtw_distance']:.2f}"
            text_size = cv2.getTextSize(dtw_text, font, 0.8, 2)[0]
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(img, dtw_text, (text_x, 280), font, 0.8, (0, 0, 0), 2)
        
        # Utilisateur
        user_text = f"Utilisateur: {self.username}"
        cv2.putText(img, user_text, (80, 380), font, 0.7, (0, 0, 0), 2)
        
        # Boutons d'action en pile verticale
        button_width = 600
        button_height = 80
        button_x = 60
        y_btn = 1050
        spacing = 100
        
        buttons = [
            ("OK", 'ok'),
            ("RELANCER", 'restart'),
            ("VALIDATION", 'validation'),
            ("GESTION", 'manage'),
            ("QUITTER", 'quit')
        ]
        
        for idx, (btn_text, action) in enumerate(buttons):
            btn_y = y_btn + idx * spacing
            self.draw_button(img, button_x, btn_y, button_width, button_height, btn_text)
        
        cv2.imshow(self.window_name, img)
        
        selected_action = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for idx, (btn_text, action) in enumerate(buttons):
                    btn_y = y_btn + idx * spacing
                    if button_x <= x <= button_x + button_width and btn_y <= y <= btn_y + button_height:
                        selected_action[0] = action
                        cv2.setMouseCallback(self.window_name, lambda *args: None)
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
        # Attendre sélection
        while selected_action[0] is None:
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                selected_action[0] = 'quit'
                break
        
        return selected_action[0]
    
    def select_model_screen(self):
        """Écran de sélection du modèle à valider (scrollable)"""
        models_dir = PROJECT_DIR / "models" / "users"
        model_files = sorted([f.stem for f in models_dir.glob("*.npz") if f.name != ".gitkeep"])
        
        if not model_files:
            # Pas de modèles disponibles
            img = self.create_blank_screen()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Aucun modele disponible", (100, 400), font, 0.9, (0, 0, 0), 2)
            cv2.putText(img, "Veuillez d'abord enroler", (100, 500), font, 0.9, (0, 0, 0), 2)
            cv2.putText(img, "un utilisateur", (100, 600), font, 0.9, (0, 0, 0), 2)
            
            # Bouton retour
            self.draw_button(img, 60, 1300, 600, 80, "RETOUR")
            cv2.imshow(self.window_name, img)
            
            selected = [None]
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if 60 <= x <= 660 and 1300 <= y <= 1380:
                        selected[0] = 'back'
            
            cv2.setMouseCallback(self.window_name, mouse_callback)
            while selected[0] is None:
                cv2.waitKey(100)
            return None
        
        # Variables pour le scroll
        scroll_offset = [0]
        max_visible = 8  # Nombre max de modèles visibles
        selected_model = [None]
        
        def draw_model_list():
            img = self.create_blank_screen()
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Titre
            cv2.putText(img, "Selectionner un modele", (60, 100), font, 1.0, (0, 0, 0), 3)
            
            # Zone de liste
            list_y = 180
            button_height = 120
            button_width = 600
            button_x = 60
            
            # Afficher les modèles visibles
            start_idx = scroll_offset[0]
            end_idx = min(start_idx + max_visible, len(model_files))
            
            for i in range(start_idx, end_idx):
                model_name = model_files[i]
                btn_y = list_y + (i - start_idx) * button_height
                self.draw_button(img, button_x, btn_y, button_width, button_height - 20, model_name)
            
            # Flèches de scroll si nécessaire
            if scroll_offset[0] > 0:
                # Flèche haut
                cv2.arrowedLine(img, (360, 150), (360, 120), (0, 0, 0), 8, tipLength=0.4)
            
            if end_idx < len(model_files):
                # Flèche bas
                cv2.arrowedLine(img, (360, list_y + max_visible * button_height), 
                              (360, list_y + max_visible * button_height + 30), (0, 0, 0), 8, tipLength=0.4)
            
            # Compteur
            count_text = f"{len(model_files)} modele(s) disponible(s)"
            cv2.putText(img, count_text, (60, 1250), font, 0.7, (100, 100, 100), 2)
            
            # Bouton annuler
            self.draw_button(img, 60, 1300, 600, 80, "ANNULER")
            
            cv2.imshow(self.window_name, img)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Vérifier les modèles cliquables
                list_y = 180
                button_height = 120
                button_x = 60
                button_width = 600
                
                start_idx = scroll_offset[0]
                end_idx = min(start_idx + max_visible, len(model_files))
                
                for i in range(start_idx, end_idx):
                    btn_y = list_y + (i - start_idx) * button_height
                    if button_x <= x <= button_x + button_width and btn_y <= y <= btn_y + button_height - 20:
                        selected_model[0] = model_files[i]
                        return
                
                # Flèche haut
                if 300 <= x <= 420 and 120 <= y <= 150 and scroll_offset[0] > 0:
                    scroll_offset[0] = max(0, scroll_offset[0] - 1)
                    draw_model_list()
                
                # Flèche bas
                list_bottom = 180 + max_visible * button_height
                if 300 <= x <= 420 and list_bottom <= y <= list_bottom + 30 and end_idx < len(model_files):
                    scroll_offset[0] = min(len(model_files) - max_visible, scroll_offset[0] + 1)
                    draw_model_list()
                
                # Bouton annuler
                if 60 <= x <= 660 and 1300 <= y <= 1380:
                    selected_model[0] = 'cancel'
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
        draw_model_list()
        
        while selected_model[0] is None:
            cv2.waitKey(100)
        
        return None if selected_model[0] == 'cancel' else selected_model[0]
    
    def select_validation_mode_screen(self, model_name):
        """Écran de sélection du mode de validation"""
        img = self.create_blank_screen()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Titre
        cv2.putText(img, f"Validation: {model_name}", (60, 100), font, 0.9, (0, 0, 0), 2)
        cv2.putText(img, "Choisir le mode", (60, 180), font, 1.0, (0, 0, 0), 3)
        
        # Boutons de choix
        button_width = 600
        button_height = 140
        button_x = 60
        y_start = 280
        spacing = 160
        
        modes = [
            ("1:1 Verification", "1:1", "Verifier contre ce modele"),
            ("1:N Identification", "1:N", "Identifier dans la base"),
            ("Mode Spatial", "spatial", "Comparaison frame par frame"),
            ("Mode Sequentiel", "sequential", "Multi-criteres (securite++)")
        ]
        
        for idx, (title, mode, desc) in enumerate(modes):
            btn_y = y_start + idx * spacing
            self.draw_button(img, button_x, btn_y, button_width, button_height, title)
            # Description
            cv2.putText(img, desc, (button_x + 20, btn_y + button_height - 15), 
                       font, 0.5, (150, 150, 150), 1)
        
        # Bouton retour
        self.draw_button(img, 60, 1300, 600, 80, "RETOUR")
        
        cv2.imshow(self.window_name, img)
        
        selected_mode = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for idx, (title, mode, desc) in enumerate(modes):
                    btn_y = y_start + idx * spacing
                    if button_x <= x <= button_x + button_width and btn_y <= y <= btn_y + button_height:
                        selected_mode[0] = mode
                        return
                
                # Bouton retour
                if 60 <= x <= 660 and 1300 <= y <= 1380:
                    selected_mode[0] = 'back'
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
        while selected_mode[0] is None:
            cv2.waitKey(100)
        
        return selected_mode[0]
    
    def validation_capture_screen(self, model_name, mode):
        """Écran de capture pour la validation avec bouton START"""
        img = self.create_blank_screen()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Titre
        cv2.putText(img, "Validation - Capture", (60, 100), font, 1.0, (0, 0, 0), 3)
        cv2.putText(img, f"Modele: {model_name}", (60, 180), font, 0.8, (0, 0, 0), 2)
        cv2.putText(img, f"Mode: {mode}", (60, 240), font, 0.8, (0, 0, 0), 2)
        
        # Instructions
        instructions = [
            "1. Positionnez-vous face a la camera",
            "2. Assurez un bon eclairage",
            "3. Appuyez sur START pour commencer",
            "4. La capture durera 3-5 secondes"
        ]
        
        y_inst = 350
        for inst in instructions:
            cv2.putText(img, inst, (80, y_inst), font, 0.65, (50, 50, 50), 2)
            y_inst += 70
        
        # Bouton START (grand et visible)
        button_start_y = 700
        button_start_height = 180
        self.draw_button(img, 60, button_start_y, 600, button_start_height, "START VALIDATION")
        
        # Bouton annuler
        button_cancel_y = 1000
        button_cancel_height = 100
        self.draw_button(img, 60, button_cancel_y, 600, button_cancel_height, "ANNULER")
        
        cv2.imshow(self.window_name, img)
        
        selected_action = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # START (zone élargie)
                if 60 <= x <= 660 and button_start_y <= y <= (button_start_y + button_start_height):
                    selected_action[0] = 'start'
                    print(f"[DEBUG] START cliqué à ({x}, {y})")
                # ANNULER
                elif 60 <= x <= 660 and button_cancel_y <= y <= (button_cancel_y + button_cancel_height):
                    selected_action[0] = 'cancel'
                    print(f"[DEBUG] ANNULER cliqué à ({x}, {y})")
                else:
                    print(f"[DEBUG] Clic hors zone: ({x}, {y})")
        
        cv2.setMouseCallback(self.window_name, mouse_callback)
        
        while selected_action[0] is None:
            cv2.waitKey(100)
        
        return selected_action[0]
    
    def validation_result_screen(self, result):
        """Écran de résultats de validation"""
        img = self.create_blank_screen()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Titre
        cv2.putText(img, "Resultats de Validation", (60, 100), font, 1.0, (0, 0, 0), 3)
        
        # Statut (grand et visible)
        verified = result.get('verified', False)
        if verified:
            status_text = "VALIDE"
            status_color = (0, 200, 0)  # Vert
        else:
            status_text = "REFUSE"
            status_color = (0, 0, 200)  # Rouge
        
        cv2.putText(img, status_text, (200, 250), font, 2.0, status_color, 4)
        
        # Détails
        y_detail = 350
        details = [
            f"Utilisateur: {result.get('user_id', 'INCONNU')}",
            f"Distance: {result.get('distance', 0):.4f}",
            f"Frames: {result.get('frames_used', 'N/A')}",
            f"Couverture: {result.get('coverage', 'N/A')}"
        ]
        
        for detail in details:
            cv2.putText(img, detail, (80, y_detail), font, 0.8, (0, 0, 0), 2)
            y_detail += 80
        
        # Message additionnel
        if 'error' in result:
            cv2.putText(img, f"Erreur: {result['error']}", (80, y_detail), font, 0.7, (0, 0, 200), 2)
        
        # Boutons d'action
        button_width = 600
        button_height = 80
        button_x = 60
        y_btn = 1050
        spacing = 100
        
        buttons = [
            ("NOUVELLE VALIDATION", 'restart'),
            ("MENU PRINCIPAL", 'menu'),
            ("QUITTER", 'quit')
        ]
        
        for idx, (btn_text, action) in enumerate(buttons):
            btn_y = y_btn + idx * spacing
            self.draw_button(img, button_x, btn_y, button_width, button_height, btn_text)
        
        # S'assurer que la fenêtre existe avant d'afficher
        try:
            cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)  # Forcer le refresh
        
        selected_action = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for idx, (btn_text, action) in enumerate(buttons):
                    btn_y = y_btn + idx * spacing
                    if button_x <= x <= button_x + button_width and btn_y <= y <= btn_y + button_height:
                        selected_action[0] = action
                        return
        
        try:
            cv2.setMouseCallback(self.window_name, mouse_callback)
        except cv2.error as e:
            print(f"[WARNING] Erreur setMouseCallback: {e}")
            # Recréer la fenêtre si nécessaire
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
            cv2.imshow(self.window_name, img)
            cv2.setMouseCallback(self.window_name, mouse_callback)
        
        while selected_action[0] is None:
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                selected_action[0] = 'menu'
                break
        
        # Nettoyer le callback avant de quitter
        try:
            cv2.setMouseCallback(self.window_name, lambda *args: None)
        except:
            pass
        
        return selected_action[0]
    
    def run_validation_capture(self, model_name, model_path):
        """Capture vidéo pour validation avec affichage du flux"""
        import time
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from scipy.spatial.transform import Rotation as R
        
        # Charger la configuration
        sys.path.insert(0, str(PROJECT_DIR / "src"))
        from fr_core import get_config, VerificationDTW
        config = get_config()
        
        # Charger le modèle enrollé
        verifier = VerificationDTW()
        enrollment_result = verifier.load_enrollment(model_path.stem, model_path.parent)
        if enrollment_result is None:
            return {'verified': False, 'error': 'Modèle non trouvé', 'distance': float('inf')}
        enrolled_landmarks, enrolled_poses = enrollment_result
        
        # Créer le détecteur MediaPipe
        mediapipe_model = PROJECT_DIR / "models" / "mediapipe" / "face_landmarker_v2_with_blendshapes.task"
        base_options = python.BaseOptions(model_asset_path=str(mediapipe_model))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        
        # Ouvrir la caméra
        cap = cv2.VideoCapture(self.selected_camera)
        if not cap.isOpened():
            return {'verified': False, 'error': 'Caméra non accessible', 'distance': float('inf')}
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Buffers pour les landmarks
        buffer_landmarks = []
        buffer_poses = []
        
        # Paramètres de capture
        duration = 4.0  # 4 secondes
        min_frames = 15
        start_time = time.time()
        frame_count = 0
        
        # Créer la fenêtre
        validation_window = "Validation - Capture en cours"
        cv2.namedWindow(validation_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(validation_window, self.screen_width, self.screen_height)
        
        try:
            while (time.time() - start_time) < duration or len(buffer_landmarks) < min_frames:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # Traiter avec MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = detector.detect(mp_image)
                
                # Extraire landmarks et pose
                if result.face_landmarks:
                    h, w = frame.shape[:2]
                    landmarks_all = result.face_landmarks[0]
                    landmarks_468 = np.array([
                        [lm.x * w, lm.y * h, lm.z * w] for lm in landmarks_all[:468]
                    ], dtype=np.float32)
                    
                    # Pose
                    if result.facial_transformation_matrixes:
                        pose_matrix = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
                        rot_mat = pose_matrix[:3, :3]
                        rot = R.from_matrix(rot_mat)
                        angles = rot.as_euler('XZY', degrees=True)
                        yaw, pitch, roll = angles[2], angles[0], angles[1]
                    else:
                        yaw, pitch, roll = 0.0, 0.0, 0.0
                    
                    buffer_landmarks.append(landmarks_468.copy())
                    buffer_poses.append([float(yaw), float(pitch), float(roll)])
                
                # Redimensionner et afficher
                frame_resized = cv2.resize(frame, (self.screen_width, self.screen_height))
                
                # Overlay avec info
                font = cv2.FONT_HERSHEY_SIMPLEX
                elapsed = time.time() - start_time
                remaining = max(0, duration - elapsed)
                
                # Fond semi-transparent pour le texte
                overlay = frame_resized.copy()
                cv2.rectangle(overlay, (0, 0), (self.screen_width, 120), (0, 0, 0), -1)
                frame_resized = cv2.addWeighted(overlay, 0.6, frame_resized, 0.4, 0)
                
                # Texte
                cv2.putText(frame_resized, f"VALIDATION: {model_name}", (20, 40), 
                           font, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_resized, f"Frames: {len(buffer_landmarks)}/{min_frames}", (20, 75), 
                           font, 0.7, (100, 255, 100), 2)
                cv2.putText(frame_resized, f"Temps: {remaining:.1f}s", (20, 105), 
                           font, 0.7, (100, 200, 255), 2)
                
                # Barre de progression
                progress = min(1.0, elapsed / duration)
                bar_width = int(self.screen_width * 0.8)
                bar_x = (self.screen_width - bar_width) // 2
                bar_y = self.screen_height - 50
                
                cv2.rectangle(frame_resized, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), 
                             (100, 100, 100), -1)
                cv2.rectangle(frame_resized, (bar_x, bar_y), 
                             (bar_x + int(bar_width * progress), bar_y + 20), 
                             (100, 200, 255), -1)
                cv2.rectangle(frame_resized, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), 
                             (255, 255, 255), 2)
                
                cv2.imshow(validation_window, frame_resized)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC pour annuler
                    cv2.destroyWindow(validation_window)
                    cap.release()
                    return {'verified': False, 'error': 'Annulé par utilisateur', 'distance': float('inf')}
        
        finally:
            cap.release()
            cv2.destroyWindow(validation_window)
        
        # Vérifier si assez de frames
        if len(buffer_landmarks) < min_frames:
            return {'verified': False, 'error': 'Pas assez de frames', 'distance': float('inf'), 
                   'frames_used': len(buffer_landmarks)}
        
        # Conversion pour vérification
        max_frames = min(45, len(buffer_landmarks))
        probe_landmarks = np.stack(buffer_landmarks[-max_frames:], axis=0).astype(np.float32)
        probe_poses = np.array(buffer_poses[-max_frames:], dtype=np.float32)
        
        # Vérification
        is_match, distance, details = verifier.verify_auto(
            probe_landmarks, probe_poses, enrolled_landmarks, enrolled_poses
        )
        
        # Construire le résultat
        result = {
            'verified': is_match,
            'distance': distance,
            'user_id': model_name,
            'frames_used': max_frames,
            'coverage': f"{details.get('coverage', 0):.1f}%" if 'coverage' in details else 'N/A'
        }
        
        return result
    
    def run_validation_workflow(self):
        """Workflow complet de validation"""
        while True:
            # 1. Sélection du modèle
            model_name = self.select_model_screen()
            if model_name is None:
                return 'back'  # Retour au menu principal
            
            # 2. Sélection du mode
            mode = self.select_validation_mode_screen(model_name)
            if mode == 'back':
                continue  # Retour à la sélection de modèle
            
            # 3. Écran de capture
            action = self.validation_capture_screen(model_name, mode)
            if action == 'cancel':
                continue  # Retour à la sélection de mode
            
            # 4. Lancer la validation avec flux vidéo visible
            print(f"\n=== Démarrage validation pour {model_name} ===")
            print(f"Mode: {mode}")
            print(f"Caméra: {self.selected_camera}")
            
            model_path = PROJECT_DIR / "models" / "users" / f"{model_name}.npz"
            
            try:
                # Lancer la validation avec flux vidéo
                result = self.run_validation_capture(model_name, model_path)
                
                # 5. Afficher les résultats
                try:
                    action = self.validation_result_screen(result)
                except Exception as e:
                    print(f"[ERROR] Erreur dans validation_result_screen: {e}")
                    return 'menu'
                
                if action == 'restart':
                    continue  # Nouvelle validation
                elif action == 'quit':
                    return 'quit'
                else:  # menu
                    return 'menu'
                    
            except Exception as e:
                print(f"Erreur lors de la validation: {e}")
                import traceback
                traceback.print_exc()
                result = {'verified': False, 'error': str(e), 'distance': float('inf')}
                try:
                    action = self.validation_result_screen(result)
                    if action == 'quit':
                        return 'quit'
                    elif action == 'menu':
                        return 'menu'
                except:
                    return 'menu'
    
    def parse_verification_output(self, output):
        """Parser la sortie du script de vérification"""
        result = {
            'verified': False,
            'distance': float('inf'),
            'frames_used': 'N/A',
            'coverage': 'N/A'
        }
        
        try:
            # Chercher "Verified   : YES" ou "NO"
            if 'Verified   : YES' in output or 'Verified   : True' in output:
                result['verified'] = True
            
            # Chercher la distance
            distance_match = re.search(r'Distance\s*:\s*([0-9.]+)', output)
            if distance_match:
                result['distance'] = float(distance_match.group(1))
            
            # Chercher la couverture
            coverage_match = re.search(r'Coverage\s*:\s*([0-9.]+)', output)
            if coverage_match:
                result['coverage'] = f"{float(coverage_match.group(1)):.1f}%"
            
            # Chercher le nombre de frames
            frames_match = re.search(r'Frames used\s*:\s*([0-9]+)', output)
            if frames_match:
                result['frames_used'] = frames_match.group(1)
            
            # Détecter les erreurs
            if 'error' in output.lower() or 'not enough frames' in output.lower():
                error_lines = [line for line in output.split('\n') if 'error' in line.lower()]
                if error_lines:
                    result['error'] = error_lines[0]
        
        except Exception as e:
            result['error'] = f"Erreur de parsing: {e}"
        
        return result


def main():
    """Fonction principale"""
    print("[INFO] Démarrage D-Face Hunter")
    
    try:
        ui = TouchscreenUI()
        cv2.namedWindow(ui.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ui.window_name, ui.screen_width, ui.screen_height)
        
        # Menu principal
        action = ui.main_menu_screen()
        
        if action == 'quit':
            cv2.destroyAllWindows()
            ui.enable_sleep()
            sys.exit(0)
        elif action == 'validation':
            ui.camera_selection_screen()
            while ui.current_step == 'camera_selection':
                cv2.waitKey(100)
            result = ui.run_validation_workflow()
            if result == 'quit':
                cv2.destroyAllWindows()
                ui.enable_sleep()
                sys.exit(0)
            return main()
        elif action == 'manage':
            print("Gestion à implémenter")
            cv2.destroyAllWindows()
            return main()
        
        # Enrollment workflow
        ui.camera_selection_screen()
        while ui.current_step == 'camera_selection':
            cv2.waitKey(100)
        
        ui.username_input_screen()
        while ui.current_step == 'username_input':
            cv2.waitKey(100)
        
        ui.confirm_start_screen()
        while ui.current_step == 'confirm_start':
            cv2.waitKey(100)
        
        cmd = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "enroll_landmarks.py"),
            ui.username,
            "--camera", str(ui.selected_camera)
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ui.animate_loading(process)
        stdout, stderr = process.communicate()
        output = stdout + stderr
        parsed_result = ui.parse_validation_output(output)
        action = ui.final_result_screen(parsed_result)
        
        cv2.destroyAllWindows()
        
        if action == 'restart':
            return main()
        elif action == 'validation':
            # Recréer la fenêtre avant d'entrer dans le workflow de validation
            cv2.namedWindow(ui.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(ui.window_name, 720, 1440)
            result = ui.run_validation_workflow()
            if result == 'quit':
                ui.enable_sleep()
                sys.exit(0)
            return main()
        else:
            ui.enable_sleep()
            sys.exit(0)
            
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        ui.enable_sleep()
        sys.exit(0)
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        try:
            cv2.destroyAllWindows()
            ui.enable_sleep()
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
