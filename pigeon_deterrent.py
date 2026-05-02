import time
import os
import cv2
import platform
import subprocess
from datetime import datetime, timedelta
from ultralytics import YOLO

# ==========================================
# KONFIGURATION
# ==========================================
COOLDOWN_TIME = 15            # Sekunden warten nach dem Sprühen
IMAGE_DIR = "erkannte_tauben" # Ordner für die Beweisfotos
DAYS_TO_KEEP_IMAGES = 7       # Wie lange sollen Bilder gespeichert werden?
SHOW_LIVE_PREVIEW = False     # Zeigt ein Fenster mit dem Kamera-Bild an (mit 'q' schließen)

# Automatische Erkennung, ob wir auf Windows oder dem Raspberry Pi sind
IS_WINDOWS = platform.system() == "Windows"

def spritzwasser_an():
    print("💦 RELAIS AN: Pssshhhh! Taube wird nass gemacht!")
    # Hier kommt später der Pi-Code hin (z.B. relais.on())
    time.sleep(2) # 2 Sekunden sprühen
    print("💧 RELAIS AUS.")
    # relais.off()

# ==========================================
# INITIALISIERUNG
# ==========================================
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

print("Lade KI-Modell (YOLOv8 Nano)...")
# Lädt automatisch das Modell herunter beim ersten Start
model = YOLO('yolov8n.pt') 

if IS_WINDOWS:
    print("Starte Windows-Webcam...")
    cap = cv2.VideoCapture(0)
    time.sleep(2) 
else:
    print("Nutze Raspberry Pi Kamera (rpicam-jpeg)...")
    cap = None

last_spray_time = 0

print("🦅 Tauben-Abwehr-System gestartet!")
if SHOW_LIVE_PREVIEW and IS_WINDOWS:
    print("Drücke die Taste 'q' im Live-Bild-Fenster, um zu beenden.")
else:
    print("Drücke STRG+C im Terminal zum Beenden.")

# ==========================================
# HAUPTSCHLEIFE
# ==========================================
try:
    while True:
        if IS_WINDOWS:
            ret, frame = cap.read()
        else:
            # Auf dem Raspberry Pi nutzen wir den nativen Befehl rpicam-jpeg
            try:
                # -t 500 = 500ms Zeit für den Autofokus/Belichtung, --nopreview = kein Overlay
                subprocess.run(["rpicam-jpeg", "-o", "temp_capture.jpg", "-t", "500", "--nopreview"], 
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                frame = cv2.imread("temp_capture.jpg")
                ret = frame is not None
            except Exception as e:
                print(f"Fehler bei rpicam-jpeg: {e}")
                ret = False

        if not ret:
            print("Fehler beim Abrufen des Kamerabildes. Versuche es erneut...")
            time.sleep(2)
            continue

        # KI-Erkennung durchführen (Klasse 14 = 'bird')
        results = model(frame, classes=[14], verbose=False) 
        
        # Bild mit Rahmen versehen (für Live-Bild und Beweisfoto)
        annotated_frame = results[0].plot()

        bird_detected = False
        for result in results:
            if len(result.boxes) > 0: 
                bird_detected = True
                break

        current_time = time.time()
        
        if bird_detected:
            if current_time - last_spray_time > COOLDOWN_TIME:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] TAUBE ERKANNT! Starte Abwehrmaßnahme.")
                
                # Beweisfoto speichern
                filename = os.path.join(IMAGE_DIR, f"taube_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(filename, annotated_frame)
                
                spritzwasser_an()
                
                last_spray_time = time.time()
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Taube im Bild, aber Cooldown ist noch aktiv...")
        
        # Alte Bilder aufräumen
        now = time.time()
        for filename in os.listdir(IMAGE_DIR):
            filepath = os.path.join(IMAGE_DIR, filename)
            if os.path.isfile(filepath):
                if os.stat(filepath).st_mtime < now - (DAYS_TO_KEEP_IMAGES * 86400):
                    os.remove(filepath)
                    print(f"Altes Bild gelöscht: {filename}")

        # Live-Vorschau anzeigen oder einfach warten
        if SHOW_LIVE_PREVIEW and IS_WINDOWS:
            cv2.imshow("Tauben-Abwehr Live", annotated_frame)
            if cv2.waitKey(2000) & 0xFF == ord('q'):
                print("\n'q' gedrückt. System wird beendet...")
                break
        else:
            time.sleep(2)

except KeyboardInterrupt:
    print("\nSystem wird beendet...")
finally:
    if IS_WINDOWS and cap is not None:
        cap.release()
        if SHOW_LIVE_PREVIEW:
            cv2.destroyAllWindows()
    # Temporäres Bild aufräumen
    if not IS_WINDOWS and os.path.exists("temp_capture.jpg"):
        os.remove("temp_capture.jpg")
    print("Kamera freigegeben.")
