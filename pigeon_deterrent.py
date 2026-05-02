import time
import os
import cv2
from datetime import datetime, timedelta
from ultralytics import YOLO

# ==========================================
# KONFIGURATION
# ==========================================
COOLDOWN_TIME = 15            # Sekunden warten nach dem Sprühen
IMAGE_DIR = "erkannte_tauben" # Ordner für die Beweisfotos
DAYS_TO_KEEP_IMAGES = 7       # Wie lange sollen Bilder gespeichert werden?
SHOW_LIVE_PREVIEW = True      # Zeigt ein Fenster mit dem Kamera-Bild an (mit 'q' schließen)

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

# Kamera initialisieren (0 ist meist die Standard-Webcam am Laptop)
cap = cv2.VideoCapture(0)
time.sleep(2) 

last_spray_time = 0

print("🦅 Tauben-Abwehr-System gestartet!")
if SHOW_LIVE_PREVIEW:
    print("Drücke die Taste 'q' im Live-Bild-Fenster, um zu beenden.")
else:
    print("Drücke STRG+C im Terminal zum Beenden.")

# ==========================================
# HAUPTSCHLEIFE
# ==========================================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Abrufen des Kamerabildes.")
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
        if SHOW_LIVE_PREVIEW:
            cv2.imshow("Tauben-Abwehr Live", annotated_frame)
            # waitKey(2000) wartet 2 Sekunden und hält das Fenster in der Zeit aktiv
            if cv2.waitKey(2000) & 0xFF == ord('q'):
                print("\n'q' gedrückt. System wird beendet...")
                break
        else:
            time.sleep(2)

except KeyboardInterrupt:
    print("\nSystem wird beendet...")
finally:
    cap.release()
    if SHOW_LIVE_PREVIEW:
        cv2.destroyAllWindows()
    print("Kamera freigegeben.")
