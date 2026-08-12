import cv2
import os

def extract_frames(video_path, output_folder, fps=10):
    # VideoCapture objesini oluştur
    cap = cv2.VideoCapture(video_path)
    
    # Çıktı klasörünü oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)  # Saniyede 10 kare almak için aralık
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Belirlenen aralıklarla kareleri kaydet
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Tüm kareler {output_folder} klasörüne kaydedildi.")

# Kullanım örneği
if __name__ == "__main__":
    video_path = "video.mp4"  # Videonun yolu
    output_folder = "extracted_frames"  # Çıktı klasörü
    extract_frames(video_path, output_folder)
