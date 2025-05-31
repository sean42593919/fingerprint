import cv2
import numpy as np

# 圖片讀取
img_path = r"C:\Users\sean4\OneDrive\桌面\code\python\fingerprint\card_with_fingerprint.png" #put your file address
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

# 滑鼠框選指紋區域
roi = cv2.selectROI("請框選指紋區域", img, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()
x, y, w, h = roi
cropped = img[y:y+h, x:x+w]

# 灰階
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
cv2.imwrite('fingerprint_gray.png', gray)

# 銳化
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
cv2.imwrite('fingerprint_sharpened.png', sharpened)

# 自適應門檻值
thresh_adaptive = cv2.adaptiveThreshold(
    sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('fingerprint_thresh_adaptive.png', thresh_adaptive)

# 形態學處理：先開再閉
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
opened = cv2.morphologyEx(thresh_adaptive, cv2.MORPH_OPEN, kernel)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('fingerprint_cleaned.png', cleaned)

# 預覽最終結果
cv2.imshow("Final", cleaned)
cv2.waitKey(0)
cv2.destroyAllWindows()