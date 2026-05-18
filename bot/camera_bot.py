import os

# TensorFlow loglarini o'chirish
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import asyncio

import cv2
import httpx
import numpy as np
import uvicorn
from deepface import DeepFace
from fastapi import FastAPI, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

# ── CONFIGURATION ────────────────────────────────────────────────────────────
BACKEND_BASE_URL = "http://127.0.0.1:8000"  # Asosiy backend URL
BACKEND_CREATE_USER_URL = f"{BACKEND_BASE_URL}/users"

# ── ORIGINAL CAMERA PIPELINE ──────────────────────────────────────────────────
# ... (Sizdagi mavjud `get_weather_condition`, `send_pipeline` funksiyalari o'zgarishsiz qoladi) ...
BACKEND_EVALUATE_URL = f"{BACKEND_BASE_URL}/recognition/evaluate"
FRONTEND_WEBHOOK_URL = "http://127.0.0.1:3000/api/display-greeting"
WEATHER_API_URL = "https://api.openmeteo.com/v1/forecast"
LATITUDE, LONGITUDE, LOCATION_NAME = 41.2995, 69.2401, "Tashkent"
DETECTION_COOLDOWN, last_detection_time = 5, 0


# ── BOT API SERVER (New Endpoint for creating users via image) ───────────────
bot_api = FastAPI(title="Edge Camera Bot Control API")


def l2_normalize(vector: list[float]) -> list[float]:
    """Vektorni L2 normallashtirish (Evklid uzunligini 1.0 ga keltirish)"""
    np_vector = np.array(vector)
    l2_norm = np.linalg.norm(np_vector)
    if l2_norm == 0:
        return vector
    return (np_vector / l2_norm).tolist()


@bot_api.post("/register-user-sync")
async def register_user_from_image(
    full_name: str = Form(...),
    role: str = Form(...),
    image_file: UploadFile = File(...),
):
    """
    Tashqaridan (Frontend/Admin panel) rasm va ma'lumotlarni qabul qiladi.
    Rasmdan 128-D vektor oladi va barchasini asosiy backend'ga POST qiladi.
    """
    try:
        contents = await image_file.read()

        # 1. Kelgan rasmni vaqtinchalik xotiraga o'qish
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            # A. Agar rasm o'lchami juda kichik bo'lsa, uni kattalashtiramiz (Interpolation yordamida)
            h, w, _ = img.shape
            if w < 800 or h < 800:
                scale_factor = 2
                img = cv2.resize(
                    img,
                    (w * scale_factor, h * scale_factor),
                    interpolation=cv2.INTER_CUBIC,
                )

            # B. Raqamli shovqinlarni o'chirish (Denoising) - tasvirdagi "g'ubor" va g'adir-budirlikni yo'qotadi
            img = cv2.fastNlMeansDenoisingColored(
                img, None, h=3, templateWindowSize=7, searchWindowSize=21
            )

            # C. Kontrastni to'g'rilash (Yorug'lik va soyani balanslash)
            # Rasm yorug'ligini chiroyli qilish uchun CLAHE algoritmini qo'llaymiz
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # D. Vaqtinchalik faylga maksimal sifatda (95%+) yozamiz
            temp_filename = f"temp_register_{image_file.filename}"
            cv2.imwrite(temp_filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        else:
            # Agar rasm o'qilmasa, eski usulda yozib turamiz
            temp_filename = f"temp_register_{image_file.filename}"
            with open(temp_filename, "wb") as f:
                f.write(contents)

        # 2. DeepFace orqali rasmdan 128 o'lchamli yuz embeddingini (vektor) olish
        print(f"📸 '{full_name}' uchun yuz vektori hisoblanmoqda...")
        embeddings = DeepFace.represent(
            img_path=temp_filename,
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="retinaface",
        )

        # Vaqtinchalik faylni o'chirib yuboramiz
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        if not embeddings:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": "Rasmda yuz aniqlanmadi. Iltimos boshqa rasm yuklang."
                },
            )

        raw_vector = embeddings[0]["embedding"]
        detected_vector = l2_normalize(raw_vector)

        # 3. Asosiy Backend'ga birinchi USER profile yaratish uchun so'rov yuboramiz
        async with httpx.AsyncClient() as client:
            user_payload = {"full_name": full_name, "role": role}
            print("🚀 Backend'da user yaratilmoqda...")
            user_resp = await client.post(
                BACKEND_CREATE_USER_URL, json=user_payload, timeout=5.0
            )

            if user_resp.status_code != 201:
                return JSONResponse(
                    status_code=user_resp.status_code,
                    content={"detail": f"Backend user yaratmadi: {user_resp.text}"},
                )

            created_user = user_resp.json()
            user_id = created_user["id"]  # Backend yaratib bergan unikal ID

            # 4. Endi olingan ID bo'yicha yuz vektorini upsert (save) endpointiga yuboramiz
            vector_payload = {"embedding": detected_vector}
            backend_vector_url = f"{BACKEND_BASE_URL}/users/{user_id}/vectors"
            print(f"🧬 Vektor user_id={user_id} profiliga biriktirilmoqda...")
            vector_resp = await client.post(
                backend_vector_url, json=vector_payload, timeout=5.0
            )

            if vector_resp.status_code in [200, 201]:
                print(f"✅ Foydalanuvchi muvaffaqiyatli ro'yxatdan o'tdi: {full_name}")
                return {
                    "status": "success",
                    "user_id": user_id,
                    "full_name": full_name,
                    "message": "Foydalanuvchi va uning yuz vektori backendga muvaffaqiyatli saqlandi.",
                }
            else:
                return JSONResponse(
                    status_code=vector_resp.status_code,
                    content={"detail": f"Vektorni saqlashda xato: {vector_resp.text}"},
                )

    except Exception as e:
        # Xatolik bo'lsa vaqtinchalik fayl o'chib ketishini ta'minlash
        if "temp_filename" in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)
        print(f"🚨 Ro'yxatga olishda xatolik: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Bot ichki xatoligi: {str(e)}"},
        )


# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
async def get_weather_condition() -> str:
    try:
        async with httpx.AsyncClient() as client:
            params = {
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "current_weather": True,
            }
            response = await client.get(WEATHER_API_URL, params=params, timeout=3.0)

            if response.status_code == 200:
                data = response.json()
                temp = data["current_weather"]["temperature"]
                code = data["current_weather"]["weathercode"]

                # 1. Agar harorat juda issiq bo'lsa va yomg'ir/qor yog'mayotgan bo'lsa
                if temp >= 30.0 and code in [0, 1, 2, 3]:
                    return "hot"

                # 2. WMO kodlarini siz so'ragan formatga o'tkazish
                if code == 0:
                    return "sunny"  # Mutloq ochiq osmon
                elif code in [1, 2, 3]:
                    return "cloudy"  # Qisman yoki to'liq bulutli
                elif code in [
                    45,
                    48,
                    51,
                    53,
                    55,
                    56,
                    57,
                    61,
                    63,
                    65,
                    66,
                    67,
                    80,
                    81,
                    82,
                    95,
                    96,
                    99,
                ]:
                    return "rainy"  # Tuman, mayda yomg'ir, jala va momaqaldiroq
                elif code in [71, 73, 75, 77, 85, 86]:
                    return "snowy"  # Qor yog'ishi
                else:
                    return "sunny"  # Default holat (boshqa kodlar uchun)

    except Exception as e:
        print(f"⚠️ Ob-havoni olishda xatolik: {e}")

    # Xatolik yuz berganda standart holat sifatida "sunny" qaytariladi
    return "sunny"


async def send_pipeline(detected_vector: list[float], current_emotion: str) -> None:
    """Backendga yuborish va natijani Frontendga uzatish zanjiri"""
    weather_info = await get_weather_condition()
    print(detected_vector)

    payload = {
        "detected_vector": detected_vector,
        "weather_condition": f"{LOCATION_NAME}: {weather_info}",
        "current_emotion": current_emotion,
    }

    async with httpx.AsyncClient() as client:
        try:
            # 1. Backend API (Recognition & Greeting Engine) ga so'rov yuborish
            print("🚀 Backendga so'rov jo'natilmoqda...")
            backend_resp = await client.post(
                BACKEND_EVALUATE_URL, json=payload, timeout=5.0
            )

            if backend_resp.status_code == 200:
                backend_data = backend_resp.json()
                print(
                    f"🎯 Odam aniqlandi: {backend_data.get('full_name')} ({backend_data.get('role')})"
                )

                # 2. Kelgan javobni Frontendga POST qilish
                print("📺 Frontendga ma'lumot uzatilmoqda...")
                frontend_resp = await client.post(
                    FRONTEND_WEBHOOK_URL, json=backend_data, timeout=3.0
                )
                if frontend_resp.status_code in [200, 201, 204]:
                    print("✅ Frontend muvaffaqiyatli yangilandi.")
                else:
                    print(f"❌ Frontend xato qaytardi: {frontend_resp.status_code}")
            else:
                print(f"❌ Backend xato qaytardi: {backend_resp.status_code}")

        except Exception as e:
            print(f"🚨 Pipeline uzatishda xatolik: {e}")


# ── MAIN CORE PIPELINE ────────────────────────────────────────────────────────
async def main_camera_loop():
    global last_detection_time

    # Kamerani ochish (0 - ichki kamera, yoki RTSP video oqim URL manzili)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamerani ochib bo'lmadi!")
        return

    print("📸 Real-time Kamera kuzatuvi boshlandi. Fondagi bot faol...")

    loop = asyncio.get_running_loop()

    while True:
        # OpenCV kadrlarni o'qishi blokirovka qiluvchi (blocking) operatsiya bo'lgani uchun
        # uni thread_pool ichida bajaramiz, asinxron mantiq buzilmasligi uchun
        ret, frame = await loop.run_in_executor(None, cap.read)
        if not ret:
            break

        current_time = asyncio.get_event_loop().time()

        # Har 2 soniyada kadrda yuz borligini tekshirish (Kamera qotmasligi va CPU qizib ketmasligi uchun)
        if current_time - last_detection_time > DETECTION_COOLDOWN:
            try:
                # DeepFace orqali yuzni tahlil qilish (Vektor + Emotsiya)
                # Model sifatida loyihangiz o'lchamiga qarab 'Facenet' (128-D) yoki boshqasini tanlang
                analysis = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    enforce_detection=True,
                    silent=True,
                )

                embeddings = DeepFace.represent(
                    img_path=frame,
                    model_name="ArcFace",
                    enforce_detection=True,
                    detector_backend="retinaface",
                )

                if embeddings and analysis:
                    # Ma'lumotlarni ajratib olamiz
                    raw_vector = embeddings[0]["embedding"]
                    # 🎯 Kameradan olingan kadr vektorini ham L2 normallashtiramiz:
                    detected_vector = l2_normalize(raw_vector)
                    current_emotion = analysis[0]["dominant_emotion"]

                    last_detection_time = current_time

                    # Backend va Frontendga yuborish vazifasini fonda (background) ishga tushiramiz
                    # Bu kamerani keyingi kadrlarni o'qishda davom etishiga imkon beradi
                    asyncio.create_task(send_pipeline(detected_vector, current_emotion))

            except ValueError:
                # Kadrda yuz topilmasa DeepFace xato tashlaydi, uni shunchaki o'tkazib yuboramiz
                pass

        # Ekran oynasini ko'rsatish (Lokalda kuzatish uchun, serverda buni o'chirib qo'yish mumkin)
        cv2.imshow("Smart Greeting System - Edge Camera", frame)

        # 'q' bosilsa skript to'xtaydi
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        await asyncio.sleep(0.01)  # CPU yuklamasini muvozanatlash

    cap.release()
    cv2.destroyAllWindows()


# ── SYSTEM ORCHESTRATION ──────────────────────────────────────────────────────
async def start_bot_api():
    """Bot ichidagi API serverni ishga tushirish (8001-portda)"""
    config = uvicorn.Config(bot_api, host="127.0.0.1", port=8001, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """Kamera oqimi va Bot API serverini parallel (Asynchronous) ishga tushirish"""
    # Ikkala vazifani ham bitta event loop ichida parallel boshlaymiz
    await asyncio.gather(start_bot_api(), main_camera_loop())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Kamera boti to'xtatildi.")
