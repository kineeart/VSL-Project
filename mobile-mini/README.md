# VSL Mobile Mini (Expo)

Mini app mobile de demo cung flow voi web hien tai:
- Camera live
- Top-k predictions
- Confidence
- Debug raw/smooth

## 1) Prerequisites

- Node.js 18+
- Expo CLI (`npx expo`)
- Neu dung backend cloud/public domain thi khong can localhost/LAN

## 2) Run (development)

```bash
cd mobile-mini
npm install
npm run start
```

Quet QR bang Expo Go, sau do nhap Backend host theo dang:

192.168.x.x:8000

Hoac backend public:

https://api.your-domain.com

## 3) Build APK (khong can localhost)

1. Tao file .env tu .env.example va set backend host public.
2. Dang nhap Expo account: npx eas login
3. Chay build:

build-apk.bat

Hoac:

npx eas build --platform android --profile preview

Ket qua: APK download link tren EAS dashboard.

## 4) Luu y

- Mobile app dang goi websocket backend hien tai (`/ws/predict`), nen web va mobile co the dung song song.
- Neu khong nhan duoc prediction, kiem tra firewall va dia chi LAN host.
- Day la MVP de demo NCKH; ban co the nang cap sang on-device inference khi artifacts da toi uu.

## 5) Build config env

- EXPO_PUBLIC_BACKEND_HOST: backend host/domain
- EXPO_PUBLIC_LOCK_BACKEND_HOST=true: khoa o nhap host trong app
