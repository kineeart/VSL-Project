import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

const resources = {
  vi: {
    translation: {
      appName: 'Nhận dạng Ngôn ngữ Ký hiệu Việt Nam',
      nav: { camera: 'Camera', upload: 'Tải video', training: 'Huấn luyện', status: 'Trạng thái' },
      camera: {
        title: 'Nhận dạng Realtime',
        start: 'Bắt đầu',
        stop: 'Dừng',
        waiting: 'Đang chờ nhận dạng...',
        noModel: 'Chưa có model. Vui lòng huấn luyện trước.',
        detected: 'Nhận dạng được',
        confidence: 'Độ tin cậy'
      },
      upload: {
        title: 'Tải video lên để nhận dạng',
        select: 'Chọn video',
        predict: 'Nhận dạng',
        processing: 'Đang xử lý...',
        result: 'Kết quả'
      },
      training: {
        title: 'Huấn luyện thủ công',
        record: 'Quay video',
        stopRecord: 'Dừng quay',
        label: 'Nhãn hành động',
        labelPlaceholder: 'VD: xin chào, cảm ơn...',
        save: 'Lưu mẫu',
        saved: 'Đã lưu!',
        trainModel: 'Huấn luyện Model',
        trainStart: 'Bắt đầu huấn luyện',
        trainRunning: 'Đang huấn luyện...',
        trainDone: 'Huấn luyện xong!',
        maxVideos: 'Số video tối đa',
        customSamples: 'Mẫu tự tạo'
      },
      status: {
        title: 'Trạng thái hệ thống',
        modelLoaded: 'Model đã tải',
        modelNotLoaded: 'Chưa có model',
        numClasses: 'Số lớp nhận dạng',
        yes: 'Có',
        no: 'Chưa'
      },
      theme: { light: 'Sáng', dark: 'Tối' },
      lang: { vi: 'Tiếng Việt', en: 'English' }
    }
  },
  en: {
    translation: {
      appName: 'Vietnamese Sign Language Recognition',
      nav: { camera: 'Camera', upload: 'Upload', training: 'Training', status: 'Status' },
      camera: {
        title: 'Realtime Recognition',
        start: 'Start',
        stop: 'Stop',
        waiting: 'Waiting for recognition...',
        noModel: 'No model found. Please train first.',
        detected: 'Detected',
        confidence: 'Confidence'
      },
      upload: {
        title: 'Upload video for recognition',
        select: 'Select video',
        predict: 'Recognize',
        processing: 'Processing...',
        result: 'Result'
      },
      training: {
        title: 'Manual Training',
        record: 'Record video',
        stopRecord: 'Stop recording',
        label: 'Action label',
        labelPlaceholder: 'E.g.: hello, thank you...',
        save: 'Save sample',
        saved: 'Saved!',
        trainModel: 'Train Model',
        trainStart: 'Start training',
        trainRunning: 'Training...',
        trainDone: 'Training complete!',
        maxVideos: 'Max videos',
        customSamples: 'Custom samples'
      },
      status: {
        title: 'System Status',
        modelLoaded: 'Model loaded',
        modelNotLoaded: 'No model',
        numClasses: 'Number of classes',
        yes: 'Yes',
        no: 'No'
      },
      theme: { light: 'Light', dark: 'Dark' },
      lang: { vi: 'Tiếng Việt', en: 'English' }
    }
  }
};

i18n.use(LanguageDetector).use(initReactI18next).init({
  resources,
  fallbackLng: 'vi',
  interpolation: { escapeValue: false }
});

export default i18n;
