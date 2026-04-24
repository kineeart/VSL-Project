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
        chooseLabel: 'Chọn nhãn trước khi quay',
        record: 'Quay video',
        stopRecord: 'Dừng quay',
        label: 'Nhãn hành động',
        labelPlaceholder: 'VD: xin chào, cảm ơn...',
        quantity: 'Số video trong nhãn',
        duration: 'Thời lượng mỗi video',
        autoRecord: 'Bắt đầu quay tự động',
        autoSaveNote: 'Video sẽ được lưu tự động vào custom_videos, không cần bấm Lưu mẫu.',
        currentLabel: 'Nhãn hiện tại',
        preparing: 'Sẵn sàng trong {{seconds}}s',
        batchProgress: 'Đã quay {{current}}/{{total}} video cho nhãn {{label}}',
        existingLabels: 'Nhãn đã có',
        downloadLabel: 'Tải nhãn',
        deleteLabel: 'Xóa nhãn',
        deleteConfirm: 'Xóa toàn bộ video của nhãn "{{label}}"?',
        batchDone: 'Đã hoàn tất và lưu vào custom_videos',
        recordFailed: 'Không thể quay video. Hãy kiểm tra camera và quyền truy cập.',
        saveFailed: 'Không thể lưu video thứ {{current}}.',
        backendOffline: 'Backend chưa chạy (http://localhost:8000). Hãy chạy start.bat rồi thử lại.',
        downloadFailed: 'Không tải được nhãn này.',
        deleteFailed: 'Không xóa được nhãn này.',
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
        chooseLabel: 'Choose a label before recording',
        record: 'Record video',
        stopRecord: 'Stop recording',
        label: 'Action label',
        labelPlaceholder: 'E.g.: hello, thank you...',
        quantity: 'Number of videos in the label',
        duration: 'Duration per video',
        autoRecord: 'Start auto recording',
        autoSaveNote: 'Each video is saved automatically to custom_videos, no manual Save button needed.',
        currentLabel: 'Current label',
        preparing: 'Ready in {{seconds}}s',
        batchProgress: 'Recorded {{current}}/{{total}} videos for {{label}}',
        existingLabels: 'Existing labels',
        downloadLabel: 'Download label',
        deleteLabel: 'Delete label',
        deleteConfirm: 'Delete all videos for label "{{label}}"?',
        batchDone: 'Finished and saved to custom_videos',
        recordFailed: 'Could not record video. Check camera access.',
        saveFailed: 'Could not save video {{current}}.',
        backendOffline: 'Backend is offline (http://localhost:8000). Start start.bat and try again.',
        downloadFailed: 'Could not download this label.',
        deleteFailed: 'Could not delete this label.',
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
