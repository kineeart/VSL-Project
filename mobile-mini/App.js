import React, { useState } from 'react';
import { SafeAreaView, View, Text, TouchableOpacity, StyleSheet, TextInput, ScrollView } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';

const DEFAULT_BACKEND_HOST = process.env.EXPO_PUBLIC_BACKEND_HOST || '192.168.123.9:8000';
const LOCK_BACKEND_HOST = (process.env.EXPO_PUBLIC_LOCK_BACKEND_HOST || 'false').toLowerCase() === 'true';

function toHttpPredictUrl(hostInput) {
  const host = (hostInput || '').trim();
  if (!host) {
    return '';
  }
  const noTrail = host.replace(/\/$/, '');
  if (host.startsWith('http://')) {
    return `${noTrail}/api/predict/video`;
  }
  if (host.startsWith('https://')) {
    return `${noTrail}/api/predict/video`;
  }
  return `http://${noTrail}/api/predict/video`;
}

export default function App() {
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [debug, setDebug] = useState(null);
  const [error, setError] = useState('');
  const [backendHost, setBackendHost] = useState(DEFAULT_BACKEND_HOST);
  const [videoName, setVideoName] = useState('');

  const resetResult = () => {
    setPredictions([]);
    setDebug(null);
    setError('');
  };

  const predictFromVideo = async (asset) => {
    const url = toHttpPredictUrl(backendHost);
    if (!url) {
      setError('Backend host is empty.');
      return;
    }
    if (!asset?.uri) {
      setError('No video selected.');
      return;
    }

    setLoading(true);
    resetResult();
    setVideoName(asset.fileName || `video-${Date.now()}.mp4`);

    try {
      const form = new FormData();
      form.append('file', {
        uri: asset.uri,
        name: asset.fileName || `capture-${Date.now()}.mp4`,
        type: asset.mimeType || 'video/mp4',
      });

      const res = await fetch(url, { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || 'Prediction failed.');
        return;
      }
      setPredictions(Array.isArray(data.predictions) ? data.predictions : []);
      setDebug(data.debug || null);
    } catch (e) {
      setError(`Upload failed: ${String(e)}`);
    } finally {
      setLoading(false);
    }
  };

  const recordVideo = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      setError('Camera permission denied.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: false,
      videoMaxDuration: 12,
      quality: 0.7,
    });
    if (!result.canceled && result.assets?.length) {
      await predictFromVideo(result.assets[0]);
    }
  };

  const pickVideo = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: false,
      quality: 0.7,
    });
    if (!result.canceled && result.assets?.length) {
      await predictFromVideo(result.assets[0]);
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="dark" />
      <Text style={styles.title}>VSL Mini (Mobile)</Text>

      <View style={styles.hostRow}>
        <Text style={styles.hostLabel}>Backend host:</Text>
        <TextInput
          value={backendHost}
          onChangeText={setBackendHost}
          style={styles.hostInput}
          autoCapitalize="none"
          autoCorrect={false}
          editable={!LOCK_BACKEND_HOST}
        />
        {LOCK_BACKEND_HOST ? (
          <Text style={styles.hostHint}>Locked by build config</Text>
        ) : null}
      </View>

      <View style={styles.modeHintWrap}>
        <Text style={styles.modeHintTitle}>Video mode</Text>
        <Text style={styles.modeHintText}>Quay video ngan hoac chon video co san, app se gui video len backend de lay top 5.</Text>
        {videoName ? <Text style={styles.videoName}>Video: {videoName}</Text> : null}
      </View>

      <View style={styles.actions}>
        <TouchableOpacity onPress={recordVideo} style={[styles.btn, styles.btnStart]} disabled={loading}>
          <Text style={styles.btnText}>{loading ? 'Processing...' : 'Record + Predict'}</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={pickVideo} style={[styles.btn, styles.btnStop]} disabled={loading}>
          <Text style={styles.btnText}>{loading ? 'Processing...' : 'Pick Video'}</Text>
        </TouchableOpacity>
      </View>

      {error ? <Text style={styles.error}>{error}</Text> : null}

      <ScrollView style={styles.panel}>
        <Text style={styles.panelTitle}>Top Predictions</Text>
        {predictions.length === 0 ? (
          <Text style={styles.muted}>Waiting prediction...</Text>
        ) : predictions.map((p, idx) => (
          <View key={`${p.label}-${idx}`} style={styles.predRow}>
            <Text style={styles.predLabel}>{p.label}</Text>
            <Text style={styles.predPct}>{(p.confidence * 100).toFixed(1)}%</Text>
          </View>
        ))}

        {debug && (
          <View style={styles.debugBox}>
            <Text style={styles.panelTitle}>Debug</Text>
            <Text style={styles.debugText}>reason: {debug.reason}</Text>
            <Text style={styles.debugText}>margin: {Number(debug.margin || 0).toFixed(3)}</Text>
            <Text style={styles.debugText} numberOfLines={4}>raw: {(debug.raw_topk || []).map(x => `${x.label}:${(x.confidence * 100).toFixed(1)}%`).join(' | ')}</Text>
            <Text style={styles.debugText} numberOfLines={4}>smooth: {(debug.smooth_topk || []).map(x => `${x.label}:${(x.confidence * 100).toFixed(1)}%`).join(' | ')}</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#f6f7fb', padding: 12 },
  title: { fontSize: 20, fontWeight: '700', marginBottom: 8 },
  hostRow: { marginBottom: 8 },
  hostLabel: { fontSize: 12, color: '#555' },
  hostInput: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    marginTop: 4,
  },
  hostHint: { fontSize: 11, color: '#666', marginTop: 4 },
  modeHintWrap: { backgroundColor: '#fff', borderRadius: 10, padding: 12, borderWidth: 1, borderColor: '#e6e6e6' },
  modeHintTitle: { fontWeight: '700', marginBottom: 4 },
  modeHintText: { color: '#555' },
  videoName: { marginTop: 8, color: '#333', fontSize: 12 },
  actions: { flexDirection: 'row', justifyContent: 'center', marginTop: 10, marginBottom: 8 },
  btn: { borderRadius: 10, paddingVertical: 10, paddingHorizontal: 24 },
  btnStart: { backgroundColor: '#1f7a3f' },
  btnStop: { backgroundColor: '#b03535' },
  btnText: { color: '#fff', fontWeight: '700' },
  error: { color: '#b00020', marginBottom: 6 },
  panel: { marginTop: 6, backgroundColor: '#fff', borderRadius: 10, padding: 10 },
  panelTitle: { fontWeight: '700', marginBottom: 6 },
  muted: { color: '#666' },
  predRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4, borderBottomWidth: 1, borderBottomColor: '#f0f0f0' },
  predLabel: { flex: 1, paddingRight: 8 },
  predPct: { width: 64, textAlign: 'right', fontWeight: '600' },
  debugBox: { marginTop: 10, borderTopWidth: 1, borderTopColor: '#ececec', paddingTop: 8 },
  debugText: { fontSize: 12, color: '#555', marginBottom: 4 },
});
