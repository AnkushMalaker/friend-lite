//// plugins/with-ws-fgs.js
const { withAndroidManifest, AndroidConfig } = require('@expo/config-plugins');

/**
 * Adds:
 * - Foreground service type(s) to Notifee's service (Android 14+)
 * - Required FGS + notification permissions if missing
 *
 * Options:
 *   { microphone: boolean }  // true if you also capture mic audio
 */
module.exports = (config, { microphone = false } = {}) =>
  withAndroidManifest(config, (cfg) => {
    const manifest = cfg.modResults;

    // Ensure uses-permission array exists
    manifest.manifest['uses-permission'] ||= [];
    const addPerm = (name) => {
      const exists = manifest.manifest['uses-permission'].some(
        (p) => p.$['android:name'] === name
      );
      if (!exists) {
        manifest.manifest['uses-permission'].push({ $: { 'android:name': name } });
      }
    };

    // Required for FG services and notifications
    addPerm('android.permission.FOREGROUND_SERVICE');
    addPerm('android.permission.FOREGROUND_SERVICE_DATA_SYNC'); // we use this for the WS keep-alive
    addPerm('android.permission.POST_NOTIFICATIONS');
    if (microphone) {
      addPerm('android.permission.FOREGROUND_SERVICE_MICROPHONE');
      // If you actually record audio, you'll also need RECORD_AUDIO elsewhere.
    }

    // Ensure Notifee service exists and set the service type(s)
    const app = AndroidConfig.Manifest.getMainApplicationOrThrow(manifest);
    app.service ||= [];
    const svcName = 'app.notifee.core.ForegroundService';

    let svc = app.service.find((s) => s.$['android:name'] === svcName);
    if (!svc) {
      svc = { $: { 'android:name': svcName, 'android:exported': 'false' } };
      app.service.push(svc);
    }

    const types = ['dataSync'];
    if (microphone) types.push('microphone');
    // Android 14 allows multiple types separated by '|'
    svc.$['android:foregroundServiceType'] = types.join('|');

    return cfg;
  });
