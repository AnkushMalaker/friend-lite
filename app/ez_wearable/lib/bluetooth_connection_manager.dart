import 'dart:async';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:permission_handler/permission_handler.dart';
import 'connection_state.dart';
import 'package:flutter/foundation.dart';
import 'websocket_manager.dart'; // Imported

class BluetoothConnectionManager extends ChangeNotifier {
  static const String _friendServiceUuid =
      '19b10000-e8f2-537e-4f6c-d104768a1214';
  static const String _audioDataStreamCharacteristicUuid =
      '19b10001-e8f2-537e-4f6c-d104768a1214';

  BluetoothDevice? _connectedDevice;
  StreamSubscription? _audioStreamSubscription;
  bool _permissionsGranted = false;
  ConnectionState _state = ConnectionState.disconnected;

  ConnectionState get state => _state;
  bool get isBluetoothConnected => _connectedDevice != null;
  String? get connectedDeviceName => _connectedDevice?.platformName;
  bool get permissionsGranted => _permissionsGranted;

  final WebSocketManager _webSocketManager = WebSocketManager(); // Added

  bool _isStreaming = false; // Added

  Future<bool> toggleConnection(BluetoothDevice device) async {
    if (_state == ConnectionState.connected) {
      await disconnectBluetoothDevice();
    } else {
      await connectToDevice(device);
    }
    return _state == ConnectionState.connected;
  }

  Future<void> connectToDevice(BluetoothDevice device) async {
    if (_state != ConnectionState.disconnected) {
      await disconnectBluetoothDevice();
    }

    try {
      _setState(ConnectionState.connecting);
      await device.connect();
      _connectedDevice = device;

      await _saveConnectedDeviceId(device.remoteId.str);
      await _requestMtu(device);
      await _startAudioStreaming(device);

      _setState(ConnectionState.connected);
    } catch (e) {
      print('Failed to connect: $e');
      _setState(ConnectionState.error);
    }
  }

  Future<void> disconnectBluetoothDevice() async {
    if (_connectedDevice != null) {
      try {
        await _connectedDevice!.disconnect();
        await _audioStreamSubscription?.cancel();
        _connectedDevice = null;
        _setState(ConnectionState.disconnected);
      } catch (e) {
        print('Failed to disconnect: $e');
        _setState(ConnectionState.error);
      }
    }
  }

  void _setState(ConnectionState newState) {
    _state = newState;
    notifyListeners();
  }

  Future<void> _saveConnectedDeviceId(String deviceId) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('connectedDeviceId', deviceId);
  }

  Future<void> _requestMtu(BluetoothDevice device) async {
    if (device.platformName == 'Android') {
      await device.requestMtu(512);
    }
  }

  Future<void> _startAudioStreaming(BluetoothDevice device) async {
    try {
      final services = await device.discoverServices();
      final friendService =
          services.firstWhere((s) => s.uuid.toString() == _friendServiceUuid);
      final audioCharacteristic = friendService.characteristics.firstWhere(
          (c) => c.uuid.toString() == _audioDataStreamCharacteristicUuid);

      await audioCharacteristic.setNotifyValue(true);

      _audioStreamSubscription =
          audioCharacteristic.lastValueStream.listen(_handleAudioData);
      _setState(ConnectionState.connected);
    } catch (e) {
      print('Failed to start audio streaming: $e');
      _setState(ConnectionState.error);
    }
  }

  void _handleAudioData(List<int> value) {
    if (_isStreaming && _webSocketManager.state == ConnectionState.connected) {
      print('Sending audio data to WebSocket');
      _webSocketManager.sendMessage(value); // Send raw bytes
    }
  }

  void _handleWebSocketError(dynamic error) {
    print('WebSocket error: $error');
    _setState(ConnectionState.error);
  }

  void _handleWebSocketDisconnect() {
    print('WebSocket connection closed');
    _setState(ConnectionState.error);
  }

  void _handleWebSocketAcknowledged() {
    print('Audio streaming active');
  }

  Future<bool> checkAndRequestBluetoothPermissions() async {
    PermissionStatus bluetoothStatus = await Permission.bluetooth.status;
    PermissionStatus locationStatus = await Permission.location.status;

    if (bluetoothStatus.isDenied || locationStatus.isDenied) {
      bluetoothStatus = await Permission.bluetooth.request();
      locationStatus = await Permission.location.request();
    }

    _permissionsGranted = bluetoothStatus.isGranted && locationStatus.isGranted;
    return _permissionsGranted;
  }

  // Added: Method to set streaming state
  void setStreaming(bool streaming) {
    _isStreaming = streaming;
    print("Streaming is now ${streaming ? 'enabled' : 'disabled'}");
    // Optionally, you can notify listeners or perform other actions here
  }
}
