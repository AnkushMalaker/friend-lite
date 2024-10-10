import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_channel/io.dart';

const websocket_server_url = 'ws://192.168.0.169:8081';

class BluetoothPage extends StatefulWidget {
  const BluetoothPage({super.key});

  @override
  _BluetoothPageState createState() => _BluetoothPageState();
}

class _BluetoothPageState extends State<BluetoothPage> {
  List<ScanResult> devicesList = [];
  bool _isScanning = false;
  String _statusMessage = '';
  BluetoothDevice? _connectedDevice;
  StreamSubscription? _audioStreamSubscription;
  IOWebSocketChannel? _webSocketChannel;

  static const String friendServiceUuid =
      '19b10000-e8f2-537e-4f6c-d104768a1214';
  static const String audioDataStreamCharacteristicUuid =
      '19b10001-e8f2-537e-4f6c-d104768a1214';
  static const String audioCodecCharacteristicUuid =
      '19b10002-e8f2-537e-4f6c-d104768a1214';

  @override
  void initState() {
    super.initState();
    _checkBluetoothPermission();
  }

  @override
  void dispose() {
    _audioStreamSubscription?.cancel();
    _webSocketChannel?.sink.close();
    super.dispose();
  }

  Future<void> _checkBluetoothPermission() async {
    var bluetoothStatus = await Permission.bluetooth.status;
    var locationStatus = await Permission.location.status;

    if (bluetoothStatus.isDenied || locationStatus.isDenied) {
      await Permission.bluetooth.request();
      await Permission.location.request();
    }

    if (await Permission.bluetooth.isGranted &&
        await Permission.location.isGranted) {
      _startScan();
    } else {
      setState(() {
        _statusMessage = 'Bluetooth or Location permission denied';
      });
    }
  }

  void _startScan() async {
    setState(() {
      devicesList.clear();
      _isScanning = true;
      _statusMessage = 'Scanning...';
    });

    try {
      await FlutterBluePlus.startScan(timeout: const Duration(seconds: 4));

      FlutterBluePlus.scanResults.listen((results) {
        setState(() {
          devicesList = results;
          _statusMessage = 'Found ${devicesList.length} devices';
        });
      }, onError: (error) {
        setState(() {
          _statusMessage = 'Scan error: $error';
        });
      });

      await Future.delayed(const Duration(seconds: 4));
    } catch (e) {
      setState(() {
        _statusMessage = 'Error starting scan: $e';
      });
    } finally {
      setState(() {
        _isScanning = false;
      });
    }
  }

  Future<void> _connectToDevice(BluetoothDevice device) async {
    if (_connectedDevice != null && _connectedDevice!.id == device.id) {
      // If tapping on the already connected device, disconnect it
      await _disconnectBluetoothDevice();
    } else {
      try {
        await device.connect();
        setState(() {
          _connectedDevice = device;
          _statusMessage = 'Connected to ${device.name}';
        });

        // Save the connected device ID
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('connectedDeviceId', device.id.id);

        // Request MTU (for Android)
        if (Theme.of(context).platform == TargetPlatform.android) {
          await device.requestMtu(512);
        }

        // Start audio streaming
        _startAudioStreaming(device);
      } catch (e) {
        setState(() {
          _statusMessage = 'Failed to connect: $e';
        });
      }
    }
  }

  Future<void> _startAudioStreaming(BluetoothDevice device) async {
    try {
      // Connect to the local WebSocket
      _webSocketChannel = IOWebSocketChannel.connect(
        websocket_server_url,
        pingInterval: const Duration(seconds: 10),
        connectTimeout: const Duration(seconds: 30),
      );

      final services = await device.discoverServices();
      final friendService =
          services.firstWhere((s) => s.uuid.toString() == friendServiceUuid);
      final audioCharacteristic = friendService.characteristics.firstWhere(
          (c) => c.uuid.toString() == audioDataStreamCharacteristicUuid);

      await audioCharacteristic.setNotifyValue(true);

      _audioStreamSubscription =
          audioCharacteristic.lastValueStream.listen((value) {
        if (value.isNotEmpty) {
          // Send the audio data to the WebSocket
          _webSocketChannel?.sink.add(Uint8List.fromList(value));
        }
      });

      setState(() {
        _statusMessage = 'Audio streaming started';
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Failed to start audio streaming: $e';
      });
    }
  }

  void _reconnectWebSocket() {
    if (_webSocketChannel != null) {
      _webSocketChannel!.sink.close();
    }
    _webSocketChannel = IOWebSocketChannel.connect(websocket_server_url);
    setState(() {
      _statusMessage = 'WebSocket reconnected';
    });
  }

  // Add new method to disconnect WebSocket
  void _disconnectWebSocket() {
    if (_webSocketChannel != null) {
      _webSocketChannel!.sink.close();
      _webSocketChannel = null;
      setState(() {
        _statusMessage = 'WebSocket disconnected';
      });
    }
  }

  // Add new method to disconnect Bluetooth device
  Future<void> _disconnectBluetoothDevice() async {
    if (_connectedDevice != null) {
      try {
        await _connectedDevice!.disconnect();
        _audioStreamSubscription?.cancel();
        setState(() {
          _connectedDevice = null;
          _statusMessage = 'Bluetooth device disconnected';
        });
      } catch (e) {
        setState(() {
          _statusMessage = 'Failed to disconnect: $e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Bluetooth Devices'),
      ),
      body: Column(
        children: [
          ElevatedButton(
            onPressed: _isScanning ? null : _startScan,
            child: Text(_isScanning ? 'Scanning...' : 'Scan for Devices'),
          ),
          Text(_statusMessage),
          ElevatedButton(
            onPressed: _reconnectWebSocket,
            child: const Text('Reconnect WebSocket'),
          ),
          ElevatedButton(
            onPressed: _disconnectWebSocket,
            child: const Text('Disconnect WebSocket'),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: devicesList.length,
              itemBuilder: (context, index) {
                final device = devicesList[index].device;
                final isConnected = _connectedDevice?.id == device.id;
                return ListTile(
                  title: Text(device.platformName.isNotEmpty
                      ? device.platformName
                      : 'Unknown Device'),
                  subtitle: Text(device.remoteId.str),
                  trailing: isConnected
                      ? const Icon(Icons.bluetooth_connected)
                      : null,
                  onTap: () => _connectToDevice(device),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
