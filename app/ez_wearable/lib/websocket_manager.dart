import 'dart:async';
import 'package:web_socket_channel/io.dart';
import 'package:flutter/foundation.dart';
import 'connection_state.dart' as conn_state;

const websocketServerUrl = 'ws://192.168.0.169:8081';

class WebSocketManager extends ChangeNotifier {
  static final WebSocketManager _instance = WebSocketManager._internal();
  factory WebSocketManager() => _instance;
  WebSocketManager._internal();

  bool _isConnecting = false;
  bool get isWebSocketConnected => _webSocketChannel != null;
  IOWebSocketChannel? _webSocketChannel;
  Timer? _reconnectTimer;
  DateTime? _lastAck;
  DateTime? _lastHeartbeat;

  bool get isConnecting => _isConnecting;
  DateTime? get lastHeartbeat => _lastHeartbeat;

  conn_state.ConnectionState _state = conn_state.ConnectionState.disconnected;
  conn_state.ConnectionState get state => _state;

  Future<bool> toggleConnection() async {
    if (_state == conn_state.ConnectionState.connected) {
      await disconnectWebSocket();
    } else {
      await tryConnectWebSocket();
    }
    return _state == conn_state.ConnectionState.connected;
  }

  Future<void> tryConnectWebSocket() async {
    if (_isConnecting || isWebSocketConnected) return;

    _setState(conn_state.ConnectionState.connecting);

    try {
      _webSocketChannel = IOWebSocketChannel.connect(
        websocketServerUrl,
        pingInterval: const Duration(seconds: 5),
        connectTimeout: const Duration(seconds: 10),
      );

      _webSocketChannel!.stream.listen(
        (event) {
          if (event == 'heartbeat') {
            handleHeartbeat();
          } else if (event == 'ack') {
            handleAck();
          } else {
            handleMessage(event);
          }
          _setState(conn_state.ConnectionState.connected);
        },
        onError: (error) async {
          print('WebSocket error: $error');
          await _scheduleReconnect();
        },
        onDone: () async {
          print('WebSocket connection closed');
          await _scheduleReconnect();
        },
      );
    } catch (e) {
      print('Error connecting to WebSocket: $e');
      await _scheduleReconnect();
    }
  }

  Future<void> disconnectWebSocket() async {
    _reconnectTimer?.cancel();
    if (_webSocketChannel != null) {
      await _webSocketChannel!.sink.close();
      _webSocketChannel = null;
    }
    _setState(conn_state.ConnectionState.disconnected);
    notifyListeners();
  }

  void sendMessage(List<int> message) {
    if (_webSocketChannel != null) {
      _webSocketChannel!.sink.add(message);
    }
  }

  void handleHeartbeat() {
    _lastHeartbeat = DateTime.now();
    print('Heartbeat received at: $_lastHeartbeat');
    notifyListeners();
  }

  void handleAck() {
    _lastAck = DateTime.now();
    print('Acknowledgment received at: $_lastAck');
  }

  void handleMessage(dynamic message) {
    print('Received message: $message');
  }

  Future<void> _scheduleReconnect() async {
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 10), () async {
      print('Attempting to reconnect WebSocket');
      await tryConnectWebSocket();
    });
  }

  void _setState(conn_state.ConnectionState newState) {
    _state = newState;
    _isConnecting = (newState == conn_state.ConnectionState.connecting);
    notifyListeners();
  }

  Future<void> cancelConnection() async {
    if (state == conn_state.ConnectionState.connecting) {
      await disconnectWebSocket();
      _setState(conn_state.ConnectionState.disconnected);
    }
  }
}
