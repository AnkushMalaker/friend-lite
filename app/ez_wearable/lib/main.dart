import 'package:flutter/material.dart';
import 'bluetooth_page.dart';
import 'websocket_page.dart';
import 'bluetooth_connection_manager.dart';
import 'connection_state.dart' as conn_state;
import 'websocket_manager.dart';
import 'package:flutter/rendering.dart';

void main() {
  // debugPaintSizeEnabled = true; // Uncomment for debugging layout issues
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EZ Wearable',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'EZ Wearable Home'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final BluetoothConnectionManager _bleConnectionManager =
      BluetoothConnectionManager();
  final WebSocketManager _webSocketManager = WebSocketManager();

  conn_state.ConnectionState _bluetoothStatus =
      conn_state.ConnectionState.disconnected;
  conn_state.ConnectionState _webSocketStatus =
      conn_state.ConnectionState.disconnected;
  DateTime? _lastHeartbeat;

  bool _isStreaming = false; // Added

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _bleConnectionManager.addListener(_updateBleState);
      _webSocketManager.addListener(_updateWebSocketState);
      _webSocketManager.addListener(_updateHeartbeat);
    });
  }

  @override
  void dispose() {
    _bleConnectionManager.removeListener(_updateBleState);
    _webSocketManager.removeListener(_updateWebSocketState);
    _webSocketManager.removeListener(_updateHeartbeat);
    super.dispose();
  }

  void _updateBleState() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        setState(() {
          _bluetoothStatus = _bleConnectionManager.state;
        });
      } else {
        print('THIS BLE state IS NOT MOUNTED');
      }
    });
  }

  void _updateWebSocketState() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        setState(() {
          _webSocketStatus = _webSocketManager.state;
        });
      } else {
        print('THIS WEB SOCKET state IS NOT MOUNTED');
      }
    });
  }

  void _updateHeartbeat() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        setState(() {
          _lastHeartbeat = _webSocketManager.lastHeartbeat;
        });
      } else {
        print('THIS heartbeat IS NOT MOUNTED');
      }
    });
  }

  void _updateBleConnectionStatus(conn_state.ConnectionState status) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        setState(() {
          _bluetoothStatus = status;
        });
      } else {
        print('THIS BLE status IS NOT MOUNTED');
      }
    });
  }

  void _updateWebSocketConnectionStatus(conn_state.ConnectionState status) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        setState(() {
          _webSocketStatus = status;
        });
      } else {
        print('THIS WEB SOCKET status IS NOT MOUNTED');
      }
    });
  }

  // Implemented
  void _toggleStreaming() {
    if (!_isStreaming) {
      // Attempt to start streaming
      if (_bluetoothStatus == conn_state.ConnectionState.connected &&
          _webSocketStatus == conn_state.ConnectionState.connected) {
        setState(() {
          _isStreaming = true;
        });
        _bleConnectionManager.setStreaming(true);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Streaming started')),
        );
      } else {
        // Show error if connections are not established
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text(
                  'Cannot start streaming. Ensure both Bluetooth and WebSocket are connected.')),
        );
      }
    } else {
      // Stop streaming
      setState(() {
        _isStreaming = false;
      });
      _bleConnectionManager.setStreaming(false);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Streaming stopped')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          // mainAxisSize: MainAxisSize.min, // Optional
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Bluetooth Connection Card
            Card(
              elevation: 4,
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.bluetooth,
                      color: _bluetoothStatus ==
                              conn_state.ConnectionState.connected
                          ? Colors.green
                          : Colors.red,
                      size: 40,
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Bluetooth Device Connection',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Status: ${_bluetoothStatus.name}',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                          Text(
                            'Device: ${_bleConnectionManager.connectedDeviceName ?? 'No device'}',
                            style: Theme.of(context).textTheme.bodySmall,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            // WebSocket Connection Card
            Card(
              elevation: 4,
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.web,
                      color: _webSocketStatus ==
                              conn_state.ConnectionState.connected
                          ? Colors.green
                          : Colors.red,
                      size: 40,
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'WebSocket Connection',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Status: ${_webSocketStatus.name}',
                            style: Theme.of(context).textTheme.bodyMedium,
                          ),
                          Text(
                            'Last Heartbeat: ${_lastHeartbeat?.toLocal().toString() ?? 'N/A'}',
                            style: Theme.of(context).textTheme.bodySmall,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 32),
            // Action Buttons
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => BluetoothPage(
                      connectionManager: _bleConnectionManager,
                      onConnectionStatusChanged: _updateBleConnectionStatus,
                    ),
                  ),
                );
              },
              child: const Text('Open Bluetooth Page'),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => WebSocketPage(
                      connectionManager: _webSocketManager,
                      onConnectionStatusChanged:
                          _updateWebSocketConnectionStatus,
                    ),
                  ),
                );
              },
              child: const Text('Open WebSocket Page'),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _toggleStreaming,
              child: Text(_isStreaming ? 'Stop Streaming' : 'Start Streaming'),
              style: ElevatedButton.styleFrom(
                backgroundColor: _isStreaming ? Colors.red : Colors.blue,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
