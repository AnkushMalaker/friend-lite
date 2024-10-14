import 'websocket_manager.dart';
import 'package:flutter/material.dart';
import 'connection_state.dart' as conn_state;

class WebSocketPage extends StatefulWidget {
  final WebSocketManager connectionManager;
  final Function(conn_state.ConnectionState) onConnectionStatusChanged;

  const WebSocketPage({
    super.key,
    required this.connectionManager,
    required this.onConnectionStatusChanged,
  });

  @override
  State<WebSocketPage> createState() => _WebSocketPageState();
}

class _WebSocketPageState extends State<WebSocketPage> {
  final TextEditingController _wsUrlController = TextEditingController();

  @override
  void initState() {
    super.initState();
    widget.connectionManager.addListener(_updateState);
    widget.connectionManager.tryConnectWebSocket();
  }

  @override
  void dispose() {
    widget.connectionManager.removeListener(_updateState);
    super.dispose();
  }

  void _updateState() {
    setState(() {});
    widget.onConnectionStatusChanged(widget.connectionManager.state);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('WebSocket'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          TextField(
            controller: _wsUrlController,
            decoration: const InputDecoration(
              hintText: 'Websocket URL (default is ws://192.168.0.169:8081)',
            ),
          ),
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: () async {
              if (widget.connectionManager.state ==
                  conn_state.ConnectionState.connecting) {
                await widget.connectionManager.cancelConnection();
              } else {
                await widget.connectionManager.toggleConnection();
              }
              _updateState();
            },
            child: Text(_getButtonText()),
          ),
          const SizedBox(height: 20),
          Text('Status: ${widget.connectionManager.state.name}'),
          const SizedBox(height: 20),
          Text(
            'Last Heartbeat: ${widget.connectionManager.lastHeartbeat?.toString() ?? 'N/A'}',
          ),
        ],
      ),
    );
  }

  String _getButtonText() {
    switch (widget.connectionManager.state) {
      case conn_state.ConnectionState.connected:
        return 'Disconnect';
      case conn_state.ConnectionState.connecting:
        return 'Cancel';
      default:
        return 'Connect';
    }
  }
}
