import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'bluetooth_connection_manager.dart';
import 'connection_state.dart' as conn_state;

class BluetoothPage extends StatefulWidget {
  final BluetoothConnectionManager connectionManager;
  final Function(conn_state.ConnectionState) onConnectionStatusChanged;

  const BluetoothPage({
    super.key,
    required this.connectionManager,
    required this.onConnectionStatusChanged, // Accept the callback
  });

  @override
  State<BluetoothPage> createState() => _BluetoothPageState();
}

class _BluetoothPageState extends State<BluetoothPage> {
  List<ScanResult> devicesList = [];
  bool _isScanning = false;
  String _statusMessage = '';

  @override
  void initState() {
    super.initState();
    widget.connectionManager
        .checkAndRequestBluetoothPermissions()
        .then((value) {
      if (value) {
        _startScan();
      } else {
        setState(() {
          _statusMessage = 'Bluetooth or Location permission denied';
        });
      }
    });
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
          // New section to display connected devices
          if (widget.connectionManager.isBluetoothConnected)
            Column(
              children: [
                const Text('Connected Devices:'),
                ListTile(
                  title:
                      Text(widget.connectionManager.connectedDeviceName ?? ''),
                  trailing: const Icon(Icons.bluetooth_connected),
                  onTap: () async {
                    // Show toast to confirm disconnect
                    Fluttertoast.showToast(
                        msg:
                            "Disconnecting from ${widget.connectionManager.connectedDeviceName}",
                        toastLength: Toast.LENGTH_SHORT,
                        gravity: ToastGravity.BOTTOM,
                        timeInSecForIosWeb: 1,
                        backgroundColor: Color.fromARGB(255, 212, 124, 41),
                        textColor: Colors.white,
                        fontSize: 16.0);
                    await widget.connectionManager.disconnectBluetoothDevice();
                    _startScan();
                  },
                ),
              ],
            ),
          Expanded(
            child: ListView.builder(
              itemCount: devicesList.length,
              itemBuilder: (context, index) {
                final device = devicesList[index].device;
                final isConnected =
                    widget.connectionManager.isBluetoothConnected &&
                        widget.connectionManager.connectedDeviceName ==
                            device.platformName;
                return ListTile(
                  title: Text(device.platformName.isNotEmpty
                      ? device.platformName
                      : 'Unknown Device'),
                  subtitle: Text(device.remoteId.str),
                  trailing: isConnected
                      ? const Icon(Icons.bluetooth_connected)
                      : null,
                  onTap: () {
                    widget.connectionManager
                        .toggleConnection(device)
                        .then((isConnected) {
                      // Call the callback to update the main page
                      widget.onConnectionStatusChanged(isConnected
                          ? conn_state.ConnectionState.connected
                          : conn_state.ConnectionState.disconnected);
                      setState(() {});
                    });
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
