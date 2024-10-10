# EZ Wearable
Steps:
1. Modified main.dart to have bluetooth connection page
2. Modified android manifest to include bluetooth permissions
3. Added flutter_blue_plus and permission_handler dependencies
4. Created bluetooth_page.dart to handle bluetooth connection and scanning (_checkBluetoothPermission, _startScan, _connectToDevice)
5. Update bluetooth_page.dart within lib/ to handle the actual streaming of data. This also handles the charactaristic, which is basically services defined by a bluetooth device to describe the data that can be streamed. (_startAudioStreaming)
6. Created the websocket server within backend/main.py along with the pyproject.toml file. Implement the websocket and opus decoding using opuslib. First 3 bytes are the configuration packets for the rest of the data. 
