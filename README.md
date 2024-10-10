# EZ Wearable
I made this to understand the bluetooth streaming and websocket implementation in the Omi app. Then realized this is a pretty good point to push it since its basic enough that someone else may find it useful. Moving forawrd I want to implement the same or better memory features as in the Omi app, and definitely better overall context handling with a better backend. 
Maybe wakeword etc. 
Wainting for my dev kit 2 honestly.


Anyway, to go from Friend to Friend-lite, the steps are:
1. Modified main.dart to have bluetooth connection page
2. Modified android manifest to include bluetooth permissions
3. Added flutter_blue_plus and permission_handler dependencies
4. Created bluetooth_page.dart to handle bluetooth connection and scanning (_checkBluetoothPermission, _startScan, _connectToDevice)
5. Update bluetooth_page.dart within lib/ to handle the actual streaming of data. This also handles the charactaristic, which is basically services defined by a bluetooth device to describe the data that can be streamed. (_startAudioStreaming)
6. Created the websocket server within backend/main.py along with the pyproject.toml file. Implement the websocket and opus decoding using opuslib. First 3 bytes are metadata packets for the rest of the data. 
7. Websocket kept disconnecting, implement reconnect.
