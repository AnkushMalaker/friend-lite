Add ngrok token in .env file (check .env.template)

Run docker compose up

Paste the URL in the OMI app -> developer mode -> audio bytes webhook URL with the following suffix:

<URL>/webhook/audio_bytes

Set duration to 5-10 seconds to test
Save

You should now see audio files in the audio_recordings directory