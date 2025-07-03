#!/usr/bin/env python3
"""
Live Audio Player with Speaker Tracking - PyQt5 Version
Real-time audio playback with live speaker and timestamp updates

Run the `extras/speaker-recognition/scripts/extract_speaker_from_plots.py` to get the JSON file.
"""

import sys
import json
import threading
from pathlib import Path
import time

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                                QLineEdit, QFileDialog, QMessageBox, QGroupBox,
                                QSplitter, QFrame, QSlider, QProgressBar)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
    from PyQt5.QtGui import QFont, QTextCursor, QTextCharFormat, QColor, QPalette
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    from PyQt5.QtCore import QUrl
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

if not PYQT_AVAILABLE:
    print("PyQt5 not found. Installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt5"])
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                    QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                                    QLineEdit, QFileDialog, QMessageBox, QGroupBox,
                                    QSplitter, QFrame, QSlider, QProgressBar)
        from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
        from PyQt5.QtGui import QFont, QTextCursor, QTextCharFormat, QColor, QPalette
        from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
        from PyQt5.QtCore import QUrl
    except Exception as e:
        print(f"Failed to install PyQt5: {e}")
        print("Please install PyQt5 manually: pip install PyQt5")
        sys.exit(1)

class LiveAudioPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎵 Live Audio Player with Speaker Tracking")
        self.setGeometry(100, 100, 1400, 900)
        
        # Variables
        self.segments_data = None
        self.audio_file_path = None
        self.current_segment = None
        self.current_speaker = "No Audio Loaded"
        self.current_time = 0.0
        self.total_duration = 0.0
        
        # Audio player
        self.media_player = QMediaPlayer()
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.media_player.stateChanged.connect(self.on_state_changed)
        
        # Timer for live updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_live_display)
        self.update_timer.start(100)  # Update every 100ms
        
        self.init_ui()
        self.update_window_title()
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Live title display
        self.title_label = QLabel("🎵 No Audio Loaded - 00:00")
        self.title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6b6b, stop:0.5 #4ecdc4, stop:1 #45b7d1);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 10px;
            }
        """)
        main_layout.addWidget(self.title_label)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - File loading and JSON
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Audio player and timeline
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([500, 900])
        
        # Status bar
        self.statusBar().showMessage("Ready - Load JSON and audio file to begin")
        
    def create_left_panel(self):
        """Create the left panel with file loading and JSON display"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File loading section
        file_group = QGroupBox("📄 Load Files")
        file_layout = QVBoxLayout(file_group)
        
        # JSON file
        json_layout = QHBoxLayout()
        self.load_json_btn = QPushButton("Load JSON File")
        self.load_json_btn.clicked.connect(self.load_json_file)
        self.json_status = QLabel("No JSON loaded")
        json_layout.addWidget(self.load_json_btn)
        json_layout.addWidget(self.json_status)
        file_layout.addLayout(json_layout)
        
        # Audio file
        audio_layout = QHBoxLayout()
        self.load_audio_btn = QPushButton("Load Audio File")
        self.load_audio_btn.clicked.connect(self.load_audio_file)
        self.audio_status = QLabel("No audio loaded")
        audio_layout.addWidget(self.load_audio_btn)
        audio_layout.addWidget(self.audio_status)
        file_layout.addLayout(audio_layout)
        
        layout.addWidget(file_group)
        
        # JSON display
        json_group = QGroupBox("JSON Segments")
        json_layout = QVBoxLayout(json_group)
        
        self.json_text = QTextEdit()
        self.json_text.setFont(QFont("Consolas", 10))
        self.json_text.mousePressEvent = self.on_json_click
        json_layout.addWidget(self.json_text)
        
        layout.addWidget(json_group)
        
        return panel
        
    def create_right_panel(self):
        """Create the right panel with audio player and timeline"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Audio player controls
        player_group = QGroupBox("🎵 Audio Player")
        player_layout = QVBoxLayout(player_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6b6b, stop:1 #4ecdc4);
                border-radius: 3px;
            }
        """)
        player_layout.addWidget(self.progress_bar)
        
        # Time labels
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.total_time_label = QLabel("00:00")
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        player_layout.addLayout(time_layout)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.play_pause)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.sliderPressed.connect(self.on_seek_start)
        self.seek_slider.sliderReleased.connect(self.on_seek_end)
        self.seek_slider.setEnabled(False)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(QLabel("Seek:"))
        controls_layout.addWidget(self.seek_slider)
        
        player_layout.addLayout(controls_layout)
        
        layout.addWidget(player_group)
        
        # Current speaker display
        speaker_group = QGroupBox("🎤 Current Speaker")
        speaker_layout = QVBoxLayout(speaker_group)
        
        self.current_speaker_label = QLabel("No speaker")
        self.current_speaker_label.setAlignment(Qt.AlignCenter)
        speaker_font = QFont()
        speaker_font.setPointSize(18)
        speaker_font.setBold(True)
        self.current_speaker_label.setFont(speaker_font)
        self.current_speaker_label.setStyleSheet("""
            QLabel {
                background-color: #f0f2f6;
                border: 2px solid #ff6b6b;
                border-radius: 10px;
                padding: 20px;
                color: #ff6b6b;
            }
        """)
        speaker_layout.addWidget(self.current_speaker_label)
        
        # Segment info
        self.segment_info_label = QLabel("No segment selected")
        self.segment_info_label.setAlignment(Qt.AlignCenter)
        speaker_layout.addWidget(self.segment_info_label)
        
        layout.addWidget(speaker_group)
        
        # Speaker timeline
        timeline_group = QGroupBox("📋 Speaker Timeline")
        timeline_layout = QVBoxLayout(timeline_group)
        
        self.timeline_text = QTextEdit()
        self.timeline_text.setMaximumHeight(300)
        self.timeline_text.setReadOnly(True)
        timeline_layout.addWidget(self.timeline_text)
        
        layout.addWidget(timeline_group)
        
        return panel
    
    def load_json_file(self):
        """Load JSON file and display it"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON File", "", "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.segments_data = json.load(f)
                
                # Display JSON with formatting
                formatted_json = json.dumps(self.segments_data, indent=2)
                self.json_text.setPlainText(formatted_json)
                
                segments = self.segments_data.get('segments', [])
                self.json_status.setText(f"✅ {len(segments)} segments")
                self.update_timeline_display()
                
                # Calculate total duration
                if segments:
                    self.total_duration = max(segment.get('end', 0) for segment in segments)
                
                self.statusBar().showMessage(f"Loaded {len(segments)} segments")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load JSON file:\n{str(e)}")
    
    def load_audio_file(self):
        """Load audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio files (*.mp3 *.wav *.ogg *.m4a *.flac *.aac);;All files (*.*)"
        )
        
        if file_path:
            self.audio_file_path = file_path
            
            # Load into media player
            url = QUrl.fromLocalFile(file_path)
            content = QMediaContent(url)
            self.media_player.setMedia(content)
            
            self.audio_status.setText(f"✅ {Path(file_path).name}")
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.seek_slider.setEnabled(True)
            
            self.statusBar().showMessage("Audio file loaded - Ready to play")
    
    def on_json_click(self, event):
        """Handle clicks on JSON text"""
        if not self.segments_data:
            return
        
        # Call the original mousePressEvent
        QTextEdit.mousePressEvent(self.json_text, event)
        
        try:
            cursor = self.json_text.textCursor()
            position = cursor.position()
            
            content = self.json_text.toPlainText()
            segment_index = self.find_segment_at_position(content, position)
            
            if segment_index is not None:
                segment = self.segments_data['segments'][segment_index]
                self.select_segment(segment, segment_index)
                
                # Seek to segment start
                start_time = segment.get('start', 0) * 1000  # Convert to milliseconds
                self.media_player.setPosition(int(start_time))
                
        except Exception as e:
            print(f"Error finding segment: {e}")
    
    def find_segment_at_position(self, content, char_index):
        """Find which segment object contains the given character position"""
        lines = content.split('\n')
        current_pos = 0
        segment_index = -1
        in_segments_array = False
        brace_count = 0
        
        for line in lines:
            line_start = current_pos
            line_end = current_pos + len(line)
            
            if '"segments"' in line and '[' in line:
                in_segments_array = True
                brace_count = 0
            
            if in_segments_array:
                for char in line:
                    if char == '{':
                        if brace_count == 0:
                            segment_index += 1
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                if line_start <= char_index <= line_end and brace_count > 0:
                    return segment_index
            
            current_pos = line_end + 1
        
        return None
    
    def select_segment(self, segment, index):
        """Select and highlight a segment"""
        self.current_segment = segment
        self.highlight_segment_in_json(index)
        
        speaker = segment.get('verified_speaker', segment.get('speaker', 'Unknown'))
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        
        self.segment_info_label.setText(f"Selected: {speaker} ({start:.1f}s - {end:.1f}s)")
        self.statusBar().showMessage(f"Selected segment: {speaker}")
    
    def highlight_segment_in_json(self, segment_index):
        """Highlight the specific segment in the JSON display"""
        content = self.json_text.toPlainText()
        segment_start = -1
        segment_end = -1
        brace_count = 0
        current_segment = -1
        in_segments = False
        
        for i, char in enumerate(content):
            if not in_segments and content[i:].startswith('"segments"'):
                in_segments = True
                continue
                
            if in_segments:
                if char == '{':
                    if brace_count == 0:
                        current_segment += 1
                        if current_segment == segment_index:
                            segment_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and current_segment == segment_index:
                        segment_end = i + 1
                        break
        
        if segment_start != -1 and segment_end != -1:
            cursor = self.json_text.textCursor()
            cursor.setPosition(segment_start)
            cursor.setPosition(segment_end, QTextCursor.KeepAnchor)
            self.json_text.setTextCursor(cursor)
    
    def play_pause(self):
        """Toggle play/pause"""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("▶️ Play")
        else:
            self.media_player.play()
            self.play_btn.setText("⏸️ Pause")
    
    def stop(self):
        """Stop playback"""
        self.media_player.stop()
        self.play_btn.setText("▶️ Play")
    
    def on_position_changed(self, position):
        """Handle position changes"""
        self.current_time = position / 1000.0  # Convert to seconds
        
        # Update progress bar
        if self.total_duration > 0:
            progress = (position / (self.total_duration * 1000)) * 100
            self.progress_bar.setValue(int(progress))
        
        # Update seek slider
        if not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(position)
        
        # Update time display
        self.update_time_display()
        
        # Update current speaker
        self.update_current_speaker()
    
    def on_duration_changed(self, duration):
        """Handle duration changes"""
        self.total_duration = duration / 1000.0  # Convert to seconds
        self.seek_slider.setRange(0, duration)
        self.update_time_display()
    
    def on_state_changed(self, state):
        """Handle state changes"""
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("⏸️ Pause")
        else:
            self.play_btn.setText("▶️ Play")
    
    def on_seek_start(self):
        """Handle seek slider pressed"""
        pass
    
    def on_seek_end(self):
        """Handle seek slider released"""
        position = self.seek_slider.value()
        self.media_player.setPosition(position)
    
    def update_time_display(self):
        """Update time display labels"""
        current_min = int(self.current_time // 60)
        current_sec = int(self.current_time % 60)
        
        total_min = int(self.total_duration // 60)
        total_sec = int(self.total_duration % 60)
        
        self.current_time_label.setText(f"{current_min:02d}:{current_sec:02d}")
        self.total_time_label.setText(f"{total_min:02d}:{total_sec:02d}")
    
    def update_current_speaker(self):
        """Update current speaker based on timestamp"""
        if not self.segments_data:
            return
        
        segments = self.segments_data.get('segments', [])
        current_speaker = "No Speaker"
        current_segment = None
        
        for segment in segments:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            if start <= self.current_time <= end:
                current_speaker = segment.get('verified_speaker', segment.get('speaker', 'Unknown'))
                current_segment = segment
                break
        
        self.current_speaker = current_speaker
        self.current_speaker_label.setText(f"🎤 {current_speaker}")
        
        if current_segment:
            start = current_segment.get('start', 0)
            end = current_segment.get('end', 0)
            self.segment_info_label.setText(f"Speaking: {start:.1f}s - {end:.1f}s")
        else:
            self.segment_info_label.setText("No active speaker")
    
    def update_live_display(self):
        """Update live display elements"""
        # Update window title
        self.update_window_title()
        
        # Update timeline display
        self.update_timeline_display()
    
    def update_window_title(self):
        """Update window title with current info"""
        current_min = int(self.current_time // 60)
        current_sec = int(self.current_time % 60)
        time_str = f"{current_min:02d}:{current_sec:02d}"
        
        title_text = f"🎵 {self.current_speaker} - {time_str}"
        self.title_label.setText(title_text)
        self.setWindowTitle(f"Live Audio Player - {self.current_speaker} - {time_str}")
    
    def update_timeline_display(self):
        """Update the timeline display"""
        if not self.segments_data:
            return
        
        segments = self.segments_data.get('segments', [])
        timeline_html = "<style>body { font-family: Arial; }</style>"
        
        for i, segment in enumerate(segments):
            speaker = segment.get('speaker', 'Unknown')
            verified_speaker = segment.get('verified_speaker', speaker)
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            # Determine status
            if self.current_time < start_time:
                status = "⏳"
                color = "#6c757d"
                bg_color = "#f8f9fa"
            elif start_time <= self.current_time <= end_time:
                status = "🔴"
                color = "#ff6b6b"
                bg_color = "#fff5f5"
            else:
                status = "✅"
                color = "#28a745"
                bg_color = "#f8fff8"
            
            timeline_html += f"""
            <div style="background-color: {bg_color}; border-left: 4px solid {color}; 
                        padding: 10px; margin: 5px 0; border-radius: 5px;">
                <strong>{status} {verified_speaker}</strong><br>
                <small style="color: #666;">Speaker: {speaker}</small><br>
                <small style="color: {color};">{start_time:.1f}s - {end_time:.1f}s</small>
            </div>
            """
        
        self.timeline_text.setHtml(timeline_html)
    
    def closeEvent(self, event):
        """Handle window closing"""
        self.media_player.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    player = LiveAudioPlayer()
    player.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()