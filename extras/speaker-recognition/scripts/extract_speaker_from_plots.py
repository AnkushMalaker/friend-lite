"""
Speaker Audio Extraction and Labeling Tool

WHEN TO USE THIS FILE:
This tool is designed for extracting and labeling audio clips from speakers identified 
in conversation transcripts. Use this when you have:

1. A JSON file containing speaker segmentation data (from speaker recognition systems)
2. The corresponding audio file of the conversation
3. Need to extract clean audio clips of individual speakers for:
   - Speaker enrollment/training data
   - Voice analysis
   - Creating speaker-specific datasets
   - Quality control of speaker identification

FEATURES:
- Visual frequency plot showing speaker duration distribution
- Interactive speaker selection by clicking on plot bars
- Automatic extraction of 30-second continuous audio clips
- Audio playback with controls (play/stop)
- Manual labeling of extracted audio clips
- Batch processing capability

INPUT REQUIREMENTS:
- JSON file: Must contain 'segments' array with 'speaker', 'start', 'end' fields
- Audio file: WAV, MP3, or M4A format matching the transcript
- Optional: 'verified_speaker' field in JSON for corrected speaker labels

OUTPUT:
- Labeled WAV files saved as: {label}_{speaker}_{timestamp}.wav
- Files saved to selected output directory

USAGE:
1. Run: python extract_speaker_from_plots.py
2. Select JSON file with speaker segments
3. Select corresponding audio file
4. Choose output directory
5. Click on speaker bars in the plot to select speakers
6. Generate and play audio clips
7. Label and save the audio clips
"""

import sys
import json
import random
import os
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                            QFileDialog, QMessageBox, QDialog, QListWidget,
                            QListWidgetItem, QInputDialog, QProgressBar,
                            QTextEdit, QSplitter, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import pygame
from pydub import AudioSegment
import tempfile
import pandas as pd

class AudioPlayerThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path
        self.should_stop = False
        self.mixer_initialized = False
        
    def run(self):
        try:
            pygame.mixer.init()
            self.mixer_initialized = True
            pygame.mixer.music.load(self.audio_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy() and not self.should_stop:
                self.msleep(100)
                
            # Clean up mixer
            if self.mixer_initialized:
                pygame.mixer.quit()
                self.mixer_initialized = False
                
            self.finished.emit()
            
        except Exception as e:
            # Clean up on error
            if self.mixer_initialized:
                try:
                    pygame.mixer.quit()
                    self.mixer_initialized = False
                except:
                    pass
            self.error.emit(str(e))
    
    def stop(self):
        self.should_stop = True
        try:
            if self.mixer_initialized and pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass  # Ignore errors when stopping

class FrequencyPlotWidget(QWidget):
    speaker_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.speaker_durations = {}
        self.bars = []
        self.selected_bar = None
        self.canvas.mpl_connect('button_press_event', self.on_bar_click)
        
    def update_plot(self, json_data):
        """Update the frequency plot with new data"""
        self.speaker_durations = self.calculate_speaker_durations(json_data)
        
        if not self.speaker_durations:
            return
            
        # Clear previous plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(list(self.speaker_durations.items()), 
                         columns=['Speaker', 'Duration'])
        df = df.sort_values('Duration', ascending=False)
        
        # Create bar plot
        self.bars = ax.bar(df['Speaker'], df['Duration'], 
                          color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Customize plot
        ax.set_title('Speaker Duration Analysis - Click a bar to select', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Speaker', fontsize=12)
        ax.set_ylabel('Total Duration (seconds)', fontsize=12)
        
        # Add value labels on bars
        for bar, duration in zip(self.bars, df['Duration']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{duration:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Store speaker names for bar mapping
        self.speaker_names = df['Speaker'].tolist()
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def calculate_speaker_durations(self, json_data):
        """Calculate total duration for each speaker"""
        speaker_durations = defaultdict(float)
        
        for segment in json_data['segments']:
            speaker = segment.get('verified_speaker', segment['speaker'])
            duration = segment['end'] - segment['start']
            speaker_durations[speaker] += duration
        
        return dict(speaker_durations)
    
    def on_bar_click(self, event):
        """Handle bar click events"""
        if event.inaxes is None or not self.bars:
            return
            
        # Find which bar was clicked
        for i, bar in enumerate(self.bars):
            if bar.contains(event)[0]:
                # Reset previous selection
                if self.selected_bar is not None:
                    self.selected_bar.set_color('skyblue')
                
                # Highlight selected bar
                bar.set_color('orange')
                self.selected_bar = bar
                
                # Emit signal with selected speaker
                speaker = self.speaker_names[i]
                self.speaker_selected.emit(speaker)
                
                self.canvas.draw()
                break

class SpeakerLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.json_data = None
        self.current_speaker = None
        self.audio_thread = None
        self.temp_audio_path = None
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Speaker Audio Labeling Tool')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create top controls
        controls_layout = QHBoxLayout()
        
        # File selection
        file_group = QGroupBox("Data Files")
        file_layout = QGridLayout(file_group)
        
        self.json_path_label = QLabel("No JSON file selected")
        self.audio_path_label = QLabel("No audio file selected")
        
        json_btn = QPushButton("Select JSON File")
        json_btn.clicked.connect(self.select_json_file)
        
        audio_btn = QPushButton("Select Audio File")
        audio_btn.clicked.connect(self.select_audio_file)
        
        file_layout.addWidget(QLabel("JSON:"), 0, 0)
        file_layout.addWidget(self.json_path_label, 0, 1)
        file_layout.addWidget(json_btn, 0, 2)
        file_layout.addWidget(QLabel("Audio:"), 1, 0)
        file_layout.addWidget(self.audio_path_label, 1, 1)
        file_layout.addWidget(audio_btn, 1, 2)
        
        controls_layout.addWidget(file_group)
        
        # Output directory selection
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout(output_group)
        
        self.output_dir_label = QLabel("Current directory")
        self.output_dir = "."
        
        output_btn = QPushButton("Select Output Dir")
        output_btn.clicked.connect(self.select_output_dir)
        
        output_layout.addWidget(self.output_dir_label)
        output_layout.addWidget(output_btn)
        
        controls_layout.addWidget(output_group)
        
        main_layout.addLayout(controls_layout)
        
        # Create splitter for plot and controls
        splitter = QSplitter(Qt.Vertical)
        
        # Frequency plot
        self.plot_widget = FrequencyPlotWidget()
        self.plot_widget.speaker_selected.connect(self.on_speaker_selected)
        splitter.addWidget(self.plot_widget)
        
        # Audio control panel
        audio_control_widget = QWidget()
        audio_control_layout = QVBoxLayout(audio_control_widget)
        
        # Speaker info
        speaker_info_group = QGroupBox("Selected Speaker")
        speaker_info_layout = QVBoxLayout(speaker_info_group)
        
        self.speaker_info_label = QLabel("No speaker selected")
        self.speaker_info_label.setFont(QFont("Arial", 12, QFont.Bold))
        speaker_info_layout.addWidget(self.speaker_info_label)
        
        # Audio controls
        audio_controls_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("Generate 30s Clip")
        self.generate_btn.clicked.connect(self.generate_audio_clip)
        self.generate_btn.setEnabled(False)
        
        self.play_btn = QPushButton("Play Audio")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop Audio")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)
        
        audio_controls_layout.addWidget(self.generate_btn)
        audio_controls_layout.addWidget(self.play_btn)
        audio_controls_layout.addWidget(self.stop_btn)
        
        speaker_info_layout.addLayout(audio_controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        speaker_info_layout.addWidget(self.progress_bar)
        
        # Labeling section
        label_group = QGroupBox("Audio Labeling")
        label_layout = QVBoxLayout(label_group)
        
        label_input_layout = QHBoxLayout()
        label_input_layout.addWidget(QLabel("Label:"))
        
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Enter label for this audio clip")
        label_input_layout.addWidget(self.label_input)
        
        self.save_btn = QPushButton("Save Labeled Audio")
        self.save_btn.clicked.connect(self.save_labeled_audio)
        self.save_btn.setEnabled(False)
        label_input_layout.addWidget(self.save_btn)
        
        label_layout.addLayout(label_input_layout)
        
        # Status/log area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        label_layout.addWidget(QLabel("Log:"))
        label_layout.addWidget(self.log_text)
        
        audio_control_layout.addWidget(speaker_info_group)
        audio_control_layout.addWidget(label_group)
        
        splitter.addWidget(audio_control_widget)
        splitter.setSizes([400, 300])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready - Select JSON and audio files to begin")
        
    def log_message(self, message):
        """Add message to log"""
        self.log_text.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def select_json_file(self):
        """Select JSON file with speaker segments"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON File", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.json_data = json.load(f)
                
                self.json_path_label.setText(os.path.basename(file_path))
                self.plot_widget.update_plot(self.json_data)
                self.log_message(f"Loaded JSON file: {os.path.basename(file_path)}")
                self.statusBar().showMessage("JSON file loaded - Select audio file")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load JSON file: {str(e)}")
                
    def select_audio_file(self):
        """Select audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.m4a)"
        )
        
        if file_path:
            self.audio_path = file_path
            self.audio_path_label.setText(os.path.basename(file_path))
            self.log_message(f"Selected audio file: {os.path.basename(file_path)}")
            self.statusBar().showMessage("Audio file selected - Click on a speaker bar to continue")
            
    def select_output_dir(self):
        """Select output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(os.path.basename(dir_path))
            self.log_message(f"Output directory: {dir_path}")
            
    def on_speaker_selected(self, speaker):
        """Handle speaker selection from plot"""
        self.current_speaker = speaker
        self.speaker_info_label.setText(f"Selected: {speaker}")
        
        if hasattr(self, 'audio_path') and self.json_data:
            self.generate_btn.setEnabled(True)
            self.log_message(f"Selected speaker: {speaker}")
            self.statusBar().showMessage(f"Speaker '{speaker}' selected - Generate audio clip")
        else:
            self.generate_btn.setEnabled(False)
            self.statusBar().showMessage("Please select both JSON and audio files")
            
    def find_continuous_segments(self, target_speaker, min_duration=30):
        """Find continuous segments for speaker"""
        speaker_segments = []
        for segment in self.json_data['segments']:
            speaker = segment.get('verified_speaker', segment['speaker'])
            if speaker == target_speaker:
                speaker_segments.append(segment)
        
        if not speaker_segments:
            return []
        
        speaker_segments.sort(key=lambda x: x['start'])
        
        continuous_clips = []
        current_clip = [speaker_segments[0]]
        
        for i in range(1, len(speaker_segments)):
            current_segment = speaker_segments[i]
            last_segment = current_clip[-1]
            
            gap = current_segment['start'] - last_segment['end']
            
            if gap <= 2.0:  # Allow small gaps
                current_clip.append(current_segment)
            else:
                clip_duration = current_clip[-1]['end'] - current_clip[0]['start']
                if clip_duration >= min_duration:
                    continuous_clips.append(current_clip.copy())
                
                current_clip = [current_segment]
        
        # Check the last clip
        if current_clip:
            clip_duration = current_clip[-1]['end'] - current_clip[0]['start']
            if clip_duration >= min_duration:
                continuous_clips.append(current_clip)
        
        return continuous_clips
            
    def generate_audio_clip(self):
        """Generate 30-second audio clip for selected speaker"""
        if not self.current_speaker or not hasattr(self, 'audio_path'):
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.generate_btn.setEnabled(False)
        
        try:
            # Find continuous segments
            continuous_clips = self.find_continuous_segments(self.current_speaker, 30)
            
            if not continuous_clips:
                QMessageBox.warning(self, "No Suitable Clips", 
                                  f"No continuous clips of at least 30 seconds found for {self.current_speaker}")
                self.progress_bar.setVisible(False)
                self.generate_btn.setEnabled(True)
                return
            
            # Select random clip
            selected_clip = random.choice(continuous_clips)
            start_time = selected_clip[0]['start']
            end_time = selected_clip[-1]['end']
            
            # Limit to 30 seconds if longer
            if end_time - start_time > 30:
                end_time = start_time + 30
            
            self.log_message(f"Generating clip: {start_time:.1f}s - {end_time:.1f}s")
            
            # Load and extract audio
            audio = AudioSegment.from_file(self.audio_path)
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            clip = audio[start_ms:end_ms]
            
            # Save to temporary file
            if self.temp_audio_path:
                try:
                    os.remove(self.temp_audio_path)
                except:
                    pass
            
            self.temp_audio_path = tempfile.mktemp(suffix='.wav')
            clip.export(self.temp_audio_path, format="wav")
            
            self.play_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.log_message(f"Generated {end_time - start_time:.1f}s clip for {self.current_speaker}")
            self.statusBar().showMessage("Audio clip generated - Play and label it")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate audio clip: {str(e)}")
            self.log_message(f"Error generating clip: {str(e)}")
            
        finally:
            self.progress_bar.setVisible(False)
            self.generate_btn.setEnabled(True)
            
    def play_audio(self):
        """Play the generated audio clip"""
        if not self.temp_audio_path:
            return
            
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.audio_thread = AudioPlayerThread(self.temp_audio_path)
        self.audio_thread.finished.connect(self.on_audio_finished)
        self.audio_thread.error.connect(self.on_audio_error)
        self.audio_thread.start()
        
        self.log_message("Playing audio clip...")
        self.statusBar().showMessage("Playing audio...")
        
    def stop_audio(self):
        """Stop audio playback"""
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
            self.audio_thread.wait(1000)  # Wait up to 1 second for clean shutdown
            
        self.on_audio_finished()
        
    def on_audio_finished(self):
        """Handle audio playback finished"""
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_message("Audio playback finished")
        self.statusBar().showMessage("Audio finished - Enter label and save")
        
    def on_audio_error(self, error):
        """Handle audio playback error"""
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Audio Error", f"Audio playback failed: {error}")
        self.log_message(f"Audio error: {error}")
        
    def save_labeled_audio(self):
        """Save the audio clip with label"""
        if not self.temp_audio_path or not self.current_speaker:
            return
            
        label = self.label_input.text().strip()
        if not label:
            QMessageBox.warning(self, "No Label", "Please enter a label for the audio clip")
            return
            
        try:
            # Create output filename
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{label}_{self.current_speaker}_{timestamp}.wav"
            output_path = os.path.join(self.output_dir, filename)
            
            # Copy the temporary file to final location
            import shutil
            shutil.copy2(self.temp_audio_path, output_path)
            
            self.log_message(f"Saved labeled audio: {filename}")
            self.statusBar().showMessage(f"Audio saved as: {filename}")
            
            # Clear label input for next clip
            self.label_input.clear()
            
            # Reset for next clip
            self.play_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            
            QMessageBox.information(self, "Success", f"Audio saved as:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save audio: {str(e)}")
            self.log_message(f"Error saving audio: {str(e)}")
            
    def closeEvent(self, event):
        """Clean up when closing application"""
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
            self.audio_thread.wait(2000)  # Wait up to 2 seconds for clean shutdown
            
        if self.temp_audio_path and os.path.exists(self.temp_audio_path):
            try:
                os.remove(self.temp_audio_path)
            except:
                pass
                
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = SpeakerLabelingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()