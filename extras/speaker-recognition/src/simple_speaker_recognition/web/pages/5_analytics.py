"""Analytics dashboard for speaker recognition system."""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.visualization import create_quality_metrics_plot
from database import get_db_session
from database.queries import (
    SpeakerQueries, EnrollmentQueries, AnnotationQueries, UserQueries
)

def system_overview():
    """Display system overview metrics."""
    st.subheader("📊 System Overview")
    
    if not st.session_state.user_id:
        st.warning("Please select a user to view analytics.")
        return
    
    db = get_db_session()
    try:
        # Get user statistics
        user_stats = UserQueries.get_user_stats(db, st.session_state.user_id)
        annotation_stats = AnnotationQueries.get_user_annotation_stats(db, st.session_state.user_id)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Enrolled Speakers", user_stats["speaker_count"])
        
        with col2:
            st.metric("📝 Total Annotations", annotation_stats["total"])
        
        with col3:
            correct_ratio = annotation_stats["correct"] / annotation_stats["total"] if annotation_stats["total"] > 0 else 0
            st.metric("✅ Accuracy Rate", f"{correct_ratio:.1%}")
        
        with col4:
            # Calculate total enrollment sessions
            speakers = SpeakerQueries.get_speakers_for_user(db, st.session_state.user_id)
            total_sessions = 0
            for speaker in speakers:
                sessions = EnrollmentQueries.get_sessions_for_speaker(db, speaker.id)
                total_sessions += len(sessions)
            
            st.metric("🎯 Enrollment Sessions", total_sessions)
        
        # Annotation breakdown pie chart
        if annotation_stats["total"] > 0:
            st.subheader("📋 Annotation Quality Breakdown")
            
            labels = []
            values = []
            colors = []
            
            if annotation_stats["correct"] > 0:
                labels.append("Correct")
                values.append(annotation_stats["correct"])
                colors.append("#90EE90")
            
            if annotation_stats["incorrect"] > 0:
                labels.append("Incorrect")
                values.append(annotation_stats["incorrect"])
                colors.append("#FFB6C1")
            
            if annotation_stats["uncertain"] > 0:
                labels.append("Uncertain")
                values.append(annotation_stats["uncertain"])
                colors.append("#FFE4B5")
            
            # Create pie chart
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                hole=0.3
            )])
            
            fig.update_layout(
                title="Annotation Quality Distribution",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading system overview: {str(e)}")
    finally:
        db.close()

def speaker_quality_analysis():
    """Analyze speaker enrollment quality trends."""
    st.subheader("🎤 Speaker Quality Analysis")
    
    if not st.session_state.user_id:
        return
    
    db = get_db_session()
    try:
        speakers = SpeakerQueries.get_speakers_for_user(db, st.session_state.user_id)
        
        if not speakers:
            st.info("No speakers enrolled yet.")
            return
        
        # Collect quality data
        speaker_names = []
        avg_quality_scores = []
        total_durations = []
        avg_snr_values = []
        session_counts = []
        
        for speaker in speakers:
            sessions = EnrollmentQueries.get_sessions_for_speaker(db, speaker.id)
            
            if sessions:
                speaker_names.append(speaker.name)
                
                # Calculate averages
                quality_scores = [s.quality_score for s in sessions if s.quality_score]
                durations = [s.speech_duration_seconds for s in sessions if s.speech_duration_seconds]
                snr_values = [s.snr_db for s in sessions if s.snr_db]
                
                avg_quality_scores.append(np.mean(quality_scores) if quality_scores else 0)
                total_durations.append(sum(durations) if durations else 0)
                avg_snr_values.append(np.mean(snr_values) if snr_values else 0)
                session_counts.append(len(sessions))
        
        if speaker_names:
            # Create quality comparison chart
            metrics = {
                'Quality Score': avg_quality_scores,
                'Total Duration (s)': total_durations,
                'Avg SNR (dB)': avg_snr_values,
                'Session Count': session_counts
            }
            
            fig = create_quality_metrics_plot(
                metrics,
                speaker_names,
                title="Speaker Quality Metrics Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Quality summary table
            st.subheader("📋 Speaker Quality Summary")
            
            quality_data = []
            for i, name in enumerate(speaker_names):
                quality_level = "Excellent" if avg_quality_scores[i] >= 0.8 else \
                               "Good" if avg_quality_scores[i] >= 0.6 else \
                               "Acceptable" if avg_quality_scores[i] >= 0.4 else "Poor"
                
                quality_data.append({
                    'Speaker': name,
                    'Quality Score': f"{avg_quality_scores[i]:.1%}",
                    'Quality Level': quality_level,
                    'Total Duration': f"{total_durations[i]:.1f}s",
                    'Avg SNR': f"{avg_snr_values[i]:.1f} dB",
                    'Sessions': session_counts[i]
                })
            
            df = pd.DataFrame(quality_data)
            st.dataframe(df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading speaker quality analysis: {str(e)}")
    finally:
        db.close()

def enrollment_trends():
    """Show enrollment activity trends over time."""
    st.subheader("📈 Enrollment Activity Trends")
    
    if not st.session_state.user_id:
        return
    
    db = get_db_session()
    try:
        speakers = SpeakerQueries.get_speakers_for_user(db, st.session_state.user_id)
        
        if not speakers:
            st.info("No enrollment data available.")
            return
        
        # Collect all enrollment sessions
        all_sessions = []
        for speaker in speakers:
            sessions = EnrollmentQueries.get_sessions_for_speaker(db, speaker.id)
            for session in sessions:
                all_sessions.append({
                    'date': session.created_at.date(),
                    'quality_score': session.quality_score or 0,
                    'duration': session.speech_duration_seconds or 0,
                    'speaker_name': speaker.name,
                    'method': session.enrollment_method
                })
        
        if not all_sessions:
            st.info("No enrollment sessions found.")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_sessions)
        
        # Group by date
        daily_stats = df.groupby('date').agg({
            'quality_score': ['mean', 'count'],
            'duration': 'sum'
        }).round(3)
        
        daily_stats.columns = ['Avg Quality', 'Session Count', 'Total Duration (s)']
        daily_stats = daily_stats.reset_index()
        
        # Create timeline chart
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Session Count', 'Average Quality Score'),
            vertical_spacing=0.1
        )
        
        # Session count over time
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['Session Count'],
                mode='lines+markers',
                name='Sessions per Day',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Quality trend over time
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['Avg Quality'],
                mode='lines+markers',
                name='Avg Quality',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Enrollment Activity Over Time",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Session Count", row=1, col=1)
        fig.update_yaxes(title_text="Quality Score", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity summary
        st.subheader("📅 Recent Activity")
        
        # Show last 10 sessions
        recent_sessions = sorted(all_sessions, key=lambda x: df[df.index == all_sessions.index(x)]['date'].iloc[0], reverse=True)[:10]
        
        recent_data = []
        for session in recent_sessions:
            recent_data.append({
                'Date': session['date'].strftime('%Y-%m-%d'),
                'Speaker': session['speaker_name'],
                'Method': session['method'],
                'Quality': f"{session['quality_score']:.1%}",
                'Duration': f"{session['duration']:.1f}s"
            })
        
        if recent_data:
            recent_df = pd.DataFrame(recent_data)
            st.dataframe(recent_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading enrollment trends: {str(e)}")
    finally:
        db.close()

def system_recommendations():
    """Provide system usage recommendations."""
    st.subheader("💡 System Recommendations")
    
    if not st.session_state.user_id:
        return
    
    db = get_db_session()
    try:
        speakers = SpeakerQueries.get_speakers_for_user(db, st.session_state.user_id)
        annotation_stats = AnnotationQueries.get_user_annotation_stats(db, st.session_state.user_id)
        
        recommendations = []
        
        # Speaker-based recommendations
        if len(speakers) == 0:
            recommendations.append("🎤 **Get Started**: Enroll your first speaker using the Enrollment page")
        elif len(speakers) < 3:
            recommendations.append("👥 **Expand Coverage**: Consider enrolling more speakers for better system utility")
        
        # Quality-based recommendations
        low_quality_speakers = 0
        total_sessions = 0
        
        for speaker in speakers:
            sessions = EnrollmentQueries.get_sessions_for_speaker(db, speaker.id)
            total_sessions += len(sessions)
            
            if sessions:
                avg_quality = np.mean([s.quality_score for s in sessions if s.quality_score])
                if avg_quality < 0.6:
                    low_quality_speakers += 1
        
        if low_quality_speakers > 0:
            recommendations.append(f"📈 **Improve Quality**: {low_quality_speakers} speakers have low quality scores - consider re-enrollment")
        
        # Annotation-based recommendations
        if annotation_stats["total"] == 0:
            recommendations.append("📝 **Start Annotating**: Use the Annotation tool to label speaker segments in your audio")
        elif annotation_stats["uncertain"] > annotation_stats["correct"]:
            recommendations.append("✅ **Review Annotations**: You have many uncertain annotations - consider reviewing and updating them")
        
        # Activity-based recommendations
        if total_sessions < len(speakers) * 2:
            recommendations.append("🎯 **Multiple Sessions**: Enroll multiple sessions per speaker for better recognition accuracy")
        
        # Data export recommendations
        if len(speakers) >= 3 and annotation_stats["total"] > 10:
            recommendations.append("📤 **Export Data**: You have substantial data - consider exporting for backup or external analysis")
        
        # Display recommendations
        if recommendations:
            for rec in recommendations:
                st.markdown(f"• {rec}")
        else:
            st.success("🎉 **Great job!** Your speaker recognition system is well-configured and actively used.")
        
        # Usage tips
        st.subheader("🎯 Best Practices")
        
        st.markdown("""
        **For Better Recognition Accuracy:**
        - Enroll at least 30 seconds of clear speech per speaker
        - Use multiple enrollment sessions with varied speech content
        - Maintain consistent audio quality (SNR > 15 dB)
        - Regularly review and update uncertain annotations
        
        **For Efficient Workflow:**
        - Use the Audio Viewer to explore files before annotation
        - Batch process multiple files during enrollment
        - Export data regularly for backup and analysis
        - Compare speaker quality metrics to identify improvement opportunities
        
        **For System Maintenance:**
        - Monitor enrollment quality trends over time
        - Remove or re-enroll speakers with consistently poor quality
        - Keep annotations updated as you refine speaker identification
        - Use the analytics to track system usage and performance
        """)
    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
    finally:
        db.close()

def main():
    """Main analytics page."""
    st.title("📊 Analytics Dashboard")
    st.markdown("Monitor system performance, track quality metrics, and get recommendations for optimal usage.")
    
    # Check if user is logged in
    if "username" not in st.session_state or not st.session_state.username:
        st.warning("👈 Please select or create a user in the sidebar to continue.")
        return
    
    # System overview
    system_overview()
    
    st.divider()
    
    # Speaker quality analysis
    speaker_quality_analysis()
    
    st.divider()
    
    # Enrollment trends
    enrollment_trends()
    
    st.divider()
    
    # Recommendations
    system_recommendations()

if __name__ == "__main__":
    main()