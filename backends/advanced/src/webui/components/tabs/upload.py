"""
Audio file upload tab for the Streamlit UI
"""

import logging

import streamlit as st

from ..utils import post_data

logger = logging.getLogger("streamlit-ui")


def show_upload_tab():
    """Show the audio file upload interface."""
    st.header("📁 Upload Audio Files")
    st.caption("Upload audio files for processing through the transcription pipeline")

    # Check authentication
    if not st.session_state.get("authenticated", False):
        st.warning("🔒 Please log in to access the audio upload feature.")
        return

    # Check if user is admin
    user_info = st.session_state.get("user_info", {})
    is_admin = user_info.get("is_superuser", False) if isinstance(user_info, dict) else False

    if not is_admin:
        st.warning("🔒 This feature is only available to admin users.")
        st.info("💡 The audio upload feature requires admin privileges for security reasons.")
        return

    st.success("✅ Admin access confirmed - you can upload audio files for processing.")

    # File upload section
    with st.container():
        st.subheader("🎵 Select Audio Files")

        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            help="Select one or more audio files to process. Supported formats: WAV, MP3, M4A, OGG, FLAC, AAC",
        )

        # Device name input
        device_name = st.text_input(
            "Device Name",
            value="upload",
            help="Name to identify the source device for these uploads",
        )

        # Auto generate client option
        auto_generate_client = st.checkbox(
            "Auto-generate client ID",
            value=True,
            help="Automatically generate a client ID for processing",
        )

        if uploaded_files:
            st.write(f"**Selected files ({len(uploaded_files)}):**")

            total_size = 0
            for file in uploaded_files:
                file_size = len(file.getvalue())
                total_size += file_size
                st.write(f"• {file.name} ({file_size / (1024*1024):.1f} MB)")

            st.info(f"📊 Total size: {total_size / (1024*1024):.1f} MB")

            # Size warning
            if total_size > 100 * 1024 * 1024:  # 100MB
                st.warning(
                    "⚠️ Large files may take longer to process. Consider uploading in smaller batches."
                )

            # Upload button
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button(
                    "🚀 Upload and Process Files", type="primary", use_container_width=True
                ):
                    process_uploaded_files(uploaded_files, device_name, auto_generate_client)

    # Information section
    st.divider()

    with st.expander("ℹ️ How Audio Upload Works"):
        st.markdown(
            """
        **Processing Pipeline:**
        1. **Upload**: Files are uploaded to the backend server
        2. **Validation**: Audio format and quality are verified
        3. **Transcription**: Audio is processed through the transcription pipeline (Deepgram/Wyoming ASR)
        4. **Memory Extraction**: Conversations are analyzed and memories are extracted
        5. **Storage**: Transcripts and audio are stored in the database
        
        **Supported Formats:**
        - WAV (recommended for best quality)
        - MP3, M4A, AAC (compressed formats)
        - OGG, FLAC (other supported formats)
        
        **Processing Time:**
        - Typically 2-3x the audio duration
        - May take longer for poor quality audio or complex conversations
        
        **File Size Limits:**
        - Individual file: Up to 500MB
        - Batch upload: Up to 1GB total
        """
        )

    with st.expander("🔧 Troubleshooting"):
        st.markdown(
            """
        **Common Issues:**
        - **Upload fails**: Check file format and size limits
        - **Processing timeout**: Try smaller files or better quality audio
        - **No transcription**: Ensure audio contains clear speech
        - **Poor accuracy**: Use WAV format with good audio quality
        
        **Tips for Best Results:**
        - Use WAV format at 16kHz or higher sample rate
        - Ensure clear audio with minimal background noise
        - Split very long recordings into smaller segments
        - Test with a small file first
        """
        )


def process_uploaded_files(uploaded_files, device_name, auto_generate_client):
    """Process the uploaded files through the backend API."""
    try:
        # Prepare files for upload
        files_data = []
        for uploaded_file in uploaded_files:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            files_data.append(
                ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
            )

        with st.spinner(f"🚀 Uploading and processing {len(uploaded_files)} file(s)..."):
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("📤 Uploading files to backend...")
            progress_bar.progress(0.2)

            # Prepare request parameters
            params = {"device_name": device_name, "auto_generate_client": auto_generate_client}

            status_text.text("🔄 Processing through transcription pipeline...")
            progress_bar.progress(0.5)

            # Make the API call
            response = post_data(
                "/api/process-audio-files", params=params, require_auth=True, files=dict(files_data)
            )

            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")

            if response:
                # Check if there were any successful files
                successful_count = response.get("successful", 0)
                failed_count = response.get("failed", 0)
                total_files = len(uploaded_files)

                if successful_count > 0:
                    st.success(f"🎉 Files processed successfully!")
                elif failed_count == total_files:
                    st.error(f"❌ All {total_files} files failed to process!")
                else:
                    st.warning(
                        f"⚠️ Partial success: {successful_count} succeeded, {failed_count} failed"
                    )

                # Display summary
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Files", total_files)

                with col2:
                    st.metric("Successful", successful_count, delta_color="normal")

                with col3:
                    st.metric(
                        "Failed",
                        failed_count,
                        delta_color="inverse" if failed_count > 0 else "normal",
                    )

                # Show detailed results
                if "files" in response:
                    st.subheader("📋 Processing Results")

                    for file_result in response["files"]:
                        filename = file_result.get("filename", "Unknown")
                        status = file_result.get("status", "unknown")

                        if status == "processed" or status == "success":
                            # Handle successful processing
                            client_id = file_result.get("client_id")
                            sample_rate = file_result.get("sample_rate")
                            duration = file_result.get("duration_seconds", 0)

                            with st.container():
                                st.success(f"✅ **{filename}**")
                                if client_id:
                                    st.write(f"   🆔 Client ID: `{client_id}`")
                                if sample_rate:
                                    st.write(f"   🎵 Sample Rate: {sample_rate}Hz")
                                if duration > 0:
                                    st.write(f"   ⏱️ Duration: {duration:.1f}s")
                        else:
                            # Handle errors with detailed error message
                            error_msg = file_result.get("error", "Unknown error")
                            with st.container():
                                st.error(f"❌ **{filename}**")
                                st.write(f"   💥 Error: {error_msg}")

                                # Log the error for debugging
                                logger.error(f"File processing failed for {filename}: {error_msg}")

                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()

                if successful_count > 0:
                    st.info(
                        "💡 You can now view the processed conversations in the 'Conversations' tab."
                    )

            else:
                # This happens when post_data returns None due to an error
                st.error("❌ Failed to process files. The backend returned an error.")
                st.info(
                    "💡 Please check the error message above for details, or try again with a different file."
                )
                progress_bar.empty()
                status_text.empty()

    except Exception as e:
        logger.error(f"Error processing uploaded files: {e}", exc_info=True)
        st.error(f"❌ Error processing files: {str(e)}")
        st.info("💡 Please check that the backend is running and try again.")

        # Additional debugging information
        with st.expander("🔍 Debug Information"):
            st.code(f"Exception type: {type(e).__name__}\nException message: {str(e)}")
            st.write("This error occurred in the frontend while trying to process your upload.")
            st.write("Please check the browser console and backend logs for more details.")
