*** Settings ***
Documentation    Audio Keywords
Library          RequestsLibrary
Library          Collections
Library          OperatingSystem
Variables        ../test_data.py
Resource         session_resources.robot
Resource         conversation_keywords.robot
Resource         queue_keywords.robot

*** Keywords ***
Upload Audio File
      [Documentation]    Upload audio file using session with proper multipart form data
      [Arguments]    ${audio_file_path}    ${device_name}=robot-test
      
      # Verify file exists
      File Should Exist    ${audio_file_path}

      # Debug the request being sent
      Log    Sending file: ${audio_file_path}
      Log    Device name: ${device_name}

      # Create proper file upload using Python expressions to actually open the file
      Log    Files dictionary will contain: files -> ${audio_file_path}
      Log    Data dictionary will contain: device_name -> ${device_name}

        ${response}=       POST On Session    api    /api/audio/upload
        ...                files=${{ {'files': open('${audio_file_path}', 'rb')} }}
        ...                params=device_name=${device_name}
        ...                expected_status=any

      # Detailed debugging of the response
      Log    Upload response status: ${response.status_code}
      Log    Upload response headers: ${response.headers}
      Log    Upload response content type: ${response.headers.get('content-type', 'not set')}
      Log    Upload response text length: ${response.text.__len__()}
      Log    Upload response raw text: ${response.text}

      # Parse JSON response to dictionary
      ${upload_response}=    Set Variable    ${response.json()}
      Log    Parsed upload response: ${upload_response}

      # Validate upload was successful
      Should Be Equal As Strings    ${upload_response['summary']['processing']}    1    Upload failed: No files enqueued
      Should Be Equal As Strings    ${upload_response['files'][0]['status']}    processing    Upload failed: ${response.text}

      # Extract important values
      ${audio_uuid}=    Set Variable    ${upload_response['files'][0]['audio_uuid']}
      ${job_id}=        Set Variable    ${upload_response['files'][0]['job_id']}
      Log    Audio UUID: ${audio_uuid}
      Log    Job ID: ${job_id}

  

      # Wait for conversation to be created and transcribed
      Log    Waiting for transcription to complete...


      Wait Until Keyword Succeeds    60s    5s       Check job status   ${job_id}    completed
      ${job}=    Get Job Details    ${job_id}

     # Get the completed conversation
      ${conversation}=     Get Conversation By ID    ${job}[result][conversation_id]
      Should Exist    ${conversation}    Conversation not found after upload and processing

      Log    Found conversation: ${conversation}
      RETURN    ${conversation}

Conversation Should Be Complete
      [Documentation]    Check if conversation exists and has transcript
      [Arguments]    ${device_name}

      ${conversations}=    Get Conversations For Device    ${device_name}

      # Should have at least one conversation
      ${count}=    Get Length    ${conversations}
      Should Be True    ${count} > 0    No conversations found for device: ${device_name}

      # Check if first conversation has transcript (use segment_count from list endpoint)
      ${conversation}=    Set Variable    ${conversations}[0]
      Should Be True    ${conversation}[segment_count] > 0    Transcript not ready yet (segment_count: ${conversation}[segment_count])

      # Optional: Check if it has memories (if memory processing is expected)
      Log    Conversation ready: ${conversation}[conversation_id]

Get Conversations For Device
      [Documentation]    Get conversations filtered by device name (or return latest if device name not in client_id)
      [Arguments]    ${device_name}

      ${all_conversations}=    GET user conversations

      # Filter conversations by device name - all_conversations is now a flat list
      ${matching_conversations}=    Create List

      FOR    ${conversation}    IN    @{all_conversations}
          ${client_id}=    Set Variable    ${conversation}[client_id]
          # Case-insensitive search - check if device_name (lowercased) is in client_id (lowercased)
          ${device_lower}=    Evaluate    "${device_name}".lower()
          ${client_lower}=    Evaluate    "${client_id}".lower()
          ${is_match}=    Evaluate    "${device_lower}" in "${client_lower}"
          IF    ${is_match}
              Append To List    ${matching_conversations}    ${conversation}
          END
      END

      # If no matches found with device name, return the most recent conversation (fallback)
      ${match_count}=    Get Length    ${matching_conversations}
      IF    ${match_count} == 0
          Log    No conversations found matching device name '${device_name}', using latest conversation as fallback
          ${conv_count}=    Get Length    ${all_conversations}
          IF    ${conv_count} > 0
              ${latest_conversation}=    Set Variable    ${all_conversations}[0]
              Append To List    ${matching_conversations}    ${latest_conversation}
          END
      END

      RETURN    ${matching_conversations}