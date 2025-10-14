*** Settings ***
Documentation    Core integration workflow and system interaction keywords
...
...              This file contains keywords for complex multi-step operations that combine
...              multiple services. Keywords in this file should handle integration workflows,
...              file processing, and system interactions that don't fit other categories.
...
...              Examples of keywords that belong here:
...              - Complex multi-step operations combining services
...              - File processing and upload operations
...              - Integration workflow keywords
...              - System interaction keywords
...
...              Keywords that should NOT be in this file:
...              - Simple verification/assertion keywords (belong in tests)
...              - User management operations (belong in user_resources.robot)
...              - API session management (belong in session_resources.robot)
...              - Docker service management (belong in setup_resources.robot)
Library          RequestsLibrary
Library          Collections
Library          Process
Library          String
Library          DateTime
Library          OperatingSystem
Variables        ../test_env.py
Resource         setup_resources.robot
Resource         session_resources.robot
# Library          JSONLibrary  # Optional library, not required

*** Keywords ***




Upload Audio File For Processing
    [Documentation]    Upload audio file and return processing result
    [Arguments]    ${session_alias}    ${audio_file_path}    ${device_name}=test-device

    # Verify file exists
    File Should Exist    ${audio_file_path}

    # For now, get a fresh token for curl (Robot Framework doesn't easily expose session headers)
    ${token}=    Get Token From Session    ${session_alias}

    # Use curl for file upload (Robot Framework multipart is problematic)
    ${curl_cmd}=    Catenate    SEPARATOR=
    ...    curl -s -X POST
    ...    ${SPACE}-H "Authorization: Bearer ${token}"
    ...    ${SPACE}-F "files=@${audio_file_path}"
    ...    ${SPACE}-F "device_name=${device_name}"
    ...    ${SPACE}${API_URL}/api/process-audio-files

    ${result}=    Run Process    ${curl_cmd}    shell=True    timeout=300

    Should Be Equal As Integers    ${result.rc}    0    File upload failed: ${result.stderr}

    ${response_data}=    Evaluate    json.loads('''${result.stdout}''')    json
    Should Be True    ${response_data}[successful] > 0    No files processed successfully

    RETURN    ${response_data}

Wait For Audio Processing
    [Documentation]    Wait for audio processing to complete
    [Arguments]    ${processing_delay}=10s

    Log    Waiting ${processing_delay} for audio processing to complete    INFO
    Sleep    ${processing_delay}

Get Latest Conversation
    [Documentation]    Get the most recent conversation for a device
    [Arguments]    ${session_alias}    ${device_name}

    ${response}=    GET On Session    ${session_alias}    /api/conversations    expected_status=200
    ${conversations_list}=    Set Variable    ${response.json()}[conversations]

    # Find conversation for the specified device - conversations_list is now a flat list
    FOR    ${conversation}    IN    @{conversations_list}
        ${client_id}=    Set Variable    ${conversation}[client_id]
        ${is_target_device}=    Evaluate    "${device_name}" in "${client_id}"
        IF    ${is_target_device}
            RETURN    ${conversation}
        END
    END

    Fail    No conversation found for device: ${device_name}

Verify Transcript Content
    [Documentation]    Verify transcript contains expected content and quality
    [Arguments]    ${conversation}    ${expected_keywords}    ${min_length}=50

    Dictionary Should Contain Key    ${conversation}    transcript
    ${transcript}=    Set Variable    ${conversation}[transcript]
    Should Not Be Empty    ${transcript}

    # Check length
    ${transcript_length}=    Get Length    ${transcript}
    Should Be True    ${transcript_length} >= ${min_length}    Transcript too short: ${transcript_length}

    # Check for expected keywords
    ${transcript_lower}=    Convert To Lower Case    ${transcript}
    FOR    ${keyword}    IN    @{expected_keywords}
        ${keyword_lower}=    Convert To Lower Case    ${keyword}
        Should Contain    ${transcript_lower}    ${keyword_lower}    Missing keyword: ${keyword}
    END

    # Verify segments exist
    Dictionary Should Contain Key    ${conversation}    segments
    ${segments}=    Set Variable    ${conversation}[segments]
    ${segment_count}=    Get Length    ${segments}
    Should Be True    ${segment_count} > 0    No segments found

    Log    Transcript verification passed: ${transcript_length} chars, ${segment_count} segments    INFO

Get User Memories
    [Documentation]    Get all memories for the authenticated user
    [Arguments]    ${session_alias}

    ${response}=    GET On Session    ${session_alias}    /api/memories    expected_status=200
    ${memories_data}=    Set Variable    ${response.json()}

    RETURN    ${memories_data}

Verify Memory Extraction
    [Documentation]    Verify memories were extracted successfully
    [Arguments]    ${conversation}    ${memories_data}    ${min_memories}=0

    # Check conversation memory count
    Dictionary Should Contain Key    ${conversation}    memory_count
    ${conv_memory_count}=    Set Variable    ${conversation}[memory_count]

    # Check API memories
    Dictionary Should Contain Key    ${memories_data}    memories
    ${memories}=    Set Variable    ${memories_data}[memories]
    ${api_memory_count}=    Get Length    ${memories}

    # Verify reasonable memory extraction
    Should Be True    ${conv_memory_count} >= ${min_memories}    Insufficient memories: ${conv_memory_count}
    Should Be True    ${api_memory_count} >= ${min_memories}    Insufficient API memories: ${api_memory_count}

    Log    Memory extraction verified: conversation=${conv_memory_count}, api=${api_memory_count}    INFO

Create Test Chat Session
    [Documentation]    Create a chat session for testing
    [Arguments]    ${base_url}    ${token}    ${title}=Integration Test Chat

    Create Session    api    ${base_url}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${chat_data}=    Create Dictionary    title=${title}
    ${response}=     POST On Session    api    /api/chat/sessions    headers=${headers}    json=${chat_data}    expected_status=201

    ${session_data}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${session_data}    session_id

    Delete All Sessions    api
    RETURN    ${session_data}

Send Chat Message
    [Documentation]    Send a message to a chat session
    [Arguments]    ${base_url}    ${token}    ${session_id}    ${message_content}

    Create Session    api    ${base_url}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${message_data}=    Create Dictionary    content=${message_content}
    ${response}=        POST On Session    api    /api/chat/sessions/${session_id}/messages    headers=${headers}    json=${message_data}    expected_status=201

    ${message_response}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${message_response}    message_id

    Delete All Sessions    api
    RETURN    ${message_response}

Delete Chat Session
    [Documentation]    Delete a chat session
    [Arguments]    ${base_url}    ${token}    ${session_id}

    Create Session    api    ${base_url}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    DELETE On Session    api    /api/chat/sessions/${session_id}    headers=${headers}    expected_status=204
    Delete All Sessions    api

Check Environment Variables
    [Documentation]    Check required environment variables and return missing ones
    [Arguments]    @{required_vars}

    @{missing_vars}=    Create List
    FOR    ${var}    IN    @{required_vars}
        ${value}=    Get Environment Variable    ${var}    ${EMPTY}
        IF    '${value}' == '${EMPTY}'
            Append To List    ${missing_vars}    ${var}
        ELSE
            Log    Environment variable ${var} is set    DEBUG
        END
    END
    RETURN    ${missing_vars}

Log Test Phase
    [Documentation]    Log the current test phase with timing
    [Arguments]    ${phase_name}

    ${timestamp}=    Get Current Date    result_format=%Y-%m-%d %H:%M:%S
    Log    === PHASE: ${phase_name} (${timestamp}) ===    INFO

Measure Processing Time
    [Documentation]    Measure and log processing time for an operation
    [Arguments]    ${operation_name}    ${start_time}

    ${end_time}=        Get Current Date    result_format=epoch
    ${duration}=        Evaluate    ${end_time} - ${start_time}
    ${duration_str}=    Convert To String    ${duration}

    Log    ${operation_name} completed in ${duration_str} seconds    INFO
    RETURN    ${duration}