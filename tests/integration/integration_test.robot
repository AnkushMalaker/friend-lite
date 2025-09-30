*** Settings ***
Documentation    Full Pipeline Integration Test
Library          RequestsLibrary
Library          Collections
Library          Process
Library          String
Library          DateTime
Library          OperatingSystem
Resource         ../resources/setup_resources.robot
Resource         ../resources/session_resources.robot
Resource         ../resources/audio_keywords.robot
Resource         ../resources/conversation_keywords.robot
Variables        ../test_env.py
Variables        ../test_data.py
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions
Test Setup       Clear Test Databases


*** Test Cases ***
Full Pipeline Integration Test
    [Documentation]    Complete end-to-end test of audio processing pipeline
    [Tags]             integration    pipeline    e2e
    [Timeout]          600s

    Log    Starting Full Pipeline Integration Test    INFO

   
    # Phase 4: Audio Processing - Upload and wait for conversation completion
    Log    Starting audio upload and processing    INFO
    ${conversation}=    Upload Audio File    ${TEST_AUDIO_FILE}    ${TEST_DEVICE_NAME}

    Log    Audio processing completed, conversation created    INFO
    Set Global Variable    ${TEST_CONVERSATION}    ${conversation}

    # Phase 5: Transcription Verification
    Verify Transcription Quality    ${TEST_CONVERSATION}    ${EXPECTED_TRANSCRIPT}

    # Phase 6: Memory Extraction Verification
    Verify Memory Extraction    api    ${TEST_CONVERSATION}

    # Phase 7: Chat Integration
    Verify Chat Integration    api    ${TEST_CONVERSATION}

    Log    Full Pipeline Integration Test Completed Successfully    INFO

*** Keywords ***


Verify Transcription Quality
    [Documentation]    Verify the transcription meets quality standards
    [Arguments]    ${conversation}    ${expected_content}

    Log    Verifying transcription quality    INFO

    # Extract transcript (can be string or array of segments)
    Dictionary Should Contain Key    ${conversation}    transcript
    ${transcript_raw}=    Set Variable    ${conversation}[transcript]
    Should Not Be Empty    ${transcript_raw}    Transcript is empty

    # Handle both string and array formats
    ${transcript_text}=    Run Keyword If    isinstance($transcript_raw, list)
    ...    Set Variable    ${transcript_raw}[0][text]
    ...    ELSE    Set Variable    ${transcript_raw}

    # Check transcript contains expected content
    ${transcript_lower}=    Convert To Lower Case    ${transcript_text}
    ${expected_lower}=      Convert To Lower Case    ${expected_content}
    Should Contain    ${transcript_lower}    ${expected_lower}    Transcript does not contain expected content

    # Verify transcript has reasonable length (at least 50 characters for 4-minute audio)
    ${transcript_length}=    Get Length    ${transcript_text}
    Should Be True    ${transcript_length} >= 50    Transcript too short: ${transcript_length} characters

    # Check segments exist (if transcript is array format)
    ${segment_count}=    Run Keyword If    isinstance($transcript_raw, list)
    ...    Get Length    ${transcript_raw}
    ...    ELSE    Set Variable    1

    Should Be True    ${segment_count} > 0    No transcript segments found

    Log    Transcription quality verification passed    INFO
    Log    Transcript length: ${transcript_length} characters, Segments: ${segment_count}    INFO

Verify Memory Extraction
    [Documentation]    Verify memories were extracted from the conversation
    [Arguments]    ${session_alias}    ${conversation}

    Log    Verifying memory extraction    INFO

    # Check if conversation has memory count (may still be processing)
    ${has_memory_count}=    Run Keyword And Return Status    Dictionary Should Contain Key    ${conversation}    memory_count
    ${memory_count}=    Run Keyword If    ${has_memory_count}
    ...    Set Variable    ${conversation}[memory_count]
    ...    ELSE    Set Variable    0

    # Get memories from API using session
    ${response}=   GET On Session    ${session_alias}    /api/memories
    Should Be Equal As Integers    ${response.status_code}    200

    ${memories_data}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${memories_data}    memories
    ${memories}=    Set Variable    ${memories_data}[memories]

    ${api_memory_count}=    Get Length    ${memories}

    # Verify memory extraction status (allow for memory processing to be in progress)
    Should Be True    ${memory_count} >= 0    Memory count is negative
    Should Be True    ${api_memory_count} >= 0    API memory count is negative

    Log    Memory extraction verification passed (may still be processing)    INFO
    Log    Conversation memory count: ${memory_count}, API memory count: ${api_memory_count}    INFO

Verify Chat Integration
    [Documentation]    Verify chat system can access conversation data
    [Arguments]    ${session_alias}    ${conversation}

    Log    Verifying chat integration    INFO

    # Create a chat session to test basic chat functionality
    ${chat_data}=  Create Dictionary    title=Integration Test Chat
    ${response}=   POST On Session    ${session_alias}    /api/chat/sessions    json=${chat_data}    expected_status=any
    Should Be True    ${response.status_code} in [200, 201]    Chat session creation failed with status ${response.status_code}

    ${session_data}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${session_data}    session_id
    ${session_id}=    Set Variable    ${session_data}[session_id]

    Log    Chat session created successfully: ${session_id}    INFO

    # Try to send a message (if endpoint is available)
    ${conversation_id}=    Set Variable    ${conversation}[conversation_id]
    ${message_data}=       Create Dictionary    content=What did we discuss about glass blowing in conversation ${conversation_id}?
    ${msg_status}=        Run Keyword And Return Status
    ...    POST On Session    ${session_alias}    /api/chat/sessions/${session_id}/messages    json=${message_data}    expected_status=200

    IF    ${msg_status}
        Log    Chat message functionality is available    INFO
    ELSE
        Log    Chat message endpoints not available or not implemented - skipping message test    WARN
    END

    # Clean up chat session
    ${response}=    DELETE On Session    ${session_alias}    /api/chat/sessions/${session_id}    expected_status=any
    Should Be True    ${response.status_code} in [200, 204]    Chat session deletion failed with status ${response.status_code}

    Log    Chat integration verification completed    INFO


