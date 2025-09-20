*** Settings ***
Documentation    Conversation Management Keywords
Library          RequestsLibrary
Library          Collections
Library          Process
Variables        ../test_env.py

*** Keywords ***

Get User Conversations
    [Documentation]    Get conversations for authenticated user
    [Arguments]    ${token}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    GET On Session    api    /api/conversations    headers=${headers}
    RETURN    ${response}

Get Conversation By ID
    [Documentation]    Get a specific conversation by ID
    [Arguments]    ${token}    ${conversation_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    GET On Session    api    /api/conversations/${conversation_id}    headers=${headers}
    RETURN    ${response}

Get Conversation Versions
    [Documentation]    Get version history for a conversation
    [Arguments]    ${token}    ${conversation_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    GET On Session    api    /api/conversations/${conversation_id}/versions    headers=${headers}
    RETURN    ${response}

Reprocess Transcript
    [Documentation]    Trigger transcript reprocessing for a conversation
    [Arguments]    ${token}    ${conversation_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    POST On Session    api    /api/conversations/${conversation_id}/reprocess-transcript    headers=${headers}
    RETURN    ${response}

Reprocess Memory
    [Documentation]    Trigger memory reprocessing for a conversation
    [Arguments]    ${token}    ${conversation_id}    ${transcript_version_id}=active
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary    transcript_version_id=${transcript_version_id}

    ${response}=    POST On Session    api    /api/conversations/${conversation_id}/reprocess-memory    headers=${headers}    params=${params}
    RETURN    ${response}

Activate Transcript Version
    [Documentation]    Activate a specific transcript version
    [Arguments]    ${token}    ${conversation_id}    ${version_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    POST On Session    api    /api/conversations/${conversation_id}/activate-transcript/${version_id}    headers=${headers}
    RETURN    ${response}

Activate Memory Version
    [Documentation]    Activate a specific memory version
    [Arguments]    ${token}    ${conversation_id}    ${version_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    POST On Session    api    /api/conversations/${conversation_id}/activate-memory/${version_id}    headers=${headers}
    RETURN    ${response}

Delete Conversation
    [Documentation]    Delete a conversation
    [Arguments]    ${token}    ${audio_uuid}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    DELETE On Session    api    /api/conversations/${audio_uuid}    headers=${headers}
    RETURN    ${response}

Delete Conversation Version
    [Documentation]    Delete a specific version from a conversation
    [Arguments]    ${token}    ${conversation_id}    ${version_type}    ${version_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    DELETE On Session    api    /api/conversations/${conversation_id}/versions/${version_type}/${version_id}    headers=${headers}
    RETURN    ${response}

Close Current Conversation
    [Documentation]    Close the current conversation for a client
    [Arguments]    ${token}    ${client_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    POST On Session    api    /api/conversations/${client_id}/close    headers=${headers}
    RETURN    ${response}

Get Cropped Audio Info
    [Documentation]    Get cropped audio information for a conversation
    [Arguments]    ${token}    ${audio_uuid}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    GET On Session    api    /api/conversations/${audio_uuid}/cropped    headers=${headers}
    RETURN    ${response}

Add Speaker To Conversation
    [Documentation]    Add a speaker to the speakers_identified list
    [Arguments]    ${token}    ${audio_uuid}    ${speaker_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary    speaker_id=${speaker_id}

    ${response}=    POST On Session    api    /api/conversations/${audio_uuid}/speakers    headers=${headers}    params=${params}
    RETURN    ${response}

Update Transcript Segment
    [Documentation]    Update a specific transcript segment
    [Arguments]    ${token}    ${audio_uuid}    ${segment_index}    ${speaker_id}=${None}    ${start_time}=${None}    ${end_time}=${None}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary

    IF    '${speaker_id}' != '${None}'
        Set To Dictionary    ${params}    speaker_id=${speaker_id}
    END
    IF    '${start_time}' != '${None}'
        Set To Dictionary    ${params}    start_time=${start_time}
    END
    IF    '${end_time}' != '${None}'
        Set To Dictionary    ${params}    end_time=${end_time}
    END

    ${response}=    PUT On Session    api    /api/conversations/${audio_uuid}/transcript/${segment_index}    headers=${headers}    params=${params}
    RETURN    ${response}


Create Test Conversation
    [Documentation]    Create a test conversation by processing a test audio file
    [Arguments]    ${token}    ${device_name}=test-device

    # Upload test audio file to create a conversation
    ${test_audio_file}=    Set Variable    /Users/stu/repos/friend-lite/pytest/extras/test-audios/DIY Experts Glass Blowing_16khz_mono_4min.wav

    # Use curl to upload the file (Robot Framework doesn't handle file uploads well)
    ${curl_cmd}=    Set Variable    curl -s -X POST -H "Authorization: Bearer ${token}" -F "files=@${test_audio_file}" -F "device_name=${device_name}" http://localhost:8001/api/process-audio-files
    ${result}=      Run Process    ${curl_cmd}    shell=True

    Should Be Equal As Integers    ${result.rc}    0
    ${response_json}=    Evaluate    json.loads('''${result.stdout}''')    json

    Should Be True    ${response_json}[successful] > 0

    # Wait a moment for processing to complete
    Sleep    2s

    # Now get the actual conversation data
    ${conversations_response}=    Get User Conversations    ${token}
    Should Be Equal As Integers    ${conversations_response.status_code}    200

    ${conversations_data}=    Set Variable    ${conversations_response.json()}[conversations]
    ${client_ids}=            Get Dictionary Keys    ${conversations_data}
    ${count}=                 Get Length    ${client_ids}

    Should Be True    ${count} > 0
    ${first_client_id}=       Set Variable    ${client_ids}[0]
    ${client_conversations}=  Set Variable    ${conversations_data}[${first_client_id}]
    ${first_conv}=            Set Variable    ${client_conversations}[0]

    RETURN    ${first_conv}

Find Test Conversation
    [Documentation]    Find a conversation that exists for testing (returns first available)
    [Arguments]    ${token}
    ${response}=    Get User Conversations    ${token}
    Should Be Equal As Integers    ${response.status_code}    200

    ${conversations_data}=    Set Variable    ${response.json()}[conversations]
    ${client_ids}=            Get Dictionary Keys    ${conversations_data}
    ${count}=                 Get Length    ${client_ids}

    IF    ${count} > 0
        ${first_client_id}=       Set Variable    ${client_ids}[0]
        ${client_conversations}=  Set Variable    ${conversations_data}[${first_client_id}]
        ${conv_count}=            Get Length    ${client_conversations}
        IF    ${conv_count} > 0
            ${first_conv}=    Set Variable    ${client_conversations}[0]
            RETURN    ${first_conv}
        END
    END

    # If no conversations exist, create one
    ${new_conversation}=    Create Test Conversation    ${token}
    RETURN    ${new_conversation}