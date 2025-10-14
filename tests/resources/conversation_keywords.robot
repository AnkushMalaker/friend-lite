*** Settings ***
Documentation    Conversation Management Keywords
Library          RequestsLibrary
Library          Collections
Library          Process
Resource         session_resources.robot
Resource         audio_keywords.robot


*** Keywords ***

Get User Conversations
    [Documentation]    Get conversations for authenticated user (uses admin session)

    ${response}=    GET On Session    api    /api/conversations    expected_status=200
    RETURN    ${response.json()}[conversations]

Get Conversation By ID
    [Documentation]    Get a specific conversation by ID
    [Arguments]       ${conversation_id}
    ${response}=    GET On Session    api    /api/conversations/${conversation_id} 
    RETURN    ${response.json()}[conversation]

# Get Conversation Versions
#     [Documentation]    Get version history for a conversation
#     [Arguments]    ${conversation_id}
#     ${response}=    GET On Session    api    /api/conversations/${conversation_id}/versions 
#     RETURN    ${response.json()}[versions]

Reprocess Transcript
    [Documentation]    Trigger transcript reprocessing for a conversation
    [Arguments]     ${conversation_id}

    ${response}=    POST On Session    api    /api/conversations/${conversation_id}/reprocess-transcript   
    Should Be Equal As Integers    ${response.status_code}    200

    ${reprocess_data}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${reprocess_data}    job_id
    Dictionary Should Contain Key    ${reprocess_data}    status

    ${job_id}=    Set Variable    ${reprocess_data}[job_id]
    ${initial_status}=    Set Variable    ${reprocess_data}[status]

    Log    Reprocess job created: ${job_id} with status: ${initial_status}    INFO
    Should Be Equal As Strings    ${initial_status}    queued

    RETURN    ${response.json()}

Reprocess Memory
    [Documentation]    Trigger memory reprocessing for a conversation
    [Arguments]    ${conversation_id}    ${transcript_version_id}=active
    &{params}=     Create Dictionary    transcript_version_id=${transcript_version_id}

    ${response}=    POST On Session    api    /api/conversations/${conversation_id}/reprocess-memory    headers=${headers}    params=${params}
    RETURN    ${response.json()}

Activate Transcript Version
    [Documentation]    Activate a specific transcript version
    [Arguments]    ${conversation_id}    ${version_id}

    ${response}=    POST On Session    api    /api/conversations/${conversation_id}/activate-transcript/${version_id}    headers=${headers}
    RETURN    ${response.json()}

Activate Memory Version
    [Documentation]    Activate a specific memory version
    [Arguments]     ${conversation_id}    ${version_id}

    ${response}=    POST On Session    api    /api/conversations/${conversation_id}/activate-memory/${version_id}    headers=${headers}
    RETURN    ${response.json()}

Delete Conversation
    [Documentation]    Delete a conversation
    [Arguments]     ${audio_uuid}

    ${response}=    DELETE On Session    api    /api/conversations/${audio_uuid}    headers=${headers}
    RETURN    ${response.json()}

Delete Conversation Version
    [Documentation]    Delete a specific version from a conversation
    [Arguments]     ${conversation_id}    ${version_type}    ${version_id}

    ${response}=    DELETE On Session    api    /api/conversations/${conversation_id}/versions/${version_type}/${version_id}    headers=${headers}
    RETURN    ${response.json()}

Close Current Conversation
    [Documentation]    Close the current conversation for a client
    [Arguments]    ${client_id}

    ${response}=    POST On Session    api    /api/conversations/${client_id}/close    headers=${headers}
    RETURN    ${response.json()}

Get Cropped Audio Info
    [Documentation]    Get cropped audio information for a conversation
    [Arguments]     ${audio_uuid}

    ${response}=    GET On Session    api    /api/conversations/${audio_uuid}/cropped    headers=${headers}
    RETURN    ${response.json()}[cropped_audios]    

Add Speaker To Conversation
    [Documentation]    Add a speaker to the speakers_identified list
    [Arguments]    ${audio_uuid}    ${speaker_id}
    &{params}=     Create Dictionary    speaker_id=${speaker_id}

    ${response}=    POST On Session    api    /api/conversations/${audio_uuid}/speakers    headers=${headers}    params=${params}
    RETURN    ${response.json()}

Update Transcript Segment
    [Documentation]    Update a specific transcript segment
    [Arguments]    ${audio_uuid}    ${segment_index}    ${speaker_id}=${None}    ${start_time}=${None}    ${end_time}=${None}
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
    RETURN    ${response.json()}


Create Test Conversation
    [Documentation]    Create a test conversation by processing a test audio file
    [Arguments]     ${device_name}=test-device

    # Upload test audio file to create a conversation
    ${test_audio_file}=    Set Variable    test-assets/DIY_Experts_Glass_Blowing_16khz_mono_4min.wav

    ${conversation}=    Upload Audio File     ${test_audio_file}    ${device_name}

    RETURN    ${conversation}

Find Test Conversation
    [Documentation]    Find a conversation that exists for testing (uses admin session)
    ${conversations_data}=    Get User Conversations
    Log    Retrieved conversations data: ${conversations_data}

    # conversations_data is now a flat list
    ${count}=    Get Length    ${conversations_data}

    IF    ${count} > 0
        ${first_conv}=    Set Variable    ${conversations_data}[0]
        RETURN    ${first_conv}
    END

    # If no conversations exist, return None (let tests handle appropriately)
    RETURN    ${None}

