*** Settings ***
Documentation    Chat Service Keywords
Library          RequestsLibrary
Library          Collections
Variables        ../test_env.py

*** Keywords ***

Create Chat Session
    [Documentation]    Create a new chat session
    [Arguments]    ${token}    ${title}=${None}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}    Content-Type=application/json
    &{data}=       Create Dictionary

    IF    '${title}' != '${None}'
        Set To Dictionary    ${data}    title=${title}
    END

    ${response}=    POST On Session    api    /api/chat/sessions    json=${data}    headers=${headers}
    RETURN    ${response}

Get Chat Sessions
    [Documentation]    Get all chat sessions for user
    [Arguments]    ${token}    ${limit}=50
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary    limit=${limit}

    ${response}=    GET On Session    api    /api/chat/sessions    headers=${headers}    params=${params}
    RETURN    ${response}

Get Chat Session
    [Documentation]    Get a specific chat session
    [Arguments]    ${token}    ${session_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    GET On Session    api    /api/chat/sessions/${session_id}    headers=${headers}
    RETURN    ${response}

Update Chat Session
    [Documentation]    Update a chat session title
    [Arguments]    ${token}    ${session_id}    ${new_title}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}    Content-Type=application/json
    &{data}=       Create Dictionary    title=${new_title}

    ${response}=    PUT On Session    api    /api/chat/sessions/${session_id}    json=${data}    headers=${headers}
    RETURN    ${response}

Delete Chat Session
    [Documentation]    Delete a chat session
    [Arguments]    ${token}    ${session_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    DELETE On Session    api    /api/chat/sessions/${session_id}    headers=${headers}
    RETURN    ${response}

Get Session Messages
    [Documentation]    Get messages from a chat session
    [Arguments]    ${token}    ${session_id}    ${limit}=100
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary    limit=${limit}

    ${response}=    GET On Session    api    /api/chat/sessions/${session_id}/messages    headers=${headers}    params=${params}
    RETURN    ${response}

Send Chat Message
    [Documentation]    Send a message to chat (non-streaming)
    [Arguments]    ${token}    ${message}    ${session_id}=${None}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}    Content-Type=application/json
    &{data}=       Create Dictionary    message=${message}

    IF    '${session_id}' != '${None}'
        Set To Dictionary    ${data}    session_id=${session_id}
    END

    # Note: This endpoint streams, so we expect a streaming response
    ${response}=    POST On Session    api    /api/chat/send    json=${data}    headers=${headers}
    RETURN    ${response}

Get Chat Statistics
    [Documentation]    Get chat statistics for user
    [Arguments]    ${token}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    GET On Session    api    /api/chat/statistics    headers=${headers}
    RETURN    ${response}

Extract Memories From Session
    [Documentation]    Extract memories from a chat session
    [Arguments]    ${token}    ${session_id}
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=    POST On Session    api    /api/chat/sessions/${session_id}/extract-memories    headers=${headers}
    RETURN    ${response}


Create Test Chat Session
    [Documentation]    Create a test chat session with random title
    [Arguments]    ${token}    ${title_prefix}=Test Session
    ${random_suffix}=    Generate Random String    6    [LETTERS][NUMBERS]
    ${title}=            Set Variable    ${title_prefix} ${random_suffix}

    ${response}=         Create Chat Session    ${token}    ${title}
    Should Be Equal As Integers    ${response.status_code}    200

    RETURN    ${response.json()}

Cleanup Test Chat Session
    [Documentation]    Clean up a test chat session
    [Arguments]    ${token}    ${session_id}
    ${response}=    Delete Chat Session    ${token}    ${session_id}
    Should Be Equal As Integers    ${response.status_code}    200