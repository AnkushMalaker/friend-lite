*** Settings ***
Documentation    Chat Service Keywords
Library          RequestsLibrary
Library          Collections
Variables        ../test_env.py
Resource         session_resources.robot

*** Keywords ***

Create Chat Session
    [Documentation]    Create a new chat session (uses admin session)
    [Arguments]    ${title}=${None}    ${expected_status}=200

    # Get admin session
    Create API Session    session

    &{data}=       Create Dictionary

    IF    '${title}' != '${None}'
        Set To Dictionary    ${data}    title=${title}
    END

    ${response}=    POST On Session    admin_session    /api/chat/sessions    json=${data}    expected_status=${expected_status}
    RETURN    ${response}

Get Chat Sessions
    [Documentation]    Get all chat sessions for user (uses admin session)
    [Arguments]    ${limit}=50

    # Get admin session
    Create API Session    admin_session

    &{params}=     Create Dictionary    limit=${limit}
    ${response}=    GET On Session    admin_session    /api/chat/sessions    params=${params}
    RETURN    ${response}


Delete Chat Session
    [Documentation]    Delete a chat session (uses admin session)
    [Arguments]    ${session_id}    ${expected_status}=200

    # Get admin session
    Create API Session    session

    ${response}=    DELETE On Session    session    /api/chat/sessions/${session_id}    expected_status=${expected_status}
    RETURN    ${response}



Create Test Chat Session
    [Documentation]    Create a test chat session with random title (uses admin session)
    [Arguments]    ${title_prefix}=Test Session
    ${random_suffix}=    Generate Random String    6    [LETTERS][NUMBERS]
    ${title}=            Set Variable    ${title_prefix} ${random_suffix}

    ${response}=         Create Chat Session    ${title}
    Should Be Equal As Integers    ${response.status_code}    200

    RETURN    ${response.json()}

Cleanup Test Chat Session
    [Documentation]    Clean up a test chat session (uses admin session)
    [Arguments]    ${session_id}
    ${response}=    Delete Chat Session    ${session_id}
    Should Be Equal As Integers    ${response.status_code}    200

Get Chat Session
    [Documentation]    Get a specific chat session (uses admin session)
    [Arguments]    ${session_id}    ${expected_status}=200

    # Get admin session
    Create API Session    admin_session

    ${response}=    GET On Session    admin_session    /api/chat/sessions/${session_id}    expected_status=${expected_status}
    RETURN    ${response}

Get Session Messages
    [Documentation]    Get messages from a chat session (uses admin session)
    [Arguments]    ${session_id}    ${limit}=100    ${expected_status}=200

    # Get admin session
    Create API Session    admin_session

    &{params}=    Create Dictionary    limit=${limit}
    ${response}=    GET On Session    admin_session    /api/chat/sessions/${session_id}/messages    params=${params}    expected_status=${expected_status}
    RETURN    ${response}

Update Chat Session
    [Documentation]    Update a chat session title (uses admin session)
    [Arguments]    ${session_id}    ${new_title}    ${expected_status}=200

    # Get admin session
    Create API Session    admin_session

    &{data}=    Create Dictionary    title=${new_title}
    ${response}=    PUT On Session    admin_session    /api/chat/sessions/${session_id}    json=${data}    expected_status=${expected_status}
    RETURN    ${response}

Get Chat Statistics
    [Documentation]    Get chat statistics for user (uses admin session)

    # Get admin session
    Create API Session    admin_session

    ${response}=    GET On Session    admin_session    /api/chat/statistics
    RETURN    ${response}

