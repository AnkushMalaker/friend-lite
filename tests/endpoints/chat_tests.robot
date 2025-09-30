*** Settings ***
Documentation    Chat Service API Tests
Library          RequestsLibrary
Library          Collections
Library          String
Resource         ../resources/setup_resources.robot
Resource         ../resources/session_resources.robot
Resource         ../resources/user_resources.robot
Resource         ../resources/chat_keywords.robot
Suite Setup      Suite Setup
Suite Teardown   Suite Teardown

*** Test Cases ***

Create Chat Session Test
    [Documentation]    Test creating a new chat session
    [Tags]             chat    session    create    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${session}=        Create Test Chat Session

    # Verify chat session structure
    Dictionary Should Contain Key    ${session}    session_id
    Dictionary Should Contain Key    ${session}    title
    Dictionary Should Contain Key    ${session}    created_at
    Dictionary Should Contain Key    ${session}    updated_at
    Should Contain     ${session}[title]    Test Session

    # Cleanup
    Cleanup Test Chat Session    ${session}[session_id]

Create Chat Session With Custom Title Test
    [Documentation]    Test creating chat session with custom title
    [Tags]             chat    session    create    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${custom_title}=   Set Variable    Custom Chat Title ${RANDOM_ID}
    ${response}=       Create Chat Session    ${custom_title}

    Should Be Equal As Integers    ${response.status_code}    200
    ${session}=        Set Variable    ${response.json()}
    Should Be Equal    ${session}[title]    ${custom_title}

    # Cleanup
    Cleanup Test Chat Session    ${session}[session_id]

Get Chat Sessions Test
    [Documentation]    Test getting all chat sessions for user
    [Tags]             chat    session    list    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Create a test session first
    ${test_session}=   Create Test Chat Session

    # Get all sessions
    ${response}=       Get Chat Sessions
    Should Be Equal As Integers    ${response.status_code}    200

    ${sessions}=       Set Variable    ${response.json()}
    Should Be True     isinstance($sessions, list)

    # Should contain our test session
    ${found}=          Set Variable    ${False}
    FOR    ${session}    IN    @{sessions}
        # Verify chat session structure
    Dictionary Should Contain Key    ${session}    session_id
    Dictionary Should Contain Key    ${session}    title
    Dictionary Should Contain Key    ${session}    created_at
    Dictionary Should Contain Key    ${session}    updated_at
        IF    '${session}[session_id]' == '${test_session}[session_id]'
            ${found}=    Set Variable    ${True}
        END
    END
    Should Be True    ${found}

    # Cleanup
    Cleanup Test Chat Session    ${test_session}[session_id]

Get Specific Chat Session Test
    [Documentation]    Test getting a specific chat session
    [Tags]             chat    session    individual    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${test_session}=   Create Test Chat Session

    ${response}=       GET On Session    admin_session    /api/chat/sessions/${test_session}[session_id]
    Should Be Equal As Integers    ${response.status_code}    200

    ${session}=        Set Variable    ${response.json()}
    # Verify chat session structure
    Dictionary Should Contain Key    ${session}    session_id
    Dictionary Should Contain Key    ${session}    title
    Dictionary Should Contain Key    ${session}    created_at
    Dictionary Should Contain Key    ${session}    updated_at
    Should Be Equal    ${session}[session_id]    ${test_session}[session_id]

    # Cleanup
    Cleanup Test Chat Session    ${test_session}[session_id]

Update Chat Session Test
    [Documentation]    Test updating a chat session title
    [Tags]             chat    session    update    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${test_session}=   Create Test Chat Session

    ${new_title}=      Set Variable    Updated Title ${RANDOM_ID}
    ${response}=       Update Chat Session    ${test_session}[session_id]    ${new_title}

    Should Be Equal As Integers    ${response.status_code}    200
    ${updated_session}=    Set Variable    ${response.json()}
    Should Be Equal    ${updated_session}[title]    ${new_title}

    # Cleanup
    Cleanup Test Chat Session    ${test_session}[session_id]

Delete Chat Session Test
    [Documentation]    Test deleting a chat session
    [Tags]             chat    session    delete    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${test_session}=   Create Test Chat Session

    ${response}=       Delete Chat Session    ${test_session}[session_id]
    Should Be Equal As Integers    ${response.status_code}    200

    # Verify session is deleted
    ${response}=       Get Chat Session    ${test_session}[session_id] 
    Should Be Equal As Integers    ${response.status_code}    404

Get Session Messages Test
    [Documentation]    Test getting messages from a chat session
    [Tags]             chat    messages    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${test_session}=   Create Test Chat Session

    ${response}=       Get Session Messages    ${test_session}[session_id]
    Should Be Equal As Integers    ${response.status_code}    200

    ${messages}=       Set Variable    ${response.json()}
    Should Be True     isinstance($messages, list)

    # New session should have no messages
    ${count}=          Get Length    ${messages}
    Should Be Equal As Integers    ${count}    0

    # Cleanup
    Cleanup Test Chat Session    ${test_session}[session_id]

Get Chat Statistics Test
    [Documentation]    Test getting chat statistics for user
    [Tags]             chat    statistics    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       Get Chat Statistics

    Should Be Equal As Integers    ${response.status_code}    200
    ${stats}=          Set Variable    ${response.json()}
    # Verify chat statistics structure
    Dictionary Should Contain Key    ${stats}    total_sessions
    Dictionary Should Contain Key    ${stats}    total_messages

    # Statistics should be non-negative
    Should Be True     ${stats}[total_sessions] >= 0
    Should Be True     ${stats}[total_messages] >= 0

Chat Session Pagination Test
    [Documentation]    Test chat session pagination
    [Tags]             chat    session    pagination    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Test with different limits
    ${response1}=      Get Chat Sessions    5
    Should Be Equal As Integers    ${response1.status_code}    200

    ${response2}=      Get Chat Sessions    50
    Should Be Equal As Integers    ${response2.status_code}    200

    ${sessions1}=      Set Variable    ${response1.json()}
    ${sessions2}=      Set Variable    ${response2.json()}
    ${count1}=         Get Length    ${sessions1}
    ${count2}=         Get Length    ${sessions2}

    # Second request should have >= first request count
    Should Be True     ${count2} >= ${count1}

Unauthorized Chat Access Test
    [Documentation]    Test that chat endpoints require authentication
    [Tags]             chat    security    negative
    Get Anonymous Session    session

    # Try to access sessions without token
    ${response}=    GET On Session    session    /api/chat/sessions    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

    # Try to get statistics without token
    ${response}=    GET On Session    session    /api/chat/statistics    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

Non-Existent Session Operations Test
    [Documentation]    Test operations on non-existent chat sessions
    [Tags]             chat    session    negative    notfound
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${fake_id}=        Set Variable    non-existent-session-id

    # Try to get non-existent session
    ${response}=       Get Chat Session    ${fake_id}    404
    Should Be Equal As Integers    ${response.status_code}    404

    # Try to update non-existent session
    ${response}=       Update Chat Session    ${fake_id}    New Title    404
    Should Be Equal As Integers    ${response.status_code}    404

    # Try to delete non-existent session
    ${response}=       Delete Chat Session    ${fake_id}    404
    Should Be Equal As Integers    ${response.status_code}    404

    # Try to get messages from non-existent session
    ${response}=       Get Session Messages    ${fake_id}    expected_status=404
    Should Be Equal As Integers    ${response.status_code}    404

Invalid Chat Session Data Test
    [Documentation]    Test creating chat session with invalid data
    [Tags]             chat    session    negative    validation
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Test with title too long (over 200 characters)
    ${long_title}=     Generate Random String    201    [LETTERS]
    ${response}=       Create Chat Session    ${long_title}    422
    Should Be Equal As Integers    ${response.status_code}    422

    # Test updating with empty title
    ${test_session}=   Create Test Chat Session
    ${response}=       Update Chat Session    ${test_session}[session_id]    ${EMPTY}    422
    Should Be Equal As Integers    ${response.status_code}    422

    # Cleanup
    Cleanup Test Chat Session    ${test_session}[session_id]

User Isolation Test
    [Documentation]    Test that users can only access their own chat sessions
    [Tags]             chat    security    isolation
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Create a test user
    ${test_user}=      Create Test User    admin_session    test-user-${RANDOM_ID}@example.com    test-password-123
    Create API Session    user_session    email=test-user-${RANDOM_ID}@example.com    password=test-password-123

    # Create session as admin
    ${admin_chat_session}=  Create Test Chat Session

    # User should not be able to access admin's session
    ${response}=       GET On Session    user_session    /api/chat/sessions/${admin_chat_session}[session_id]    expected_status=404
    Should Be Equal As Integers    ${response.status_code}    404

    # User should see empty session list
    ${user_sessions}=  GET On Session    user_session    /api/chat/sessions
    Should Be Equal As Integers    ${user_sessions.status_code}    200
    ${sessions}=       Set Variable    ${user_sessions.json()}
    ${count}=          Get Length    ${sessions}
    Should Be Equal As Integers    ${count}    0

    # Cleanup
    Cleanup Test Chat Session    ${admin_chat_session}[session_id]

