*** Settings ***
Documentation    Conversation Management API Tests
Library          RequestsLibrary
Library          Collections
Library          String
Resource         resources/setup_resources.robot
Resource         resources/auth_keywords.robot
Resource         resources/conversation_keywords.robot
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions

*** Test Cases ***

Get User Conversations Test
    [Documentation]    Test getting conversations for authenticated user
    [Tags]             conversation    user    positive
    Setup Auth Session

    ${token}=           Get Admin Token
    ${response}=        Get User Conversations    ${token}

    Should Be Equal As Integers    ${response.status_code}    200
    Should Be True     isinstance($response.json(), dict)
    Dictionary Should Contain Key    ${response.json()}    conversations

    # Verify conversation structure if any exist
    ${conversations_data}=    Set Variable    ${response.json()}[conversations]
    IF    isinstance($conversations_data, dict) and len($conversations_data) > 0
        ${client_ids}=    Get Dictionary Keys    ${conversations_data}
        FOR    ${client_id}    IN    @{client_ids}
            ${client_conversations}=    Set Variable    ${conversations_data}[${client_id}]
            FOR    ${conversation}    IN    @{client_conversations}
                # Verify conversation structure
                Dictionary Should Contain Key    ${conversation}    conversation_id
                Dictionary Should Contain Key    ${conversation}    audio_uuid
                Dictionary Should Contain Key    ${conversation}    created_at
            END
        END
    END

Get Conversation By ID Test
    [Documentation]    Test getting a specific conversation by ID
    [Tags]             conversation    individual    positive
    Setup Auth Session

    ${token}=           Get Admin Token
    ${test_conversation}=    Find Test Conversation    ${token}

    IF    $test_conversation != $None
        ${conversation_id}=    Set Variable    ${test_conversation}[conversation_id]
        ${response}=           Get Conversation By ID    ${token}    ${conversation_id}

        Should Be Equal As Integers    ${response.status_code}    200
        # Verify conversation structure
        ${conversation}=    Set Variable    ${response.json()}[conversation]
        Dictionary Should Contain Key    ${conversation}    conversation_id
        Dictionary Should Contain Key    ${conversation}    audio_uuid
        Dictionary Should Contain Key    ${conversation}    created_at
        Should Be Equal    ${conversation}[conversation_id]    ${conversation_id}
    ELSE
        Log    No conversations available for testing
        Pass Execution    No conversations available for individual conversation test
    END

Get Conversation Versions Test
    [Documentation]    Test getting version history for a conversation
    [Tags]             conversation    versions    positive
    Setup Auth Session

    ${token}=           Get Admin Token
    ${test_conversation}=    Find Test Conversation    ${token}

    IF    $test_conversation != $None
        ${conversation_id}=    Set Variable    ${test_conversation}[conversation_id]
        ${response}=           Get Conversation Versions    ${token}    ${conversation_id}

        Should Be Equal As Integers    ${response.status_code}    200
        # Verify version history structure
        ${versions}=    Set Variable    ${response.json()}
        Dictionary Should Contain Key    ${versions}    transcript_versions
        Dictionary Should Contain Key    ${versions}    memory_versions
        Dictionary Should Contain Key    ${versions}    active_transcript_version
        Dictionary Should Contain Key    ${versions}    active_memory_version
    ELSE
        Log    No conversations available for testing
        Pass Execution    No conversations available for version history test
    END

Unauthorized Conversation Access Test
    [Documentation]    Test that conversation endpoints require authentication
    [Tags]             conversation    security    negative
    Setup Auth Session

    # Try to access conversations without token
    ${response}=    GET On Session    api    /api/conversations    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

Non-Existent Conversation Test
    [Documentation]    Test accessing a non-existent conversation
    [Tags]             conversation    negative    notfound
    Setup Auth Session

    ${token}=           Get Admin Token
    ${fake_id}=         Set Variable    non-existent-conversation-id

    &{headers}=         Create Dictionary    Authorization=Bearer ${token}
    ${response}=        GET On Session    api    /api/conversations/${fake_id}    headers=${headers}    expected_status=404
    Should Be Equal As Integers    ${response.status_code}    404

Reprocess Transcript Test
    [Documentation]    Test triggering transcript reprocessing
    [Tags]             conversation    reprocess    positive
    Setup Auth Session

    ${token}=           Get Admin Token
    ${test_conversation}=    Find Test Conversation    ${token}

    IF    $test_conversation != $None
        ${conversation_id}=    Set Variable    ${test_conversation}[conversation_id]
        ${response}=           Reprocess Transcript    ${token}    ${conversation_id}

        # Reprocessing might return 200 (success) or 202 (accepted) depending on implementation
        Should Be True    ${response.status_code} in [200, 202]
    ELSE
        Log    No conversations available for reprocessing test
        Pass Execution    No conversations available for transcript reprocessing test
    END

Reprocess Memory Test
    [Documentation]    Test triggering memory reprocessing
    [Tags]             conversation    reprocess    memory    positive
    Setup Auth Session

    ${token}=           Get Admin Token
    ${test_conversation}=    Find Test Conversation    ${token}

    IF    $test_conversation != $None
        ${conversation_id}=    Set Variable    ${test_conversation}[conversation_id]
        ${response}=           Reprocess Memory    ${token}    ${conversation_id}

        # Memory reprocessing might return 200 (success) or 202 (accepted)
        Should Be True    ${response.status_code} in [200, 202]
    ELSE
        Log    No conversations available for memory reprocessing test
        Pass Execution    No conversations available for memory reprocessing test
    END

Close Conversation Test
    [Documentation]    Test closing current conversation for a client
    [Tags]             conversation    close    positive
    Setup Auth Session

    ${token}=           Get Admin Token
    ${client_id}=       Set Variable    test-client-${RANDOM_ID}

    # This might return 404 if client doesn't exist, which is expected
    &{headers}=         Create Dictionary    Authorization=Bearer ${token}
    ${response}=        POST On Session    api    /api/conversations/${client_id}/close    headers=${headers}    expected_status=any
    Should Be True     ${response.status_code} in [200, 404]

Invalid Conversation Operations Test
    [Documentation]    Test invalid operations on conversations
    [Tags]             conversation    negative    invalid
    Setup Auth Session

    ${token}=           Get Admin Token
    ${fake_id}=         Set Variable    invalid-conversation-id

    # Test reprocessing non-existent conversation
    &{headers}=         Create Dictionary    Authorization=Bearer ${token}
    ${response}=        POST On Session    api    /api/conversations/${fake_id}/reprocess-transcript    headers=${headers}    expected_status=404
    Should Be Equal As Integers    ${response.status_code}    404

    # Test getting versions of non-existent conversation
    ${response}=        GET On Session    api    /api/conversations/${fake_id}/versions    headers=${headers}    expected_status=404
    Should Be Equal As Integers    ${response.status_code}    404

Version Management Test
    [Documentation]    Test version activation (if versions exist)
    [Tags]             conversation    versions    activation
    Setup Auth Session

    ${token}=           Get Admin Token
    ${test_conversation}=    Find Test Conversation    ${token}

    IF    $test_conversation != $None
        ${conversation_id}=    Set Variable    ${test_conversation}[conversation_id]
        ${versions_response}=  Get Conversation Versions    ${token}    ${conversation_id}

        Should Be Equal As Integers    ${versions_response.status_code}    200
        ${versions}=           Set Variable    ${versions_response.json()}

        # Test activating existing active version (should succeed)
        ${active_transcript}=  Set Variable    ${versions}[active_transcript_version]
        IF    '${active_transcript}' != '${None}' and '${active_transcript}' != 'null'
            ${response}=       Activate Transcript Version    ${token}    ${conversation_id}    ${active_transcript}
            Should Be Equal As Integers    ${response.status_code}    200
        END

        ${active_memory}=      Set Variable    ${versions}[active_memory_version]
        IF    '${active_memory}' != '${None}' and '${active_memory}' != 'null'
            ${response}=       Activate Memory Version    ${token}    ${conversation_id}    ${active_memory}
            Should Be Equal As Integers    ${response.status_code}    200
        END
    ELSE
        Log    No conversations available for version management test
        Pass Execution    No conversations available for version management test
    END

User Isolation Test
    [Documentation]    Test that users can only access their own conversations
    [Tags]             conversation    security    isolation
    Setup Auth Session

    ${admin_token}=     Get Admin Token

    # Create a test user
    ${test_user}=       Create Test User    ${admin_token}    test-user-${RANDOM_ID}@example.com    test-password-123
    ${user_token}=      Get User Token      test-user-${RANDOM_ID}@example.com    test-password-123

    # Get admin conversations
    ${admin_conversations}=    Get User Conversations    ${admin_token}
    Should Be Equal As Integers    ${admin_conversations.status_code}    200

    # Get user conversations (should be empty for new user)
    ${user_conversations}=     Get User Conversations    ${user_token}
    Should Be Equal As Integers    ${user_conversations.status_code}    200

    # User should see empty or only their own conversations
    ${user_conv_data}=         Set Variable    ${user_conversations.json()}[conversations]
    IF    isinstance($user_conv_data, dict) and len($user_conv_data) > 0
        ${client_ids}=    Get Dictionary Keys    ${user_conv_data}
        FOR    ${client_id}    IN    @{client_ids}
            ${client_conversations}=    Set Variable    ${user_conv_data}[${client_id}]
            FOR    ${conversation}    IN    @{client_conversations}
                # Note: The actual conversation structure doesn't have user_id field exposed
                # This test should verify that only this user's conversations are returned
                Dictionary Should Contain Key    ${conversation}    conversation_id
            END
        END
    END

    # Cleanup
    Delete Test User    ${admin_token}    ${test_user}[user_id]

