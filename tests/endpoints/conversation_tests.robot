*** Settings ***
Documentation    Conversation Management API Tests
Library          RequestsLibrary
Library          Collections
Library          String
Resource         ../resources/setup_resources.robot
Resource         ../resources/user_resources.robot
Resource         ../resources/conversation_keywords.robot
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions

*** Test Cases ***

Get User Conversations Test
    [Documentation]    Test getting conversations for authenticated user
    [Tags]             conversation    user    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${conversations_data}=        Get User Conversations

    # Verify conversation structure if any exist
   
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

    Create API Session    admin_session
    ${test_conversation}=    Find Test Conversation

    IF    $test_conversation != $None
        ${conversation_id}=    Set Variable    ${test_conversation}[conversation_id]
        ${conversation}=           Get Conversation By ID       ${conversation_id}

        # Verify conversation structure
        Dictionary Should Contain Key    ${conversation}    conversation_id
        Dictionary Should Contain Key    ${conversation}    audio_uuid
        Dictionary Should Contain Key    ${conversation}    created_at
        Should Be Equal    ${conversation}[conversation_id]    ${conversation_id}
    ELSE
        Log    No conversations available for testing
        Pass Execution    No conversations available for individual conversation test
    END

# Get Conversation Versions Test
#     [Documentation]    Test getting version history for a conversation
#     [Tags]             conversation    versions    positive

#     ${test_conversation}=    Find Test Conversation

#     IF    $test_conversation != $None
#         ${conversation_id}=    Set Variable    ${test_conversation}[conversation_id]
#         ${versions}=           Get Conversation Versions     ${conversation_id}

  
#         # Verify version history structure
#         Dictionary Should Contain Key    ${versions}    transcript_versions
#         Dictionary Should Contain Key    ${versions}    memory_versions
#         Dictionary Should Contain Key    ${versions}    active_transcript_version
#         Dictionary Should Contain Key    ${versions}    active_memory_version
#     ELSE
#         Log    No conversations available for testing
#         Pass Execution    No conversations available for version history test
#     END

Unauthorized Conversation Access Test
    [Documentation]    Test that conversation endpoints require authentication
    [Tags]             conversation    security    negative
    Get Anonymous Session    session

    # Try to access conversations without token
    ${response}=    GET On Session    session    /api/conversations    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

Non-Existent Conversation Test
    [Documentation]    Test accessing a non-existent conversation
    [Tags]             conversation    negative    notfound
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${fake_id}=         Set Variable    non-existent-conversation-id

    ${response}=        GET On Session    admin_session    /api/conversations/${fake_id}    expected_status=404
    Should Be Equal As Integers    ${response.status_code}    404

Reprocess Transcript Test
    [Documentation]    Test triggering transcript reprocessing
    [Tags]             conversation    reprocess    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${test_conversation}=    Find Test Conversation

    IF    $test_conversation != $None
        ${conversation_id}=    Set Variable    ${test_conversation}[conversation_id]
        ${response}=           Reprocess Transcript       ${conversation_id}

        # Reprocessing might return 200 (success) or 202 (accepted) depending on implementation
        Should Be True    ${response.status_code} in [200, 202]
    ELSE
        Log    No conversations available for reprocessing test
        Pass Execution    No conversations available for transcript reprocessing test
    END

Reprocess Memory Test
    [Documentation]    Test triggering memory reprocessing
    [Tags]             conversation    reprocess    memory    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${test_conversation}=    Find Test Conversation

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
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${client_id}=       Set Variable    test-client-${RANDOM_ID}

    # This might return 404 if client doesn't exist, which is expected
    ${response}=        POST On Session    admin_session    /api/conversations/${client_id}/close    expected_status=any
    Should Be True     ${response.status_code} in [200, 404]

Invalid Conversation Operations Test
    [Documentation]    Test invalid operations on conversations
    [Tags]             conversation    negative    invalid
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${fake_id}=         Set Variable    invalid-conversation-id

    # Test reprocessing non-existent conversation
    ${response}=        POST On Session    admin_session    /api/conversations/${fake_id}/reprocess-transcript    expected_status=404
    Should Be Equal As Integers    ${response.status_code}    404

    # Test getting versions of non-existent conversation
    ${response}=        GET On Session    admin_session    /api/conversations/${fake_id}/versions    expected_status=404
    Should Be Equal As Integers    ${response.status_code}    404

Version Management Test
    [Documentation]    Test version activation (if versions exist)
    [Tags]             conversation    versions    activation
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${test_conversation}=    Find Test Conversation

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
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Create a test user
    ${test_user}=       Create Test User    admin_session    test-user-${RANDOM_ID}@example.com    test-password-123
    Create API Session    user_session    email=test-user-${RANDOM_ID}@example.com    password=test-password-123

    # Get admin conversations
    ${admin_conversations}=    Get User Conversations
    Should Be Equal As Integers    ${admin_conversations.status_code}    200

    # Get user conversations (should be empty for new user)
    ${user_conversations}=     GET On Session    user_session    /api/conversations
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
    Delete Test User    ${test_user}[user_id]

