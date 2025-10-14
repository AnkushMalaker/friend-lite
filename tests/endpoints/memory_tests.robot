*** Settings ***
Documentation    Memory Management API Tests
Library          RequestsLibrary
Library          Collections
Library          String
Resource         ../resources/setup_resources.robot
Resource         ../resources/session_resources.robot
Resource         ../resources/user_resources.robot
Resource         ../resources/memory_keywords.robot
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions

*** Test Cases ***

Get User Memories Test
    [Documentation]    Test getting memories for authenticated user
    [Tags]             memory    user    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/memories

    Should Be Equal As Integers    ${response.status_code}    200
    Should Be True    isinstance($response.json(), list)

    # Verify memory structure if any exist
    ${memories}=    Set Variable    ${response.json()}
    FOR    ${memory}    IN    @{memories}
        # Verify memory structure
        Dictionary Should Contain Key    ${memory}    id
        Dictionary Should Contain Key    ${memory}    user_id
        Dictionary Should Contain Key    ${memory}    text
        Dictionary Should Contain Key    ${memory}    created_at
    END

Get Memories With Transcripts Test
    [Documentation]    Test getting memories with their source transcripts
    [Tags]             memory    transcripts    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/memories/with-transcripts

    Should Be Equal As Integers    ${response.status_code}    200
    Should Be True    isinstance($response.json(), list)

    # Verify enhanced structure if any exist
    ${memories}=    Set Variable    ${response.json()}
    FOR    ${memory}    IN    @{memories}
        # Verify memory structure
        Dictionary Should Contain Key    ${memory}    id
        Dictionary Should Contain Key    ${memory}    user_id
        Dictionary Should Contain Key    ${memory}    text
        Dictionary Should Contain Key    ${memory}    created_at
        # May have additional transcript fields
    END

Search Memories Test
    [Documentation]    Test searching memories by query
    [Tags]             memory    search    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    &{params}=         Create Dictionary    query=test    limit=20    score_threshold=0.0
    ${response}=       GET On Session    admin_session    /api/memories/search    params=${params}

    Should Be Equal As Integers    ${response.status_code}    200
    Should Be True    isinstance($response.json(), list)

    # Verify search results structure
    ${results}=    Set Variable    ${response.json()}
    # Verify search results structure
    FOR    ${memory}    IN    @{results}
        # Verify memory structure
        Dictionary Should Contain Key    ${memory}    id
        Dictionary Should Contain Key    ${memory}    user_id
        Dictionary Should Contain Key    ${memory}    text
        Dictionary Should Contain Key    ${memory}    created_at
    END

Search Memories With High Threshold Test
    [Documentation]    Test searching memories with high similarity threshold
    [Tags]             memory    search    threshold
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    &{params}=         Create Dictionary    query=nonexistent-query    limit=10    score_threshold=0.9
    ${response}=       GET On Session    admin_session    /api/memories/search    params=${params}

    Should Be Equal As Integers    ${response.status_code}    200
    Should Be True    isinstance($response.json(), list)

    # High threshold might return fewer results
    ${results}=    Set Variable    ${response.json()}
    FOR    ${memory}    IN    @{results}
        # Verify memory structure
        Dictionary Should Contain Key    ${memory}    id
        Dictionary Should Contain Key    ${memory}    user_id
        Dictionary Should Contain Key    ${memory}    text
        Dictionary Should Contain Key    ${memory}    created_at
    END

Get Unfiltered Memories Test
    [Documentation]    Test getting unfiltered memories for debugging
    [Tags]             memory    debug    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/memories/unfiltered

    Should Be Equal As Integers    ${response.status_code}    200
    Should Be True    isinstance($response.json(), list)

    # Unfiltered may include more memories than filtered
    ${memories}=    Set Variable    ${response.json()}
    FOR    ${memory}    IN    @{memories}
        # Verify memory structure
        Dictionary Should Contain Key    ${memory}    id
        Dictionary Should Contain Key    ${memory}    user_id
        Dictionary Should Contain Key    ${memory}    text
        Dictionary Should Contain Key    ${memory}    created_at
    END

Get All Memories Admin Test
    [Documentation]    Test getting all memories across users (admin only)
    [Tags]             memory    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/admin/memories

    Should Be Equal As Integers    ${response.status_code}    200
    Should Be True    isinstance($response.json(), list)

    # Admin endpoint should return memories from all users
    ${memories}=    Set Variable    ${response.json()}
    FOR    ${memory}    IN    @{memories}
        # Verify memory structure
        Dictionary Should Contain Key    ${memory}    id
        Dictionary Should Contain Key    ${memory}    user_id
        Dictionary Should Contain Key    ${memory}    text
        Dictionary Should Contain Key    ${memory}    created_at
        # Should have user_id from potentially different users
        Dictionary Should Contain Key    ${memory}    user_id
    END

Memory Pagination Test
    [Documentation]    Test memory pagination with different limits
    [Tags]             memory    pagination    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Test with small limit
    &{params1}=    Create Dictionary    limit=5
    ${response1}=  GET On Session    admin_session    /api/memories    params=${params1}
    Should Be Equal As Integers    ${response1.status_code}    200
    ${memories1}=  Set Variable    ${response1.json()}
    ${count1}=     Get Length    ${memories1}
    Should Be True    ${count1} <= 5

    # Test with larger limit
    &{params2}=    Create Dictionary    limit=100
    ${response2}=  GET On Session    admin_session    /api/memories    params=${params2}
    Should Be Equal As Integers    ${response2.status_code}    200
    ${memories2}=  Set Variable    ${response2.json()}
    ${count2}=     Get Length    ${memories2}

    # Second request should have >= first request count
    Should Be True    ${count2} >= ${count1}

Non-Admin Cannot Access Admin Memories Test
    [Documentation]    Test that non-admin users cannot access admin memory endpoint
    [Tags]             memory    security    negative
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Create a non-admin user
    ${test_user}=      Create Test User    admin_session    test-user-${RANDOM_ID}@example.com    test-password-123
    Create API Session    user_session    email=test-user-${RANDOM_ID}@example.com    password=test-password-123

    # Try to access admin memories endpoint
    ${response}=       GET On Session    user_session    /api/memories/admin    expected_status=403
    Should Be Equal As Integers    ${response.status_code}    403

    # Cleanup
    Delete Test User    ${test_user}[user_id]

Unauthorized Memory Access Test
    [Documentation]    Test that memory endpoints require authentication
    [Tags]             memory    security    negative
    Get Anonymous Session    session

    # Try to access memories without token
    ${response}=    GET On Session    session    /api/memories    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

    # Try to search memories without token
    &{params}=     Create Dictionary    query=test
    ${response}=   GET On Session    session    /api/memories/search    params=${params}    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

Invalid Search Parameters Test
    [Documentation]    Test search with invalid parameters
    [Tags]             memory    search    negative
    Get Anonymous Session    anon_session

    Create API Session    admin_session

    # Test with empty query (should fail)
    &{params}=     Create Dictionary    query=${EMPTY}
    ${response}=   GET On Session    admin_session    /api/memories/search    params=${params}    expected_status=422
    Should Be Equal As Integers    ${response.status_code}    422

    # Test with invalid score threshold
    &{params}=     Create Dictionary    query=test    score_threshold=2.0
    ${response}=   GET On Session    admin_session    /api/memories/search    params=${params}    expected_status=422
    Should Be Equal As Integers    ${response.status_code}    422

