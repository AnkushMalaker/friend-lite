*** Settings ***
Documentation    Client and Queue Management API Tests
Library          RequestsLibrary
Library          Collections
Resource         resources/setup_resources.robot
Resource         resources/auth_keywords.robot
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions

*** Test Cases ***

Get Active Clients Test
    [Documentation]    Test getting active client information
    [Tags]             client    active    positive
    Setup Auth Session

    ${token}=      Get Admin Token
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=   GET On Session    api    /api/clients/active    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${clients}=    Set Variable    ${response.json()}
    Should Be True    isinstance($clients, (dict, list))

    # Structure depends on implementation - may be dict with client info or list
    IF    isinstance($clients, list)
        FOR    ${client}    IN    @{clients}
            Should Be True    isinstance($client, dict)
        END
    END

Get Queue Jobs Test
    [Documentation]    Test getting queue jobs with pagination
    [Tags]             queue    jobs    positive
    Setup Auth Session

    ${token}=      Get Admin Token
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}
    &{params}=     Create Dictionary    limit=20    offset=0

    ${response}=   GET On Session    api    /api/queue/jobs    headers=${headers}    params=${params}
    Should Be Equal As Integers    ${response.status_code}    200

    ${result}=     Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${result}    jobs
    Dictionary Should Contain Key    ${result}    pagination

    ${jobs}=       Set Variable    ${result}[jobs]
    Should Be True    isinstance($jobs, list)

    ${pagination}=    Set Variable    ${result}[pagination]
    Dictionary Should Contain Key    ${pagination}    total
    Dictionary Should Contain Key    ${pagination}    limit
    Dictionary Should Contain Key    ${pagination}    offset
    Dictionary Should Contain Key    ${pagination}    has_more

Get Queue Jobs With Different Limits Test
    [Documentation]    Test queue jobs pagination with different limits
    [Tags]             queue    jobs    pagination    positive
    Setup Auth Session

    ${token}=      Get Admin Token
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    # Test with small limit
    &{params1}=    Create Dictionary    limit=5    offset=0
    ${response1}=  GET On Session    api    /api/queue/jobs    headers=${headers}    params=${params1}
    Should Be Equal As Integers    ${response1.status_code}    200

    # Test with larger limit
    &{params2}=    Create Dictionary    limit=50    offset=0
    ${response2}=  GET On Session    api    /api/queue/jobs    headers=${headers}    params=${params2}
    Should Be Equal As Integers    ${response2.status_code}    200

    ${result1}=    Set Variable    ${response1.json()}
    ${result2}=    Set Variable    ${response2.json()}
    ${count1}=     Get Length    ${result1}[jobs]
    ${count2}=     Get Length    ${result2}[jobs]

    # Second request should have >= first request count
    Should Be True    ${count2} >= ${count1}

Get Queue Statistics Test
    [Documentation]    Test getting queue statistics
    [Tags]             queue    statistics    positive
    Setup Auth Session

    ${token}=      Get Admin Token
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    ${response}=   GET On Session    api    /api/queue/stats    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${stats}=      Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${stats}    queued
    Dictionary Should Contain Key    ${stats}    processing
    Dictionary Should Contain Key    ${stats}    completed
    Dictionary Should Contain Key    ${stats}    failed

    # All counts should be non-negative
    Should Be True    ${stats}[queued] >= 0
    Should Be True    ${stats}[processing] >= 0
    Should Be True    ${stats}[completed] >= 0
    Should Be True    ${stats}[failed] >= 0

Get Queue Health Test
    [Documentation]    Test getting queue health status
    [Tags]             queue    health    positive
    Setup Auth Session

    ${response}=   GET On Session    api    /api/queue/health
    Should Be Equal As Integers    ${response.status_code}    200

    ${health}=     Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${health}    status
    Dictionary Should Contain Key    ${health}    worker_running
    Dictionary Should Contain Key    ${health}    message

    # Status should be one of expected values
    Should Be True    '${health}[status]' in ['healthy', 'stopped', 'unhealthy']

Queue Jobs User Isolation Test
    [Documentation]    Test that regular users only see their own queue jobs
    [Tags]             queue    security    isolation
    Setup Auth Session

    ${admin_token}=    Get Admin Token

    # Create a test user
    ${test_user}=      Create Test User    ${admin_token}    test-user-${RANDOM_ID}@example.com    test-password-123
    ${user_token}=     Get User Token      test-user-${RANDOM_ID}@example.com    test-password-123

    # Get user's jobs (should be filtered to their user_id)
    &{headers}=        Create Dictionary    Authorization=Bearer ${user_token}
    ${response}=       GET On Session    api    /api/queue/jobs    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${result}=         Set Variable    ${response.json()}
    ${jobs}=           Set Variable    ${result}[jobs]

    # All jobs should belong to the test user
    FOR    ${job}    IN    @{jobs}
        IF    'user_id' in $job
            Should Be Equal    ${job}[user_id]    ${test_user}[id]
        END
    END

    # Cleanup
    Delete Test User    ${admin_token}    ${test_user}[id]

Invalid Queue Parameters Test
    [Documentation]    Test queue endpoints with invalid parameters
    [Tags]             queue    negative    validation
    Setup Auth Session

    ${token}=      Get Admin Token
    &{headers}=    Create Dictionary    Authorization=Bearer ${token}

    # Test with invalid limit (too high)
    &{params}=     Create Dictionary    limit=1000    offset=0
    ${response}=   GET On Session    api    /api/queue/jobs    headers=${headers}    params=${params}    expected_status=422
    Should Be Equal As Integers    ${response.status_code}    422

    # Test with negative offset
    &{params}=     Create Dictionary    limit=20    offset=-1
    ${response}=   GET On Session    api    /api/queue/jobs    headers=${headers}    params=${params}    expected_status=422
    Should Be Equal As Integers    ${response.status_code}    422

    # Test with invalid limit (too low)
    &{params}=     Create Dictionary    limit=0    offset=0
    ${response}=   GET On Session    api    /api/queue/jobs    headers=${headers}    params=${params}    expected_status=422
    Should Be Equal As Integers    ${response.status_code}    422

Unauthorized Client Access Test
    [Documentation]    Test that client endpoints require authentication
    [Tags]             client    security    negative
    Setup Auth Session

    # Try to access active clients without token
    ${response}=    GET On Session    api    /api/clients/active    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

Unauthorized Queue Access Test
    [Documentation]    Test that queue endpoints require authentication
    [Tags]             queue    security    negative
    Setup Auth Session

    # Try to access queue jobs without token
    ${response}=    GET On Session    api    /api/queue/jobs    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

    # Try to access queue stats without token
    ${response}=    GET On Session    api    /api/queue/stats    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

Queue Health Public Access Test
    [Documentation]    Test that queue health endpoint is publicly accessible
    [Tags]             queue    health    public
    Setup Auth Session

    # Queue health should be accessible without authentication
    ${response}=   GET On Session    api    /api/queue/health
    Should Be Equal As Integers    ${response.status_code}    200

    ${health}=     Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${health}    status

Client Manager Integration Test
    [Documentation]    Test client manager functionality
    [Tags]             client    manager    integration
    Setup Auth Session

    ${admin_token}=    Get Admin Token

    # Get active clients (may be empty)
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}
    ${response}=       GET On Session    api    /api/clients/active    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${clients}=        Set Variable    ${response.json()}
    # Verify structure - should be a valid JSON response
    Should Be True     isinstance($clients, (dict, list))

