*** Settings ***
Documentation    Health and Status Endpoint API Tests
Library          RequestsLibrary
Library          Collections
Library          BuiltIn
Resource         resources/setup_resources.robot
Resource         resources/auth_keywords.robot
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions

*** Test Cases ***

Health Check Test
    [Documentation]    Test main health check endpoint
    [Tags]             health    status    positive
    Setup Auth Session

    ${response}=    GET On Session    api    /health
    Should Be Equal As Integers    ${response.status_code}    200

    ${health}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${health}    status
    Dictionary Should Contain Key    ${health}    timestamp
    Dictionary Should Contain Key    ${health}    services
    Dictionary Should Contain Key    ${health}    overall_healthy
    Dictionary Should Contain Key    ${health}    critical_services_healthy

    # Verify status is one of expected values
    Should Be True    '${health}[status]' in ['healthy', 'degraded', 'critical']

Readiness Check Test
    [Documentation]    Test readiness check endpoint for container orchestration
    [Tags]             readiness    status    positive
    Setup Auth Session

    ${response}=    GET On Session    api    /readiness
    Should Be Equal As Integers    ${response.status_code}    200

    ${readiness}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${readiness}    status
    Dictionary Should Contain Key    ${readiness}    timestamp
    Should Be Equal        ${readiness}[status]    ready

Auth Health Check Test
    [Documentation]    Test authentication service health check
    [Tags]             auth    health    positive
    Setup Auth Session

    ${response}=    GET On Session    api    /api/auth/health
    Should Be Equal As Integers    ${response.status_code}    200

    ${auth_health}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${auth_health}    status
    Dictionary Should Contain Key    ${auth_health}    database
    Dictionary Should Contain Key    ${auth_health}    memory_service
    Dictionary Should Contain Key    ${auth_health}    timestamp

Queue Health Check Test
    [Documentation]    Test queue system health check
    [Tags]             queue    health    positive
    Setup Auth Session

    ${response}=    GET On Session    api    /api/queue/health
    Should Be Equal As Integers    ${response.status_code}    200

    ${queue_health}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${queue_health}    status
    Dictionary Should Contain Key    ${queue_health}    worker_running
    Dictionary Should Contain Key    ${queue_health}    message

Chat Health Check Test
    [Documentation]    Test chat service health check
    [Tags]             chat    health    positive
    Setup Auth Session

    ${response}=    GET On Session    api    /api/chat/health
    Should Be Equal As Integers    ${response.status_code}    200

    ${chat_health}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${chat_health}    status
    Dictionary Should Contain Key    ${chat_health}    service
    Dictionary Should Contain Key    ${chat_health}    timestamp
    Should Be Equal        ${chat_health}[service]    chat

System Metrics Test
    [Documentation]    Test system metrics endpoint (admin only)
    [Tags]             metrics    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}

    ${response}=       GET On Session    api    /api/metrics    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${metrics}=        Set Variable    ${response.json()}
    # Metrics structure may vary, just verify it's a valid response
    Should Be True     isinstance($metrics, dict)

Processor Status Test
    [Documentation]    Test processor status endpoint (admin only)
    [Tags]             processor    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}

    ${response}=       GET On Session    api    /api/processor/status    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${status}=         Set Variable    ${response.json()}
    # Processor status structure may vary
    Should Be True     isinstance($status, dict)

Processing Tasks Test
    [Documentation]    Test processing tasks endpoint (admin only)
    [Tags]             processor    tasks    admin    positive
    Setup Auth Session

    ${admin_token}=    Get Admin Token
    &{headers}=        Create Dictionary    Authorization=Bearer ${admin_token}

    ${response}=       GET On Session    api    /api/processor/tasks    headers=${headers}
    Should Be Equal As Integers    ${response.status_code}    200

    ${tasks}=          Set Variable    ${response.json()}
    Should Be True     isinstance($tasks, (dict, list))

Health Check Service Details Test
    [Documentation]    Test detailed service health information
    [Tags]             health    services    detailed
    Setup Auth Session

    ${response}=    GET On Session    api    /health
    Should Be Equal As Integers    ${response.status_code}    200

    ${health}=    Set Variable    ${response.json()}
    ${services}=    Set Variable    ${health}[services]

    # Check for expected services
    ${expected_services}=    Create List    mongodb    audioai    memory_service    speech_to_text

    FOR    ${service}    IN    @{expected_services}
        IF    '${service}' in $services
            ${service_info}=    Set Variable    ${services}[${service}]
            Dictionary Should Contain Key    ${service_info}    status
            Dictionary Should Contain Key    ${service_info}    healthy
            Dictionary Should Contain Key    ${service_info}    critical
        END
    END

Non-Admin Cannot Access Admin Endpoints Test
    [Documentation]    Test that non-admin users cannot access admin health endpoints
    [Tags]             health    security    negative
    Setup Auth Session

    ${admin_token}=    Get Admin Token

    # Create a non-admin user
    ${test_user}=      Create Test User    ${admin_token}    test-user-${RANDOM_ID}@example.com    test-password-123
    ${user_token}=     Get User Token      test-user-${RANDOM_ID}@example.com    test-password-123

    # Try to access admin endpoints
    &{headers}=        Create Dictionary    Authorization=Bearer ${user_token}

    # Metrics endpoint should be forbidden
    ${response}=       GET On Session    api    /api/metrics    headers=${headers}    expected_status=403
    Should Be Equal As Integers    ${response.status_code}    403

    # Processor status should be forbidden
    ${response}=       GET On Session    api    /api/processor/status    headers=${headers}    expected_status=403
    Should Be Equal As Integers    ${response.status_code}    403

    # Processing tasks should be forbidden
    ${response}=       GET On Session    api    /api/processor/tasks    headers=${headers}    expected_status=403
    Should Be Equal As Integers    ${response.status_code}    403

    # Cleanup
    Delete Test User    ${admin_token}    ${test_user}[user_id]

Unauthorized Health Access Test
    [Documentation]    Test health endpoints that require authentication
    [Tags]             health    security    negative
    Setup Auth Session

    # Admin-only endpoints should require authentication
    ${response}=    GET On Session    api    /api/metrics    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

    ${response}=    GET On Session    api    /api/processor/status    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

Health Configuration Test
    [Documentation]    Test that health check returns configuration information
    [Tags]             health    config    positive
    Setup Auth Session

    ${response}=    GET On Session    api    /health
    Should Be Equal As Integers    ${response.status_code}    200

    ${health}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${health}    config

    ${config}=    Set Variable    ${health}[config]
    Dictionary Should Contain Key    ${config}    mongodb_uri
    Dictionary Should Contain Key    ${config}    qdrant_url
    Dictionary Should Contain Key    ${config}    transcription_service
    Dictionary Should Contain Key    ${config}    asr_uri
    Dictionary Should Contain Key    ${config}    provider_type
    Dictionary Should Contain Key    ${config}    chunk_dir
    Dictionary Should Contain Key    ${config}    active_clients
    Dictionary Should Contain Key    ${config}    new_conversation_timeout_minutes
    Dictionary Should Contain Key    ${config}    audio_cropping_enabled
    Dictionary Should Contain Key    ${config}    llm_provider
    Dictionary Should Contain Key    ${config}    llm_model
    Dictionary Should Contain Key    ${config}    llm_base_url

    # Verify config values are not empty
    Should Not Be Empty    ${config}[mongodb_uri]
    Should Not Be Empty    ${config}[qdrant_url]
    Should Not Be Empty    ${config}[transcription_service]
    Should Not Be Empty    ${config}[asr_uri]
    Should Not Be Empty    ${config}[provider_type]
    Should Not Be Empty    ${config}[chunk_dir]
    Should Be True        isinstance(${config}[active_clients], int)
    Should Be True        ${config}[new_conversation_timeout_minutes] > 0
    Should Be True        isinstance(${config}[audio_cropping_enabled], bool)
    Should Not Be Empty    ${config}[llm_provider]
    Should Not Be Empty    ${config}[llm_model]
    Should Not Be Empty    ${config}[llm_base_url]

