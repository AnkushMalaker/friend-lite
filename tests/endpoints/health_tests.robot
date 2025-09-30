*** Settings ***
Documentation    Health and Status Endpoint API Tests
Library          RequestsLibrary
Library          Collections
Resource         ../resources/setup_resources.robot
Resource         ../resources/user_resources.robot
Resource         ../resources/session_resources.robot
Suite Setup      Suite Setup
Suite Teardown   Delete All Sessions

*** Test Cases ***

Readiness Check Test
    [Documentation]    Test readiness check endpoint for container orchestration
    [Tags]             readiness    status    positive
    Get Anonymous Session    anon_session

    ${response}=    GET On Session    anon_session    /readiness
    Should Be Equal As Integers    ${response.status_code}    200

    ${readiness}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${readiness}    status
    Dictionary Should Contain Key    ${readiness}    timestamp
    Should Be Equal        ${readiness}[status]    ready

Health Check Test
    [Documentation]    Test main health check endpoint
    [Tags]             health    status    positive
    # Get Anonymous Session
    Create API Session    health_check_session  
    ${response}=    GET On Session    health_check_session    /health
    Should Be Equal As Integers    ${response.status_code}    200

    ${health}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${health}    status
    Dictionary Should Contain Key    ${health}    timestamp
    Dictionary Should Contain Key    ${health}    services
    Dictionary Should Contain Key    ${health}    overall_healthy
    Dictionary Should Contain Key    ${health}    critical_services_healthy
    
    ${services}=    Set Variable    ${health}[services]
    Log To Console    \n
    Log To Console    Mongodb: ${services}[mongodb][status]    
    Log To Console    AudioAI: ${services}[audioai][status]
    Log To Console    Memory Service: ${services}[memory_service][status]
    Log To Console    Speech to Text: ${services}[speech_to_text][status]
    Log To Console    Speaker recognition: ${services}[speaker_recognition][status]
    # Verify status is one of expected values
    Should Be True    '${health}[status]' in ['healthy', 'degraded', 'critical']
    
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

Auth Health Check Test
    [Documentation]    Test authentication service health check
    [Tags]             auth    health    positive
    Get Anonymous Session    session

    ${response}=    GET On Session    session   /api/auth/health
    Should Be Equal As Integers    ${response.status_code}    200

    ${auth_health}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${auth_health}    status
    Dictionary Should Contain Key    ${auth_health}    database
    Dictionary Should Contain Key    ${auth_health}    memory_service
    Dictionary Should Contain Key    ${auth_health}    timestamp

Queue Health Check Test
    [Documentation]    Test queue system health check
    [Tags]             queue    health    positive
    Get Anonymous Session    session

    ${response}=    GET On Session    session    /api/queue/health
    Should Be Equal As Integers    ${response.status_code}    200

    ${queue_health}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${queue_health}    status
    Dictionary Should Contain Key    ${queue_health}    worker_running
    Dictionary Should Contain Key    ${queue_health}    message

Chat Health Check Test
    [Documentation]    Test chat service health check
    [Tags]             chat    health    positive
    Get Anonymous Session    session

    ${response}=    GET On Session    session    /api/chat/health
    Should Be Equal As Integers    ${response.status_code}    200

    ${chat_health}=    Set Variable    ${response.json()}
    Dictionary Should Contain Key    ${chat_health}    status
    Dictionary Should Contain Key    ${chat_health}    service
    Dictionary Should Contain Key    ${chat_health}    timestamp
    Should Be Equal        ${chat_health}[service]    chat

System Metrics Test
    [Documentation]    Test system metrics endpoint (admin only)
    [Tags]             metrics    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/metrics
    Should Be Equal As Integers    ${response.status_code}    200

    ${metrics}=        Set Variable    ${response.json()}
    # Metrics structure may vary, just verify it's a valid response
    Should Be True     isinstance($metrics, dict)

Processor Status Test
    [Documentation]    Test processor status endpoint (admin only)
    [Tags]             processor    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/processor/status
    Should Be Equal As Integers    ${response.status_code}    200

    ${status}=         Set Variable    ${response.json()}
    # Processor status structure may vary
    Should Be True     isinstance($status, dict)

Processing Tasks Test
    [Documentation]    Test processing tasks endpoint (admin only)
    [Tags]             processor    tasks    admin    positive
    Get Anonymous Session    anon_session

    Create API Session    admin_session
    ${response}=       GET On Session    admin_session    /api/processor/tasks
    Should Be Equal As Integers    ${response.status_code}    200

    ${tasks}=          Set Variable    ${response.json()}
    Should Be True     isinstance($tasks, (dict, list))

Health Check Service Details Test
    [Documentation]    Test detailed service health information
    [Tags]             health    services    detailed
    Get Anonymous Session    session
    ${response}=    GET On Session    session    /health
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
    Get Anonymous Session    session

    Create API Session    admin_session

    # Create a non-admin user
    ${test_user}=      Create Test User    admin_session    test-user-${RANDOM_ID}@example.com    test-password-123
    Create API Session    user_session    email=test-user-${RANDOM_ID}@example.com    password=test-password-123

    # Metrics endpoint should be forbidden
    ${response}=       GET On Session    user_session    /api/metrics    expected_status=403
    Should Be Equal As Integers    ${response.status_code}    403

    # Processor status should be forbidden
    ${response}=       GET On Session    user_session    /api/processor/status    expected_status=403
    Should Be Equal As Integers    ${response.status_code}    403

    # Processing tasks should be forbidden
    ${response}=       GET On Session    user_session    /api/processor/tasks    expected_status=403
    Should Be Equal As Integers    ${response.status_code}    403

    # Cleanup
    Delete Test User    ${test_user}[user_id]

Unauthorized Health Access Test
    [Documentation]    Test health endpoints that require authentication
    [Tags]             health    security    negative
    Get Anonymous Session    session

    # Admin-only endpoints should require authentication
    ${response}=    GET On Session    session    /api/metrics    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

    ${response}=    GET On Session    session    /api/processor/status    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

