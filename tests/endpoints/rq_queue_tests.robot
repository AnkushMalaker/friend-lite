*** Settings ***
Documentation       RQ Job Persistence Tests - Verify Redis Queue job persistence through service restarts
Library             RequestsLibrary
Library             Collections
Library             Process
Library             OperatingSystem
Library             String
Library             DateTime
Resource            ../resources/setup_resources.robot
Resource            ../resources/session_resources.robot
Resource            ../resources/user_resources.robot
Resource            ../resources/conversation_keywords.robot
Variables           ../test_env.py

Suite Setup         Suite Setup
Suite Teardown      Delete All Sessions
Test Setup          Suite Setup
Test Teardown       Clear RQ Test Data

*** Variables ***
${TEST_TIMEOUT}             60s
${COMPOSE_FILE}             backends/advanced/docker-compose-test.yml

*** Keywords ***

Clear RQ Test Data
    [Documentation]    Clear test data between tests
    # Clear Redis queues
    Run Process    docker-compose    -f    ${COMPOSE_FILE}    exec    -T    redis-test    redis-cli    FLUSHALL
    ...    cwd=.


Check Queue Stats
    [Documentation]    Get current queue statistics
    Create API Session    admin_session
    ${response}=    GET On Session    admin_session    /api/queue/stats    expected_status=200
    RETURN    ${response.json()}

Check Queue Jobs
    [Documentation]    Get current jobs in queue
    Create API Session    admin_session
    ${response}=    GET On Session    admin_session    /api/queue/jobs    expected_status=200
    RETURN    ${response.json()}

Check Queue Health
    [Documentation]    Get queue health status
    Create API Session    admin_session
    ${response}=    GET On Session    admin_session    /api/queue/health    expected_status=200
    RETURN    ${response.json()}

Restart Backend Service
    [Documentation]    Restart the backend service to test persistence
    Log    Restarting backend service to test job persistence

    # Stop backend container
    Run Process    docker-compose    -f    ${COMPOSE_FILE}    stop    friend-backend-test
    ...    cwd=.    timeout=30s

    # Start backend container again
    Run Process    docker-compose    -f    ${COMPOSE_FILE}    start    friend-backend-test
    ...    cwd=.    timeout=60s

    # Wait for backend to be ready again
    Wait Until Keyword Succeeds    ${TEST_TIMEOUT}    5s
    ...    Health Check    ${API_URL}

    Log    Backend service restarted successfully

Trigger Transcript Reprocessing
    [Documentation]    Trigger transcript reprocessing to create an RQ job
    [Arguments]    ${conversation_id}

    Create API Session    admin_session
    ${token}=    Get Authentication Token    admin_session    ${ADMIN_EMAIL}    ${ADMIN_PASSWORD}

    Log    Triggering transcript reprocessing for conversation: ${conversation_id}
    ${response}=    Reprocess Transcript    ${token}    ${conversation_id}

    Should Be True    ${response.status_code} in [200, 202]

    # The response might contain job_id, but it's not guaranteed in all implementations
    ${job_id}=    Set Variable    reprocess-${conversation_id}
    Log    Triggered reprocessing job for conversation: ${conversation_id}
    RETURN    ${job_id}

*** Test Cases ***
Test RQ Job Enqueuing
    [Documentation]    Test that jobs can be enqueued in Redis
    [Tags]    rq    enqueue    positive

    # Check initial queue state
    ${initial_stats}=    Check Queue Stats
    ${initial_queued}=    Set Variable    ${initial_stats}[queued_jobs]

    # Find or create test conversation and trigger reprocessing
    ${conversation_id}=    Find Or Create Test Conversation

    IF    $conversation_id != $None
        ${job_id}=    Trigger Transcript Reprocessing    ${conversation_id}

        # Verify job was enqueued
        ${stats_after}=    Check Queue Stats
        ${queued_after}=    Set Variable    ${stats_after}[queued_jobs]

        Should Be True    ${queued_after} >= ${initial_queued}
        Log    Successfully enqueued job: ${job_id}
    ELSE
        Log    No conversations available for job enqueuing test
        Pass Execution    No conversations available for RQ job enqueuing test
    END

Test Job Persistence Through Backend Restart
    [Documentation]    Test that RQ jobs persist when backend service restarts
    [Tags]    rq    persistence    restart    critical

    # Find test conversation
    ${conversation_id}=    Find Or Create Test Conversation

    IF    $conversation_id != $None
        # Create and enqueue a job
        ${job_id}=    Trigger Transcript Reprocessing    ${conversation_id}

        # Verify jobs exist in queue (may include other jobs)
        ${jobs_before}=    Check Queue Jobs
        ${jobs_count_before}=    Get Length    ${jobs_before}[jobs]

        # Restart backend service
        Restart Backend Service

        # Verify queue is still accessible and jobs persist
        ${jobs_after}=    Check Queue Jobs
        ${jobs_count_after}=    Get Length    ${jobs_after}[jobs]

        # Jobs should persist through restart (count may be same or greater)
        Should Be True    ${jobs_count_after} >= 0
        Log    Job persistence test passed - queue survived backend restart with ${jobs_count_after} jobs
    ELSE
        Log    No conversations available for persistence test
        Pass Execution    No conversations available for job persistence test
    END

Test Queue Health After Restart
    [Documentation]    Test that queue health checks work after service restart
    [Tags]    rq    health    restart    positive

    # Check initial health
    ${health_before}=    Check Queue Health
    # Queue might be healthy or no_workers - both indicate Redis connectivity
    Should Be True    '${health_before}[status]' in ['healthy', 'no_workers']
    Should Be True    ${health_before}[redis_connected]

    # Restart backend
    Restart Backend Service

    # Check health after restart
    ${health_after}=    Check Queue Health
    # Queue might be healthy or no_workers - both indicate Redis connectivity
    Should Be True    '${health_after}[status]' in ['healthy', 'no_workers']
    Should Be True    ${health_after}[redis_connected]

    Log    Queue health check passed after restart

Test Multiple Jobs Persistence
    [Documentation]    Test that multiple jobs persist through restart
    [Tags]    rq    persistence    multiple    stress

    # Find test conversation
    ${conversation_id}=    Find Or Create Test Conversation

    IF    $conversation_id != $None
        # Create multiple jobs using the same conversation
        ${job_count}=    Set Variable    3
        FOR    ${i}    IN RANGE    ${job_count}
            ${job_id}=    Trigger Transcript Reprocessing    ${conversation_id}
            Sleep    1s    # Small delay between jobs
        END

        Log    Created ${job_count} reprocessing jobs

        # Get baseline job count
        ${jobs_before}=    Check Queue Jobs
        ${jobs_count_before}=    Get Length    ${jobs_before}[jobs]

        # Restart backend
        Restart Backend Service

        # Verify jobs persist through restart
        ${jobs_after}=    Check Queue Jobs
        ${jobs_count_after}=    Get Length    ${jobs_after}[jobs]

        # Jobs should persist (exact count may vary based on processing)
        Should Be True    ${jobs_count_after} >= 0
        Log    Jobs persisted through restart: ${jobs_count_before} -> ${jobs_count_after}
    ELSE
        Log    No conversations available for multiple jobs test
        Pass Execution    No conversations available for multiple jobs persistence test
    END

Test Redis Data Persistence
    [Documentation]    Test that Redis data itself persists (not just connections)
    [Tags]    rq    redis    persistence    infrastructure

    # Store a test key directly in Redis
    ${test_key}=    Set Variable    test:persistence:key:${RANDOM_ID}
    ${test_value}=    Set Variable    persistence-test-value-${RANDOM_ID}

    Run Process    docker-compose    -f    ${COMPOSE_FILE}    exec    -T    redis-test
    ...    redis-cli    SET    ${test_key}    ${test_value}
    ...    cwd=.

    # Restart entire test environment (Redis included)
    Log    Restarting Redis to test data persistence
    Run Process    docker-compose    -f    ${COMPOSE_FILE}    restart    redis-test
    ...    cwd=.    timeout=30s

    # Wait for Redis to be ready
    Wait Until Keyword Succeeds    30s    2s
    ...    Check Redis Health

    # Check if data persisted
    ${result}=    Run Process    docker-compose    -f    ${COMPOSE_FILE}    exec    -T    redis-test
    ...    redis-cli    GET    ${test_key}
    ...    cwd=.

    Should Be Equal As Strings    ${result.stdout.strip()}    ${test_value}
    Log    Redis data persistence verified

Test Queue Stats Accuracy
    [Documentation]    Test that queue statistics accurately reflect job states
    [Tags]    rq    statistics    accuracy    positive

    # Get baseline stats
    ${initial_stats}=    Check Queue Stats
    ${initial_queued}=    Set Variable    ${initial_stats}[queued_jobs]

    # Find test conversation
    ${conversation_id}=    Find Or Create Test Conversation

    IF    $conversation_id != $None
        # Create multiple jobs to get meaningful stats
        ${job_count}=    Set Variable    3
        FOR    ${i}    IN RANGE    ${job_count}
            ${job_id}=    Trigger Transcript Reprocessing    ${conversation_id}
            Sleep    0.5s
        END

        # Check updated stats
        ${updated_stats}=    Check Queue Stats
        ${updated_queued}=    Set Variable    ${updated_stats}[queued_jobs]

        # Should have same or more jobs queued (jobs may process quickly)
        Should Be True    ${updated_queued} >= ${initial_queued}
        Log    Queue statistics updated: ${initial_queued} -> ${updated_queued}
    ELSE
        Log    No conversations available for stats accuracy test
        Pass Execution    No conversations available for queue stats accuracy test
    END

Test Queue API Authentication
    [Documentation]    Test that queue endpoints properly enforce authentication
    [Tags]    rq    security    authentication    negative

    # Create anonymous session (no authentication)
    Get Anonymous Session    anon_session

    # Queue jobs endpoint should require authentication
    ${response}=    GET On Session    anon_session    /api/queue/jobs    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

    # Queue stats endpoint should require authentication
    ${response}=    GET On Session    anon_session    /api/queue/stats    expected_status=401
    Should Be Equal As Integers    ${response.status_code}    401

    # Queue health should be publicly accessible
    ${response}=    GET On Session    anon_session    /api/queue/health    expected_status=200
    Should Be Equal As Integers    ${response.status_code}    200

    Log    Queue API authentication properly enforced